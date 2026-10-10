"""Dependency-free regression tests for local-only GPU monitoring and guard isolation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    import runpod_gpu_monitor as monitor
    import runpod_gpu_telemetry as telemetry


def sample(clock=2700, util=99, uuid="GPU-fixture"):
    return {
        "uuid": uuid,
        "name": "NVIDIA GeForce RTX 5090",
        "pstate": "P2",
        "sm_mhz": clock,
        "graphics_mhz": clock,
        "memory_mhz": 14000,
        "max_sm_mhz": 3100,
        "application_sm_mhz": None,
        "utilization_pct": util,
        "temperature_c": 60,
        "power_w": 430,
        "power_limit_w": 575,
        "default_power_limit_w": 575,
        "reasons": dict.fromkeys(telemetry.REASONS, False),
        "counters_us": {
            "sw_power_cap": 9000000000,
            "sw_thermal_slowdown": 0,
            "hw_thermal_slowdown": 0,
            "hw_power_brake_slowdown": 0,
        },
    }


def nvidia_xml(group="clocks_event_reasons", reason_prefix="clocks_event_reason_"):
    reasons = "".join(
        f"<{reason_prefix}{key}>Not Active</{reason_prefix}{key}>" for key in telemetry.REASONS
    )
    return f"""<?xml version="1.0"?><!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">
<nvidia_smi_log><gpu><product_name>NVIDIA GeForce RTX 5090</product_name><uuid>GPU-fixture</uuid>
<performance_state>P2</performance_state><clocks><graphics_clock>2700 MHz</graphics_clock>
<sm_clock>2700 MHz</sm_clock><mem_clock>14000 MHz</mem_clock></clocks>
<max_clocks><sm_clock>3100 MHz</sm_clock></max_clocks>
<utilization><gpu_util>99 %</gpu_util></utilization>
<temperature><gpu_temp>60 C</gpu_temp></temperature>
<gpu_power_readings><power_draw>430.0 W</power_draw>
<current_power_limit>575.0 W</current_power_limit>
<default_power_limit>575.0 W</default_power_limit></gpu_power_readings>
<{group}>{reasons}</{group}>
<clocks_event_reasons_counters><sw_power_capping>9000000000 us</sw_power_capping>
<hw_thermal_slowdown>0 us</hw_thermal_slowdown>
</clocks_event_reasons_counters></gpu></nvidia_smi_log>"""


class TelemetryTests(unittest.TestCase):
    def test_real_blackwell_counter_tag_spellings(self):
        raw = nvidia_xml().replace(
            "<hw_thermal_slowdown>0 us</hw_thermal_slowdown>",
            "<clocks_event_reasons_counters_hw_therm_slowdown>12 us"
            "</clocks_event_reasons_counters_hw_therm_slowdown>"
            "<clocks_event_reasons_counters_sw_therm_slowdown>34 us"
            "</clocks_event_reasons_counters_sw_therm_slowdown>"
            "<clocks_event_reasons_counters_hw_power_brake>56 us"
            "</clocks_event_reasons_counters_hw_power_brake>",
        )
        value = telemetry.parse_nvidia_xml(raw)[0]
        self.assertEqual(value["counters_us"]["hw_thermal_slowdown"], 12)
        self.assertEqual(value["counters_us"]["sw_thermal_slowdown"], 34)
        self.assertEqual(value["counters_us"]["hw_power_brake_slowdown"], 56)

    def test_current_and_older_driver_xml_preserve_unknown_values(self):
        for group, prefix in (
            ("clocks_event_reasons", "clocks_event_reason_"),
            ("clocks_throttle_reasons", "clocks_throttle_reason_"),
        ):
            with self.subTest(group=group):
                value = telemetry.parse_nvidia_xml(nvidia_xml(group, prefix))[0]
                self.assertEqual(value["sm_mhz"], 2700)
                self.assertEqual(value["power_limit_w"], 575)
                self.assertEqual(value["counters_us"]["sw_power_cap"], 9000000000)
                self.assertIs(value["reasons"]["hw_slowdown"], False)
        value = telemetry.parse_nvidia_xml(nvidia_xml().replace("2700 MHz", "N/A"))[0]
        self.assertIsNone(value["sm_mhz"])

    def test_absent_reason_is_not_false(self):
        raw = nvidia_xml().replace(
            "<clocks_event_reason_hw_slowdown>Not Active</clocks_event_reason_hw_slowdown>", ""
        )
        value = telemetry.parse_nvidia_xml(raw)[0]
        self.assertNotIn("hw_slowdown", value["reasons"])

    def test_rejects_entity_expansion_oversize_and_non_gpu_documents(self):
        for raw in (
            "<!ENTITY x 'bad'>" + nvidia_xml(),
            "x" * (telemetry.MAX_OUTPUT_BYTES + 1),
            "<x/>",
            "<nvidia_smi_log/>",
        ):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                telemetry.parse_nvidia_xml(raw)

    def test_endpoint_validates_identity_and_allows_official_proxy(self):
        pod = {
            "id": "pod-a",
            "desiredStatus": "RUNNING",
            "ssh": {"direct": {"host": "127.0.0.1", "port": 12345, "username": "root"}},
        }
        self.assertEqual(telemetry.endpoint(pod, "pod-a"), ("127.0.0.1", 12345, "root"))
        with self.assertRaises(ValueError):
            telemetry.endpoint(pod, "pod-b")
        pod["ssh"] = {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "pod-a"}}
        self.assertEqual(telemetry.endpoint(pod, "pod-a")[0], "ssh.runpod.io")
        pod["ssh"]["proxy"]["host"] = "untrusted.example"
        with self.assertRaises(ValueError):
            telemetry.endpoint(pod, "pod-a")

    def test_bounded_call_reaps_timeout_and_limits_output(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            telemetry.bounded_call(
                [sys.executable, "-c", "import time; time.sleep(20)"], timeout=0.2
            )
        self.assertLess(time.monotonic() - started, 3)
        with self.assertRaises(ValueError):
            telemetry.bounded_call([sys.executable, "-c", "print('x'*1100000)"], timeout=3)

    def test_connection_owns_its_socket_and_only_queries_gpu(self):
        with tempfile.TemporaryDirectory() as name:
            key = Path(name) / "key"
            key.write_text("fixture, never read as a private key")
            pod = {
                "id": "pod-a",
                "desiredStatus": "RUNNING",
                "name": "experiment",
                "ssh": {"direct": {"host": "127.0.0.1", "port": 12345, "username": "root"}},
            }
            with patch.object(
                telemetry, "bounded_call", side_effect=[json.dumps(pod), nvidia_xml(), ""]
            ) as call:
                connection = telemetry.TelemetrySSH(ROOT, "pod-a", key)
                self.assertEqual(len(connection.sample()), 1)
                connection.close()
            command = call.call_args_list[1].args[0]
            self.assertEqual(command[-1], "nvidia-smi -q -x")
            self.assertIn("ForwardAgent=no", command)
            self.assertIn("BatchMode=yes", command)
            self.assertIn("ControlPersist=90", command)
            self.assertTrue(command[command.index("-S") + 1].startswith("/tmp/rpg-"))

    def test_invalid_proxy_reply_refreshes_to_direct_without_polling_api_each_time(self):
        for raw in ("Error: Your SSH client doesn't support PTY", "<not_gpu/>"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as name:
                key = Path(name) / "key"
                key.write_text("fixture, never read as a private key")
                proxy = {
                    "id": "pod-a", "desiredStatus": "RUNNING", "name": "experiment",
                    "ssh": {"proxy": {
                        "host": "ssh.runpod.io", "port": 22, "username": "pod-a",
                    }},
                }
                direct = copy.deepcopy(proxy)
                direct["ssh"]["direct"] = {
                    "host": "127.0.0.1", "port": 12345, "username": "root",
                }
                replies = iter((proxy, direct))
                now = [100.0]

                def reply(command, **kwargs):
                    if command[0] == "bash":
                        return json.dumps(next(replies))
                    if command[-1] == "nvidia-smi -q -x":
                        return raw if command[-2].endswith("@ssh.runpod.io") else nvidia_xml()
                    self.assertEqual(command[-3:-1], ["-O", "exit"])
                    return ""

                with patch.object(telemetry, "bounded_call", side_effect=reply) as call, \
                        patch.object(telemetry.time, "monotonic", side_effect=lambda: now[0]):
                    connection = telemetry.TelemetrySSH(ROOT, "pod-a", key)
                    try:
                        for elapsed in (0, 59, 60):
                            now[0] = 100.0 + elapsed
                            with self.assertRaises((ValueError, telemetry.ET.ParseError)):
                                connection.sample()
                            self.assertEqual(connection.base is None, elapsed == 60)
                        self.assertEqual(connection.sample()[0]["uuid"], "GPU-fixture")
                        self.assertEqual(connection.base[-1], "root@127.0.0.1")
                        self.assertEqual(
                            sum(item.args[0][0] == "bash" for item in call.call_args_list), 2
                        )
                    finally:
                        connection.close()


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = monitor.Detector(monitor.Settings())

    def warm(self):
        for now in range(0, 180, 30):
            self.detector.observe(sample(), now)

    def test_p2_and_below_max_clock_are_not_faults(self):
        self.warm()
        result = self.detector.observe(sample(), 180)
        self.assertEqual(result["state"], "no_throttling_observed")
        self.assertEqual(result["alerts"], [])

    def test_idle_and_checkpoint_gaps_do_not_trigger_low_clock_alert(self):
        self.warm()
        for now in range(180, 420, 30):
            result = self.detector.observe(sample(clock=210, util=0), now)
            self.assertEqual(result["state"], "idle_or_low_load")
            self.assertEqual(result["alerts"], [])

    def test_historical_counter_nonzero_is_not_a_current_event(self):
        for now in (0, 30, 60):
            result = self.detector.observe(sample(), now)
            self.assertEqual(result["alerts"], [])

    def test_hardware_reasons_are_recorded_but_not_pushed_without_clock_evidence(self):
        for reason in monitor.THERMAL_HARDWARE:
            value = sample()
            value["reasons"][reason] = True
            result = monitor.Detector(monitor.Settings()).observe(value, 0)
            self.assertIn(reason, result["alerts"])
            self.assertEqual(result["notification_alerts"], [])
            self.assertEqual(result["state"], "throttling_detected")

    def test_notification_requires_strictly_below_eighty_percent_for_most_window(self):
        self.warm()
        for index, clock in enumerate((2700, 2400, 2160)):
            for now in range(180 + index * 180, 360 + index * 180, 30):
                value = sample(clock=clock)
                value["reasons"]["hw_thermal_slowdown"] = True
                result = self.detector.observe(value, now)
                self.assertEqual(result["notification_alerts"], [])
        for now in range(720, 900, 30):
            result = self.detector.observe(sample(clock=2159), now)
            self.assertEqual(result["notification_alerts"], [])
        self.assertEqual(result["clock_low_fraction"], 0.5)
        result = self.detector.observe(sample(clock=2159), 900)
        self.assertEqual(result["notification_alerts"], ["suspected_low_clock"])
        self.assertLess(result["clock_ratio"], 0.8)
        self.assertEqual(result["reference_sm_mhz"], 2700)

    def test_mild_threshold_override_cannot_push_above_eighty_percent(self):
        self.detector = monitor.Detector(monitor.Settings(clock_drop_fraction=0.05))
        self.warm()
        for now in range(180, 420, 30):
            result = self.detector.observe(sample(clock=2400), now)
        self.assertEqual(result["notification_alerts"], [])
        self.assertEqual(result["notification_clock_ratio"], 0.8)

    def test_minority_short_dips_do_not_notify(self):
        self.warm()
        for now in range(180, 900, 30):
            result = self.detector.observe(sample(clock=1700 if now % 120 == 0 else 2700), now)
            self.assertEqual(result["notification_alerts"], [])

    def test_nonconsecutive_majority_dips_do_notify(self):
        self.warm()
        for now in range(180, 900, 30):
            result = self.detector.observe(sample(clock=1700 if now % 120 else 2700), now)
        self.assertGreater(result["clock_low_fraction"], 0.5)
        self.assertEqual(result["notification_alerts"], ["suspected_low_clock"])

    def test_sparse_activity_cannot_inflate_the_fraction_by_excluding_idle_time(self):
        self.warm()
        for now in range(180, 510, 30):
            result = self.detector.observe(sample(clock=300, util=0), now)
        for now in range(510, 690, 30):
            result = self.detector.observe(sample(clock=1700), now)
        self.assertEqual(result["clock_observed_seconds"], 150)
        self.assertEqual(result["clock_low_seconds"], 150)
        self.assertEqual(result["clock_low_fraction"], 0.5)
        self.assertEqual(result["notification_alerts"], [])
        result = self.detector.observe(sample(clock=1700), 690)
        self.assertEqual(result["notification_alerts"], ["suspected_low_clock"])

    def test_time_weighted_fraction_not_fraction_of_sample_count(self):
        self.warm()
        now = 180
        self.detector.observe(sample(), now)
        for _ in range(8):
            now += 15
            result = self.detector.observe(sample(), now)
        self.detector.observe(sample(clock=1700), now + 1)
        now += 1
        for _ in range(4):
            now += 45
            result = self.detector.observe(sample(clock=1700), now)
        self.assertAlmostEqual(result["clock_low_fraction"], 0.6)
        self.assertEqual(result["clock_observed_seconds"], 300)
        self.assertEqual(result["notification_alerts"], ["suspected_low_clock"])

    def test_counter_increment_catches_events_between_polls(self):
        value = sample()
        self.detector.observe(value, 0)
        value = copy.deepcopy(value)
        value["counters_us"]["hw_power_brake_slowdown"] = 1000000
        result = self.detector.observe(value, 30)
        self.assertIn("hw_power_brake_slowdown", result["alerts"])
        self.assertEqual(result["counter_deltas_us"]["hw_power_brake_slowdown"], 1000000)

    def test_counter_reset_and_long_disconnect_are_not_events(self):
        value = sample()
        value["counters_us"]["hw_thermal_slowdown"] = 5000
        self.detector.observe(value, 0)
        value["counters_us"] = {"hw_thermal_slowdown": 0}
        self.assertEqual(self.detector.observe(value, 30)["alerts"], [])
        value["counters_us"] = {"hw_thermal_slowdown": 99999}
        self.assertEqual(self.detector.observe(value, 300)["alerts"], [])

    def test_sustained_low_clock_alert_and_recovery_confirmation(self):
        self.warm()
        for now in range(180, 360, 30):
            result = self.detector.observe(sample(clock=1700), now)
            self.assertEqual(result["state"], "observing_low_clock")
            self.assertNotIn("suspected_low_clock", result["alerts"])
        result = self.detector.observe(sample(clock=1700), 360)
        self.assertIn("suspected_low_clock", result["alerts"])
        recovered = []
        for now in range(390, 900, 30):
            recovered.extend(self.detector.observe(sample(), now)["recovered"])
        self.assertIn("suspected_low_clock", recovered)

    def test_short_dip_and_disconnect_reset_confirmation(self):
        self.warm()
        self.detector.observe(sample(clock=1700), 180)
        self.detector.observe(sample(clock=1700), 210)
        self.detector.observe(sample(), 240)
        self.assertNotIn(
            "suspected_low_clock", self.detector.observe(sample(clock=1700), 270)["alerts"]
        )
        self.assertNotIn(
            "suspected_low_clock", self.detector.observe(sample(clock=1700), 600)["alerts"]
        )

    def test_power_cap_is_information_unless_clock_drop_is_sustained(self):
        self.warm()
        value = sample()
        value["reasons"]["sw_power_cap"] = True
        result = self.detector.observe(value, 180)
        self.assertEqual(result["state"], "power_limited")
        self.assertEqual(result["alerts"], [])
        value["sm_mhz"] = 1700
        for now in range(210, 420, 30):
            result = self.detector.observe(value, now)
        self.assertIn("suspected_low_clock", result["alerts"])

    def test_missing_telemetry_never_becomes_healthy(self):
        value = sample()
        value["reasons"] = {}
        self.assertEqual(self.detector.observe(value, 0)["state"], "limited_telemetry")
        value["sm_mhz"] = value["graphics_mhz"] = None
        self.assertEqual(self.detector.observe(value, 30)["state"], "limited_telemetry")

    def test_persistent_missing_fields_alert_and_recovery_requires_data(self):
        value = sample()
        value["reasons"] = {}
        for now in range(0, 150, 30):
            result = self.detector.observe(value, now)
        self.assertEqual(result["state"], "limited_telemetry")
        self.assertIn("telemetry_incomplete", result["alerts"])
        for now in range(150, 300, 30):
            result = self.detector.observe(sample(), now)
        self.assertIn("telemetry_incomplete", result["recovered"])

    def test_normal_power_capping_can_establish_clock_reference(self):
        value = sample()
        value["reasons"]["sw_power_cap"] = True
        for now in range(0, 180, 30):
            result = self.detector.observe(value, now)
        self.assertEqual(result["state"], "power_limited")
        self.assertEqual(result["reference_sm_mhz"], 2700)
        self.assertEqual(result["alerts"], [])

    def test_initial_low_power_limit_has_a_distinct_warning(self):
        value = sample()
        value["power_limit_w"] = 250
        self.assertIn(
            "configured_power_limit_below_default", self.detector.observe(value, 0)["alerts"]
        )

    def test_devices_have_separate_bounded_baselines(self):
        self.warm()
        result = self.detector.observe(sample(clock=1400, uuid="GPU-other"), 180)
        self.assertEqual(result["alerts"], [])
        for now in range(210, 3000, 30):
            self.detector.observe(sample(), now)
        self.assertEqual(len(self.detector.devices["GPU-fixture"]["baseline"]), 20)
        self.detector.retain({"GPU-other"})
        self.assertEqual(set(self.detector.devices), {"GPU-other"})


class LocalControlTests(unittest.TestCase):
    def test_inspection_denied_is_unknown_not_a_dead_process(self):
        with (
            patch.object(monitor.os, "kill", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            monitor.alive(12345)

    def test_start_is_idempotent_and_only_passes_monitor_environment(self):
        with tempfile.TemporaryDirectory() as name:
            log = Path(name) / "guard.log"
            identity = {
                "pid": 12345,
                "pod_id": "pod-a",
                "run_id": "run-a",
                "state": "armed",
                "armed_at": "fixture",
                "host_boot_id": "fixture",
            }
            files = monitor.paths(log)
            monitor.atomic_json(files["ready"], identity)
            args = argparse.Namespace(
                guard_log=log, guard_pid=12345, pod_id="pod-a", run_id="run-a"
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "RUNPOD_GPU_MONITOR": "1",
                        "SSH_AUTH_SOCK": "/tmp/fixture.sock",
                        "TRAINING_SECRET": "must-not-be-inherited",
                    },
                ),
                patch.object(monitor.subprocess, "Popen") as child,
            ):
                child.return_value.poll.return_value = None
                monitor.start(args)
                environment = child.call_args.kwargs["env"]
                self.assertEqual(environment["SSH_AUTH_SOCK"], "/tmp/fixture.sock")
                self.assertNotIn("TRAINING_SECRET", environment)
                monitor.atomic_json(
                    files["status"],
                    {"guard_identity": identity, "pid": os.getpid(), "state": "monitoring"},
                )
                monitor.start(args)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(monitor.read_json(files["ready"]), identity)

    def test_watch_recovers_after_transport_failure_and_stops_with_guard(self):
        with tempfile.TemporaryDirectory() as name:
            log = Path(name) / "guard.log"
            identity = {
                "pid": 12345,
                "pod_id": "pod-a",
                "run_id": "run-a",
                "state": "armed",
                "armed_at": "fixture",
                "host_boot_id": "fixture",
            }
            monitor.atomic_json(monitor.paths(log)["ready"], identity)
            args = argparse.Namespace(
                guard_log=log, guard_pid=12345, pod_id="pod-a", run_id="run-a"
            )
            fault = sample()
            fault["reasons"]["hw_thermal_slowdown"] = True
            with (
                patch.object(monitor, "guard_matches", side_effect=[True, True, False]),
                patch.object(
                    monitor.Settings,
                    "from_env",
                    return_value=monitor.Settings(poll_seconds=0.01, startup_grace_seconds=0),
                ),
                patch.object(monitor, "TelemetrySSH") as transport,
                patch.object(monitor, "signal"),
            ):
                transport.return_value.name = "pod-a-name"
                transport.return_value.sample.side_effect = [
                    TimeoutError("bounded timeout"),
                    [fault],
                ]
                with patch.object(monitor.Reporter, "alert") as alerts:
                    monitor.monitor(args)
                keys = [call.args[0] for call in alerts.call_args_list]
                self.assertIn("monitor_unavailable", keys)
                self.assertIn("connection_recovered", keys)
                self.assertNotIn("GPU-fixture:hw_thermal_slowdown", keys)
                transport.return_value.close.assert_called_once()
            result = monitor.read_json(monitor.paths(log)["status"])
            self.assertEqual(result["state"], "stopped")
            self.assertEqual(result["gpus"][0]["state"], "throttling_detected")

    def test_unexpected_monitor_failure_is_visible_and_does_not_delete_anything(self):
        with tempfile.TemporaryDirectory() as name:
            log = Path(name) / "guard.log"
            identity = {"pid": 12345, "pod_id": "pod-a", "run_id": "run-a", "state": "armed"}
            monitor.atomic_json(monitor.paths(log)["ready"], identity)
            args = argparse.Namespace(
                guard_log=log, guard_pid=12345, pod_id="pod-a", run_id="run-a"
            )
            with (
                patch.object(
                    monitor, "guard_matches", side_effect=RuntimeError("inspection denied")
                ),
                patch.object(monitor, "TelemetrySSH") as transport,
                patch.object(monitor, "signal"),
            ):
                with self.assertRaisesRegex(RuntimeError, "inspection denied"):
                    monitor.monitor(args)
                transport.return_value.close.assert_called_once()
            result = monitor.read_json(monitor.paths(log)["status"])
            self.assertEqual(result["state"], "failed")
            self.assertIn("inspection denied", result["error"])

    def test_settings_reject_invalid_and_nan_values(self):
        for value in ("0", "-1", "nan", "inf", "9999"):
            with (
                patch.dict(os.environ, {"RUNPOD_GPU_MONITOR_POLL_SECONDS": value}),
                self.assertRaises(ValueError),
            ):
                monitor.Settings.from_env()

    def test_notifications_are_argument_data_and_failures_are_nonfatal(self):
        with (
            patch.object(monitor.sys, "platform", "darwin"),
            patch.object(monitor, "bounded_call") as call,
        ):
            monitor.notify('title"', 'message" & do shell script "not a command"')
        command = call.call_args.args[0]
        self.assertNotIn("not a command", command[2])
        self.assertIn("not a command", command[-1])
        with tempfile.TemporaryDirectory() as name:
            reporter = monitor.Reporter(
                Path(name) / "events.jsonl",
                monitor.Settings(),
                notifier=lambda *_: (_ for _ in ()).throw(RuntimeError("denied")),
            )
            reporter.alert("key", "title", "message", 0)
            self.assertIn("failed", reporter.notification_status)
            reporter.close()

    def test_notification_cooldown_and_bounded_rotation(self):
        calls = []
        with tempfile.TemporaryDirectory() as name:
            reporter = monitor.Reporter(
                Path(name) / "events.jsonl",
                monitor.Settings(),
                notifier=lambda *args: calls.append(args) or "submitted",
            )
            for now in (0, 30, 899, 900):
                reporter.alert("key", "title", "message", now)
            self.assertEqual(len(calls), 2)
            reporter.handler.maxBytes = 300
            for _ in range(30):
                reporter.record("sample", value="x" * 150)
            reporter.close()
            self.assertLessEqual(len(list(Path(name).glob("events.jsonl*"))), 3)

    def test_stale_status_and_run_filter_are_read_only(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            payload = {
                "kind": monitor.KIND,
                "project_root": str(ROOT),
                "pod_id": "pod-a",
                "run_id": "run-a",
                "pid": os.getpid(),
                "state": "monitoring",
                "settings": {"poll_seconds": 30},
                "updated_at": time.time() - 300,
                "gpus": [],
            }
            path = directory / "pod-a.gpu-monitor.json"
            monitor.atomic_json(path, payload)
            before = path.read_bytes()
            self.assertEqual(monitor.status_records(directory, run_id="run-b"), [])
            self.assertEqual(monitor.status_records(directory, run_id="run-a")[0]["state"], "stale")
            self.assertEqual(path.read_bytes(), before)

    def test_guard_identity_pid_reuse_and_rearm_are_checked(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "ready.json"
            identity = {
                "pid": os.getpid(),
                "pod_id": "pod-a",
                "run_id": "run-a",
                "state": "armed",
                "armed_at": "2026-10-09T00:00:00Z",
                "host_boot_id": "fixture",
            }
            monitor.atomic_json(path, identity)
            with patch.object(
                monitor,
                "bounded_call",
                return_value="bash /tmp/terminate_runpod_after.sh pod-a 3600 /tmp/a.log",
            ):
                self.assertTrue(monitor.guard_matches(path, identity))
            with patch.object(monitor, "bounded_call", return_value="unrelated-command"):
                self.assertFalse(monitor.guard_matches(path, identity))
            monitor.atomic_json(path, {**identity, "armed_at": "later"})
            self.assertFalse(monitor.guard_matches(path, identity))

    def test_monitor_failure_cannot_disarm_guard_or_delete_pod(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            scripts = root / "scripts"
            (scripts / "lib").mkdir(parents=True)
            for relative in (
                "launch_runpod_guard.sh",
                "lib/runpod_project_env.sh",
                "lib/runpod_paths.sh",
            ):
                shutil.copyfile(ROOT / "scripts" / relative, scripts / relative)
            (root / ".env").write_text("RUNPOD_API_KEY=fixture-only\n")
            (root / ".env").chmod(0o600)
            (scripts / "runpodctl_project.sh").write_text(
                'echo unexpected-delete >> "$0.calls"\nexit 1\n'
            )
            (scripts / "runpod_gpu_monitor.py").write_text("raise SystemExit(55)\n")
            (scripts / "terminate_runpod_after.sh").write_text("""#!/usr/bin/env bash
printf '{"state":"armed","pid":%s,"pod_id":"%s"}\\n' "$$" "$1" > "$RUNPOD_GUARD_READY_FILE"
sleep 20
""")
            environment = {
                **os.environ,
                "RUNPOD_ENV_FILE": str(root / ".env"),
                "RUNPOD_GUARD_KEEP_AWAKE": "0",
                "RUNPOD_GUARD_RUN_ID": "run-fixture",
                "RUNPOD_GUARD_SOFT_LIMIT_SECONDS": "0",
                "RUNPOD_GUARD_SECTION_SAFE": "0",
            }
            result = subprocess.run(
                [
                    "bash",
                    str(scripts / "launch_runpod_guard.sh"),
                    "pod-fixture",
                    "20",
                    "lifecycle/runs/run-fixture/training.json",
                    str(root / "guard.log"),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pid = int(result.stdout.strip())
            try:
                self.assertTrue(monitor.alive(pid))
                self.assertIn("remains armed", result.stderr)
                self.assertFalse((scripts / "runpodctl_project.sh.calls").exists())
            finally:
                os.kill(pid, 15)

    def test_public_gpu_status_does_not_require_credentials_or_cloud(self):
        with tempfile.TemporaryDirectory() as name:
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/runpod_workflow.sh"),
                    "gpu-status",
                    "--guard-dir",
                    name,
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"monitors": []})

    def test_no_training_dataset_baseline_paths_in_monitor(self):
        for name in ("runpod_gpu_monitor.py", "runpod_gpu_telemetry.py"):
            source = (ROOT / "scripts" / name).read_text()
            for forbidden in (
                "stock_forecasting.models",
                "stock_forecasting.data",
                "pod delete",
                "--power-limit",
                "--lock-gpu-clocks",
                "cpu prepare",
            ):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
