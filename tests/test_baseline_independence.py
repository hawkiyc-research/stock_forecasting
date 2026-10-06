"""Dependency-free regression tests for baseline data/model lifecycle separation."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GATE = runpy.run_path(str(ROOT / "scripts/runpod_baseline_readiness.py"))
CONTRACT = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
SELECTION = GATE["SELECTION"]


def selection(start="2021-01-01"):
    args = SELECTION["build_parser"]().parse_args(
        [
            "create",
            "--project-root",
            str(ROOT),
            "--stage",
            "stage2",
            "--data-profile",
            "us_tw_eodhd",
            "--start",
            start,
            "--end",
            "2026-06-01",
            "--universe",
            "all",
        ]
    )
    return SELECTION["_build_selection"](args, ROOT)


def encode(payload):
    return json.dumps(payload, sort_keys=True).encode()


def metadata(path, data, **extra):
    return {
        "relative_path": path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **extra,
    }


def fixture(selected):
    request = selected["dataset_request"]
    prefix = f"datasets/{selected['dataset_request_sha256']}/"
    storage = GATE["DATA"]["dataset_request_identity_payload"](request)["storage_preparation"]
    code, _ = GATE["READINESS"]["_project_content_identity"](ROOT)
    content = GATE["READINESS"]["_dataset_content_identity_from_code"](
        {"data_content_identity": code}, request["selected_datasets"]
    )
    counts = {"train": 300, "validation": 100, "test": 100}
    fixed = request["preparation"]["fixed_split"]
    audit = {
        "schema_version": "causal-split-audit-v1",
        "policy": GATE["DATA"]["FIXED_SPLIT_POLICY"],
        "comparison": "label.end_at < next_split_start",
        "violations": 0,
        "fixed_split": fixed,
        "label_end_counts": counts,
        "splits": {},
        "maximum_label_end": {},
        "dates_by_market": {"US": {}},
        "validation_start": "2025-06-02T00:00:00+00:00",
        "test_start": "2025-12-01T00:00:00+00:00",
    }
    starts = {
        "train": request["date_range"]["start_inclusive"],
        "validation": "2025-06-02",
        "test": "2025-12-01",
    }

    def iso(day):
        return day.isoformat() + "T00:00:00+00:00"

    for split in counts:
        boundary = fixed[f"{split}_end"]
        first = date.fromisoformat(starts[split])
        last = date.fromisoformat(boundary) - timedelta(days=25)
        label_end = date.fromisoformat(boundary) - timedelta(days=1)
        audit[f"{split}_boundary_exclusive"] = iso(date.fromisoformat(boundary))
        audit["maximum_label_end"][split] = iso(label_end)
        audit["splits"][split] = {
            "cutoff_start_at": iso(first),
            "cutoff_end_at": iso(last),
            "label_end_max_at": iso(label_end),
            "records": counts[split],
        }
        days = [(first + timedelta(days=i)).isoformat() for i in range(80)]
        audit["dates_by_market"]["US"][split] = {
            "cutoff_dates": days,
            "unique_cutoff_count": len(days),
        }
    index = metadata("symbol-index.parquet", b"symbol-index", row_count=1)
    ranges = metadata("cutoff-ranges.parquet", b"cutoff-ranges", row_count=3)
    raw = metadata("raw/market.parquet", b"raw-ohlcv", row_count=1000)
    shard = metadata(
        "shards/bucket-0000/shard.parquet", b"ohlcv-shard", row_count=1000, row_groups=1
    )
    identity = {
        **storage,
        "kind": GATE["DATA"]["BAR_STORE_KIND"],
        "raw_sha256": raw["sha256"],
        "materialization_digest": content["bar_store_materialization_digest"],
    }
    bar = {
        "schema_version": GATE["DATA"]["BAR_STORE_SCHEMA_VERSION"],
        "state": "ready",
        "kind": GATE["DATA"]["BAR_STORE_KIND"],
        "identity": identity,
        "identity_sha256": GATE["digest"](identity),
        "split_counts": counts,
        "split_audit": audit,
        "symbol_index": index,
        "cutoff_ranges": ranges,
        "shards": [shard],
        "row_count": 1000,
        "symbol_count": 1,
    }
    common = {
        "dataset_profile": request["profile"],
        "selected_datasets": request["selected_datasets"],
        "date_range": request["date_range"],
        "training_security_scope": request["preparation"]["training_security_scope"],
    }
    download = {
        **common,
        "api_policy": {
            "symbol_limit": None,
            "include_delisted": request["universe"]["include_delisted_us"],
            "cache_revision": request["revision"],
        },
    }
    dataset = {
        **common,
        "schema_version": "4.0",
        "kind": "ohlcv-bar-store-dataset",
        "state": "ready",
        "storage_preparation_spec": storage,
        "storage_preparation_spec_sha256": GATE["digest"](storage),
        "data_content_identity": content,
        "data_pipeline_digest": GATE["digest"](content),
        "split_counts": counts,
        "split_audit": audit,
        "download_manifest": metadata("download-manifest.json", encode(download)),
        "artifacts": {
            "raw": raw,
            "bar_store_manifest": metadata("prepared/bar-store/bar-store.json", encode(bar)),
            "symbol_index": {**index, "relative_path": "prepared/bar-store/symbol-index.parquet"},
            "cutoff_ranges": {
                **ranges,
                "relative_path": "prepared/bar-store/cutoff-ranges.parquet",
            },
        },
    }
    success = {
        "schema_version": GATE["DATA"]["BAR_STORE_SCHEMA_VERSION"],
        "kind": "bar-store-success",
        "state": "ready",
        "identity_sha256": bar["identity_sha256"],
        "split_counts": counts,
        "bar_store_manifest_sha256": dataset["artifacts"]["bar_store_manifest"]["sha256"],
        "symbol_index_sha256": index["sha256"],
        "cutoff_ranges_sha256": ranges["sha256"],
    }
    return {
        prefix + name: value
        for name, value in {
            "dataset-manifest.json": encode(dataset),
            "download-manifest.json": encode(download),
            "prepared/bar-store/bar-store.json": encode(bar),
            "prepared/bar-store/_SUCCESS.json": encode(success),
            "prepared/bar-store/symbol-index.parquet": b"symbol-index",
            "prepared/bar-store/cutoff-ranges.parquet": b"cutoff-ranges",
            "prepared/bar-store/shards/bucket-0000/shard.parquet": b"ohlcv-shard",
        }.items()
    }


class MemoryReader(GATE["ArtifactReader"]):
    """Exercise the actual remote read/size validator over a deterministic S3 fixture."""

    def __init__(self, objects):
        super().__init__(ROOT, bucket="fixture-volume")
        self.objects, self.calls = objects, []

    def s3(self, *args):
        self.calls.append(args)
        if args[:2] == ("s3api", "head-object"):
            return str(len(self.objects[args[args.index("--key") + 1]])).encode()
        if args[:2] == ("s3", "cp") and args[3] == "-":
            return self.objects[args[2].split("/", 3)[3]]
        raise AssertionError(f"Unexpected mutation or download: {args}")


class BaselineIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.selected = selection()
        self.objects = fixture(self.selected)

    def verify(self, objects=None, selected=None):
        reader = MemoryReader(self.objects if objects is None else objects)
        result = GATE["verify_data"](ROOT, selected or self.selected, reader, workers=2)
        self.assertNotIn("hf-models", repr(reader.calls))
        self.assertNotIn("stage1/dataset.json", repr(reader.calls))
        return result

    def test_ab_switches_repeatedly_without_global_marker_or_model_cache(self):
        other = selection("2016-01-01")
        objects = {**self.objects, **fixture(other), "cache/hf-models.json": b"invalid json"}
        for selected in (self.selected, other, self.selected, other):
            result = self.verify(objects, selected)
            self.assertEqual(result["dataset_request_sha256"], selected["dataset_request_sha256"])
            self.assertEqual(result["state"], "ready")

    def test_corrupt_or_partial_data_still_fails(self):
        prefix = f"datasets/{self.selected['dataset_request_sha256']}/"
        for suffix in (
            "bar-store.json",
            "_SUCCESS.json",
            "symbol-index.parquet",
            "shards/bucket-0000/shard.parquet",
        ):
            objects = dict(self.objects)
            objects[prefix + "prepared/bar-store/" + suffix] = b"truncated"
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.verify(objects)

    def test_wrong_dates_storage_counts_and_leakage_are_rejected(self):
        key = f"datasets/{self.selected['dataset_request_sha256']}/dataset-manifest.json"
        for field, value in (
            ("date_range", {"start_inclusive": "1990-01-01", "end_exclusive": "2026-06-01"}),
            ("split_counts", {"train": 3, "validation": 2, "test": 1}),
            ("state", "preparing"),
            ("storage_preparation_spec_sha256", "0" * 64),
        ):
            objects = dict(self.objects)
            payload = json.loads(objects[key])
            payload[field] = value
            objects[key] = encode(payload)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(objects)
        payload = json.loads(self.objects[key])
        payload["split_audit"]["violations"] = 1
        with self.assertRaises(ValueError):
            self.verify({**self.objects, key: encode(payload)})

    def test_mounted_checksum_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            for key, data in self.objects.items():
                path = volume / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            reader = GATE["ArtifactReader"](ROOT, volume=volume)
            result = GATE["verify_data"](ROOT, self.selected, reader, workers=2)
            self.assertEqual(result["verification"], "checksums")
            shard = next(volume.glob("datasets/*/prepared/bar-store/shards/*/shard.parquet"))
            shard.write_bytes(b"x" * shard.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                GATE["verify_data"](ROOT, self.selected, reader, workers=2)

    def test_main_model_settings_do_not_change_identity_or_baseline_runtime(self):
        parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        before = CONTRACT["baseline_contract"](ROOT, self.selected)
        payload = CONTRACT["baseline_runtime_payload"](
            self.selected, parameters, Path("/runpod-volume")
        )
        changed = copy.deepcopy(self.selected)
        changed["stage"] = {
            "name": "stage1",
            "config_path": "missing.yaml",
            "config_sha256": "0" * 64,
            "feature_mode": "baseline",
        }
        self.assertEqual(before, CONTRACT["baseline_contract"](ROOT, changed))
        self.assertEqual(
            payload,
            CONTRACT["baseline_runtime_payload"](changed, parameters, Path("/runpod-volume")),
        )
        self.assertEqual(payload["data"]["train_fraction"], 1.0)
        self.assertIsNone(payload["data"]["max_samples"])
        CONTRACT["validate_optimization_alignment"](
            SimpleNamespace(
                **{
                    name: SimpleNamespace(**payload[name])
                    for name in ("data", "training", "validation")
                }
            ),
            parameters,
        )

    def test_stale_main_config_does_not_block_data_selection_but_tampering_does(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            path = project / "selection.json"
            path.write_bytes(encode(self.selected))
            # No main-model YAML exists in this project fixture.
            _, selected = SELECTION["_resolve_selection_path"](
                project, str(path), validate_local_config=False
            )
            self.assertEqual(selected, self.selected)
            with self.assertRaises(ValueError):
                SELECTION["_resolve_selection_path"](project, str(path))
            changed = copy.deepcopy(self.selected)
            changed["dataset_request"]["date_range"]["start_inclusive"] = "1990-01-01"
            path.write_bytes(encode(changed))
            with self.assertRaises(ValueError):
                SELECTION["_resolve_selection_path"](
                    project, str(path), validate_local_config=False
                )

    def test_data_environment_checks_ignore_main_model_only_fields(self):
        path = Path("/runpod-volume/lifecycle/selections/fixture.json")
        environment = SELECTION["_selection_exports"](path, self.selected)
        environment["NETWORK_VOLUME_ROOT"] = "/runpod-volume"
        environment["RUNPOD_REMOTE_SELECTION_PATH"] = str(path)
        for key in (
            "RUNPOD_STAGE",
            "RUNPOD_CONFIG",
            "RUNPOD_STAGE_CONFIG_SHA256",
            "FIN_TS_FEATURE_MODE",
        ):
            environment.pop(key)
        SELECTION["_verify_environment"](path, self.selected, environment, data_only=True)
        environment["DATA_ROOT"] = "/runpod-volume/datasets/wrong"
        with self.assertRaises(ValueError):
            SELECTION["_verify_environment"](path, self.selected, environment, data_only=True)

    def test_baseline_cache_survives_missing_main_yaml_without_reading_hf(self):
        cache = runpy.run_path(str(ROOT / "scripts/runpod_baseline_cache.py"))
        resolver = SELECTION["_resolve_selection_path"]

        def resolve(project, path, *, validate_local_config=True):
            self.assertFalse(validate_local_config)
            return Path("unused"), self.selected

        with patch.dict(SELECTION, {"_resolve_selection_path": resolve}):
            original_run = runpy.run_path

            def load(path):
                return SELECTION if path.endswith("runpod_selection.py") else original_run(path)

            with patch.object(runpy, "run_path", side_effect=load), patch("sys.stdout"):
                self.assertEqual(cache["main"](["identity"]), 0)
        self.assertIs(SELECTION["_resolve_selection_path"], resolver)

    def test_entrypoints_route_to_baseline_specific_gate(self):
        local = (ROOT / "scripts/create_runpod_pod.sh").read_text()
        remote = (ROOT / "scripts/runpod_baseline.sh").read_text()
        self.assertIn('verify_runpod_stage_readiness.sh" --baseline', local)
        self.assertIn('verify_runpod_mounted_readiness.sh" --baseline', remote)
        self.assertNotIn("CONFIG_PATH", remote)
        self.assertNotIn("--config", remote)
        cli = (ROOT / "src/stock_forecasting/cli/build_baselines.py").read_text()
        self.assertNotIn("from_yaml", cli)
        self.assertIn("baseline_runtime_payload", cli)

    def test_main_model_reuses_artifacts_without_optimizer_or_architecture_alignment(self):
        identity = CONTRACT["baseline_contract"](ROOT, self.selected)
        counts = {"train": 300, "validation": 100, "test": 100}
        membership = {"samples": counts["test"], "sha256": "fixture-membership"}
        scales = [1.0] * 14
        metrics = {
            "sample_membership": membership,
            "samples": counts["test"],
            "evaluation_robust_scales": scales,
        }
        payload = {
            "state": "complete",
            "identity": identity,
            "sample_counts": counts,
            "evaluation_membership": membership,
            "validation_membership": {"samples": 100},
            "robust_scales": scales,
            "models": {},
            "artifacts": {},
        }
        for name in identity["contract"]["parameters"]["models"]:
            learned = name in {"gru", "dlinear", "patchtst", "gbdt"}
            seeds = identity["contract"]["parameters"]["seeds"] if learned else [None]
            payload["models"][name] = {"state": "complete"}
            if learned:
                payload["models"][name]["seed_results"] = {str(seed): metrics for seed in seeds}
            else:
                payload["models"][name]["metrics"] = metrics
            for seed in seeds:
                job = f"jobs/{name}-{seed}" if learned else f"jobs/rules/{name}"
                weight = "model.pkl" if name == "gbdt" else "model.pt" if learned else "model.json"
                for suffix in (
                    weight,
                    "validation-metrics.json",
                    "test/predictions.npy",
                    "validation/predictions.npy",
                ):
                    payload["artifacts"][f"{job}/{suffix}"] = {"bytes": 1}
        payload["evaluation_data"] = CONTRACT["shared_evaluation_paths"]()
        for paths in payload["evaluation_data"].values():
            for relative in paths.values():
                payload["artifacts"][relative] = {"bytes": 1}
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            bar = volume / "prepared"
            bar.mkdir()
            (bar / "bar-store.json").write_bytes(encode({"split_counts": counts}))
            digest = hashlib.sha256((bar / "bar-store.json").read_bytes()).hexdigest()
            payload["data_identity"] = {
                "manifest_sha256": digest,
                "split_sources": {
                    split: {
                        "manifest_sha256": digest,
                        "sample_universe": "fixture-universe",
                        "samples": count,
                    }
                    for split, count in counts.items()
                },
            }
            root = volume / "baselines" / identity["baseline_id"]
            for relative in payload["artifacts"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            (root / "complete.json").write_bytes(encode(payload))
            consumer = CONTRACT["require_baselines"]

            class FixtureSource:
                root = bar
                sample_universe_identity = "fixture-universe"

                def __init__(self, split):
                    self.split = split

                def __len__(self):
                    return counts[self.split]

                def close(self):
                    pass

            dependencies = {
                "stock_forecasting.data.sample_universe": SimpleNamespace(
                    open_clean_dataset=lambda config, split, **kwargs: FixtureSource(split),
                    validated_split_roots=lambda config: dict.fromkeys(counts, bar),
                ),
                "stock_forecasting.data.manifest": SimpleNamespace(
                    sha256_file=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                ),
                "stock_forecasting.training_paths": SimpleNamespace(
                    resolve_bar_store_path=lambda path: path
                ),
            }
            # A consumer with no optimizer/model fields can reuse completed baselines.
            config = SimpleNamespace(data=SimpleNamespace(bar_store_path=bar))
            with (
                patch.dict(os.environ, {"NETWORK_VOLUME_ROOT": str(volume)}),
                patch.dict(sys.modules, dependencies),
                patch.dict(consumer.__globals__, {"runtime_contract": lambda: (ROOT, identity)}),
            ):
                self.assertEqual(consumer(config), payload)
                (bar / "bar-store.json").write_bytes(encode({"split_counts": {}}))
                with self.assertRaisesRegex(ValueError, "differs"):
                    consumer(config)

    def test_real_local_and_mounted_shell_gates_with_changed_main_yaml(self):
        with tempfile.TemporaryDirectory(prefix="baseline-gates-") as directory:
            volume = Path(directory).resolve() / "volume"
            project = volume / "stock_forecasting"
            for name in ("src", "scripts", "configs"):
                shutil.copytree(
                    ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            objects = {**self.objects, **fixture(selection("2016-01-01"))}
            for key, data in objects.items():
                path = volume / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            SELECTION["_activate_selection"](project, self.selected)
            remote = volume / "lifecycle/selections" / (self.selected["selection_id"] + ".json")
            remote.parent.mkdir(parents=True, exist_ok=True)
            remote.write_bytes(encode(self.selected))
            # Deliberately invalid main-model YAML with a different digest.
            (project / self.selected["stage"]["config_path"]).write_text("invalid: [\n")
            (project / ".env").write_text("RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
            (project / ".env").chmod(0o600)
            transport = project / "scripts/fixture_transport.py"
            transport.write_text("""import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["FAKE_VOLUME"])
with (root / "requests.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["s3", "cp"] and args[3] == "-":
    sys.stdout.buffer.write((root / args[2].split("/", 3)[3]).read_bytes())
elif args[:2] == ["s3api", "head-object"]:
    print((root / args[args.index("--key") + 1]).stat().st_size)
elif args[:2] == ["s3api", "list-objects-v2"]:
    key = args[args.index("--prefix") + 1]
    target = root / key
    paths = sorted(target.rglob("*")) if target.is_dir() else [target]
    entries = [{"Key": str(path.relative_to(root)), "Size": path.stat().st_size,
                "ETag": hashlib.sha256(path.read_bytes()).hexdigest(),
                "LastModified": str(path.stat().st_mtime_ns)}
               for path in paths if path.is_file()]
    print(json.dumps({"Contents": entries, "IsTruncated": False}))
else:
    raise RuntimeError("Unexpected mutation or cloud action")
""")
            (project / "scripts/runpod_s3_project.sh").write_text(
                "#!/usr/bin/env bash\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(transport))
                + ' "$@"\n'
            )
            buffer = io.StringIO()
            paths = sorted(
                str(path.relative_to(project))
                for path in project.rglob("*")
                if path.is_file()
                and path.suffix in (".py", ".sh", ".yaml", ".json")
                and ".runpod" not in path.parts
            )
            with contextlib.redirect_stdout(buffer):
                GATE["READINESS"]["command_code_manifest"](
                    SimpleNamespace(
                        project_root=project,
                        paths=paths,
                        pipeline_path=[],
                        state="ready",
                        remote_project_dir=str(project),
                    )
                )
            marker = volume / "lifecycle/stage1/code.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(buffer.getvalue())
            # The global dataset marker deliberately refers to neither selection.
            (marker.parent / "dataset.json").write_text('{"state":"preparing"}')
            binary = volume / "bin"
            binary.mkdir()
            (binary / "python3").symlink_to(sys.executable)
            (binary / "mountpoint").write_text("#!/usr/bin/env bash\nexit 0\n")
            (binary / "mountpoint").chmod(0o700)
            environment = {
                **os.environ,
                "PATH": f"{binary}:{os.defpath}",
                "FAKE_VOLUME": str(volume),
                "RUNPOD_ENV_FILE": str(project / ".env"),
            }

            def command(script, *args):
                return subprocess.run(
                    ["bash", str(project / "scripts" / script), *args],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            result = command("verify_runpod_stage_readiness.sh", "--baseline")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"state": "ready_for_baseline_build"', result.stdout)
            self.assertIn('"cleaning_state": "pending_build"', result.stdout)
            self.assertIn("NOT verified yet", result.stdout)
            self.assertNotIn("Baseline gate passed", result.stdout)

            def artifact_head_count():
                requests = [json.loads(line) for line in
                            (volume / "requests.jsonl").read_text().splitlines()]
                return sum(args[:2] == ["s3api", "head-object"]
                           and args[args.index("--key") + 1].endswith(".parquet")
                           for args in requests)

            before = artifact_head_count()
            self.assertEqual(before, 6)
            result = command("verify_runpod_stage_readiness.sh", "--baseline")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Baseline readiness reused", result.stdout)
            self.assertNotIn('"sources":', result.stdout)
            self.assertEqual(artifact_head_count(), before)

            shard = next(volume.glob("datasets/*/prepared/bar-store/shards/*/shard.parquet"))
            original = shard.read_bytes()
            shard.write_bytes(b"")
            result = command("verify_runpod_stage_readiness.sh", "--baseline")
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Baseline readiness reused", result.stdout)
            shard.write_bytes(original)
            # Main training keeps its strict model-config checks.
            result = command("verify_runpod_stage_readiness.sh", "--gpu")
            self.assertNotEqual(result.returncode, 0)
            environment.update(SELECTION["_selection_exports"](remote, self.selected))
            environment.update(
                {
                    "NETWORK_VOLUME_ROOT": str(volume),
                    "PROJECT_ROOT": str(project),
                    "DATA_ROOT": str(volume / "datasets" / self.selected["dataset_request_sha256"]),
                    "RUNPOD_REMOTE_SELECTION_PATH": str(remote),
                    "RUNPOD_PYTHON_BIN": str(binary / "python3"),
                    "RUNPOD_POD_ID": "fixture-pod",
                    "RUNPOD_EXPECTED_VOLUME_ID": "fixture-volume",
                    "RUNPOD_VOLUME_ID": "fixture-volume",
                }
            )
            result = command("verify_runpod_mounted_readiness.sh", "--baseline")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"verification": "checksums"', result.stdout)


if __name__ == "__main__":
    unittest.main()
