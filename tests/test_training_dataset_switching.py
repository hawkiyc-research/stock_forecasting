"""Dependency-free A/B admission, provenance, and checkpoint regression tests."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "tests/test_baseline_independence.py"))
GATE = runpy.run_path(str(ROOT / "scripts/runpod_training_readiness.py"))
DATA, SELECTION = GATE["DATA_GATE"], GATE["SELECTION"]
encode, metadata = BASE["encode"], BASE["metadata"]
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    CHECKPOINT = runpy.run_path(str(ROOT / "scripts/runpod_remote_checkpoint_preflight.py"))


def training_fixture(selected):
    objects = BASE["fixture"](selected)
    prefix = "datasets/" + selected["dataset_request_sha256"] + "/"
    dataset = json.loads(objects[prefix + "dataset-manifest.json"])
    raw = b"raw-ohlcv"
    log = b'{"status":"complete"}\n'
    dataset["request_log"] = metadata("manifests/api-request-log.jsonl", log, row_count=1)
    dataset["universe_sha256"] = "7" * 64
    # Real prepared manifests include row counts for the bar-store manifest artifact.
    dataset["artifacts"]["bar_store_manifest"]["row_count"] = 1
    objects.update({
        prefix + "raw/market.parquet": raw,
        prefix + "manifests/api-request-log.jsonl": log,
        prefix + "dataset-manifest.json": encode(dataset),
    })
    return objects


def model_fixture():
    values = GATE["model_contract"](ROOT / "configs/stage2_kronos_base_lora.yaml")
    revisions = {values[name + "_id"]: values[name + "_revision"] for name in (
        "time_series_model", "time_series_tokenizer"
    )}
    repositories, objects = {}, {}
    for repo, revision in revisions.items():
        key = "cache/huggingface/hub/models--" + repo.replace("/", "--")
        key += "/snapshots/" + revision
        repositories[repo] = "/runpod-volume/" + key
        objects[key + "/config.json"] = b'{"fixture":true}'
        objects[key + "/model.safetensors"] = b"fixture-weights"
    objects["cache/hf-models.json"] = encode({
        "repositories": repositories, "repository_revisions": revisions,
        "local_files_only_verified": True,
        "kronos_source_revision": values["kronos_source_revision"],
        "time_series_smoke_test": {
            "passed": True, "local_files_only": True, "backend": "kronos", "input_bars": 128,
            **{name: values[name] for name in (
                "kronos_source_revision", "time_series_model_revision",
                "time_series_tokenizer_revision",
            )},
        },
    })
    return objects


def write_objects(volume, objects):
    for key, data in objects.items():
        path = volume / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def load_tracking_functions():
    # Execute the actual pure provenance functions without importing Torch or an ML environment.
    path = ROOT / "src/stock_forecasting/tracking.py"
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {
        "collect_selection_provenance", "collect_dataset_provenance", "_sha256",
    }]
    scope = {
        "__file__": str(path), "os": os, "Path": Path, "json": json,
        "hashlib": hashlib, "runpy": runpy,
        "_SELECTION_ID_PATTERN": re.compile(r"selection-[0-9a-f]{16}"),
        "_SHA256_PATTERN": re.compile(r"[0-9a-f]{64}"),
        "canonical_network_volume_root": lambda: Path(os.environ["NETWORK_VOLUME_ROOT"]),
        "provenance_summary": lambda path: json.loads(path.read_text()),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), scope)
    return scope


class TrainingDatasetSwitchingTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = BASE["selection"](), BASE["selection"]("2016-01-01")
        self.objects = {**training_fixture(self.a), **training_fixture(self.b), **model_fixture()}

    def test_ab_admission_ignores_stale_or_absent_shared_marker(self):
        for stale in (None, b"not-json", encode({"state": "ready", "requested_dataset": self.b})):
            objects = dict(self.objects)
            if stale is not None:
                objects["lifecycle/stage1/dataset.json"] = stale
            reader = BASE["MemoryReader"](objects)
            for selected in (self.a, self.b, self.a):
                result = DATA["verify_data"](
                    ROOT, selected, reader, workers=2, training_artifacts=True
                )
                self.assertEqual(
                    result["dataset_request_sha256"], selected["dataset_request_sha256"]
                )
                GATE["verify_models"](
                    ROOT / selected["stage"]["config_path"], reader, workers=2
                )
            self.assertNotIn("lifecycle/stage1/dataset.json", repr(reader.calls))

    def test_model_manifest_metadata_can_change_but_revisions_cannot(self):
        key = "cache/hf-models.json"
        payload = json.loads(self.objects[key])
        payload.update({"created_at": "new-time", "config": "another-config", "verify_only": True})
        self.objects[key] = encode(payload)
        reader = BASE["MemoryReader"](self.objects)
        config = ROOT / self.a["stage"]["config_path"]
        GATE["verify_models"](config, reader, workers=2)
        payload["repository_revisions"]["NeoQuasar/Kronos-base"] = "0" * 40
        self.objects[key] = encode(payload)
        with self.assertRaisesRegex(ValueError, "repositories/revisions"):
            GATE["verify_models"](config, reader, workers=2)
        self.objects[key] = model_fixture()[key]
        weights = next(name for name in self.objects if name.endswith("model.safetensors"))
        self.objects[weights] = b""
        with self.assertRaisesRegex(ValueError, "empty or truncated"):
            GATE["verify_models"](config, reader, workers=2)

    def test_training_acquisition_artifacts_are_not_silently_skipped(self):
        for suffix in ("raw/market.parquet", "manifests/api-request-log.jsonl"):
            objects = dict(self.objects)
            key = "datasets/" + self.a["dataset_request_sha256"] + "/" + suffix
            objects[key] = b""
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "truncated"):
                DATA["verify_data"](
                    ROOT, self.a, BASE["MemoryReader"](objects), workers=2, training_artifacts=True
                )

    def test_mounted_hf_links_are_validated_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary)
            write_objects(volume, model_fixture())
            for snapshot in volume.rglob("model.safetensors"):
                content = snapshot.read_bytes()
                blob = snapshot.parents[2] / "blobs" / hashlib.sha256(content).hexdigest()
                blob.parent.mkdir(parents=True, exist_ok=True)
                snapshot.rename(blob)
                snapshot.symlink_to("../../blobs/" + blob.name)
            reader = DATA["ArtifactReader"](ROOT, volume=volume)
            config = ROOT / self.a["stage"]["config_path"]
            GATE["verify_models"](config, reader, workers=2)
            blob.write_bytes(b"broken-weights!")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                GATE["verify_models"](config, reader, workers=2)

    def test_tracking_binds_own_selection_and_dataset_not_latest_preparation(self):
        scope = load_tracking_functions()
        with tempfile.TemporaryDirectory() as temporary:
            volume = Path(temporary)
            write_objects(volume, self.objects)
            write_objects(volume, {"lifecycle/stage1/dataset.json": b"invalid-global-marker"})
            for selected in (self.a, self.b, self.a):
                path = volume / (selected["selection_id"] + ".json")
                path.write_bytes(encode(selected))
                env = SELECTION["_selection_exports"](path, selected)
                env.update({"RUNPOD_REMOTE_SELECTION_PATH": str(path),
                            "NETWORK_VOLUME_ROOT": str(volume), "RUNPOD_POD_ID": "fixture-pod"})
                with patch.dict(os.environ, env):
                    result = scope["collect_selection_provenance"]()
                    self.assertEqual(result["selection_id"], selected["selection_id"])
                    self.assertIn(selected["dataset_request_sha256"], result["path"])
                    self.assertEqual(scope["collect_dataset_provenance"]()["status"], "ready")
                    with (
                        patch.dict(os.environ, {"RUNPOD_SELECTION_SHA256": "f" * 64}),
                        self.assertRaisesRegex(ValueError, "immutable selection"),
                    ):
                        scope["collect_selection_provenance"]()
            key = "datasets/" + self.a["dataset_request_sha256"] + "/dataset-manifest.json"
            altered = json.loads(self.objects[key])
            altered["date_range"] = self.b["dataset_request"]["date_range"]
            write_objects(volume, {key: encode(altered)})
            with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "date_range"):
                scope["collect_selection_provenance"]()

    def test_model_repository_cannot_link_outside_the_volume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = root / "volume"
            write_objects(volume, model_fixture())
            repository = volume / "cache/huggingface/hub/models--NeoQuasar--Kronos-base"
            external = root / "outside-repository"
            repository.rename(external)
            repository.symlink_to(external, target_is_directory=True)
            reader = DATA["ArtifactReader"](ROOT, volume=volume)
            with self.assertRaisesRegex(ValueError, "escapes the mounted volume"):
                GATE["verify_models"](ROOT / self.a["stage"]["config_path"], reader, workers=2)

    def test_real_shell_gates_switch_a_b_a_without_preparation(self):
        with tempfile.TemporaryDirectory(prefix="training-switch-") as temporary:
            volume = Path(temporary).resolve() / "volume"
            project = volume / "stock_forecasting"
            for name in ("src", "scripts", "configs"):
                shutil.copytree(ROOT / name, project / name,
                                ignore=shutil.ignore_patterns("__pycache__"))
            write_objects(volume, self.objects)
            write_objects(volume, {"lifecycle/stage1/dataset.json": b"deliberately-unusable"})
            (project / ".env").write_text("RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n")
            (project / ".env").chmod(0o600)
            transport = project / "scripts/fixture_transport.py"
            transport.write_text('''import os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["FAKE_VOLUME"])
if args[:2] == ["s3", "cp"] and args[3] == "-":
    key = args[2].split("/", 3)[3]
    if key == "lifecycle/stage1/dataset.json":
        raise RuntimeError("The shared dataset marker must never be read")
    sys.stdout.buffer.write((root / key).read_bytes())
elif args[:2] == ["s3api", "head-object"]:
    print((root / args[args.index("--key") + 1]).stat().st_size)
else:
    raise RuntimeError("Unexpected mutation or cloud action")
''')
            (project / "scripts/runpod_s3_project.sh").write_text(
                "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " "
                + shlex.quote(str(transport)) + ' "$@"\n'
            )
            output = io.StringIO()
            paths = sorted(str(path.relative_to(project)) for path in project.rglob("*")
                           if path.is_file() and path.suffix in (".py", ".sh", ".yaml", ".json"))
            with contextlib.redirect_stdout(output):
                DATA["READINESS"]["command_code_manifest"](SimpleNamespace(
                    project_root=project, paths=paths, pipeline_path=[], state="ready",
                    remote_project_dir=str(project),
                ))
            write_objects(volume, {"lifecycle/stage1/code.json": output.getvalue().encode()})
            binary = volume / "bin"
            binary.mkdir()
            (binary / "python3").symlink_to(sys.executable)
            (binary / "mountpoint").write_text("#!/usr/bin/env bash\nexit 0\n")
            (binary / "mountpoint").chmod(0o700)
            base_env = {**os.environ, "PATH": f"{binary}:{os.defpath}", "FAKE_VOLUME": str(volume),
                        "RUNPOD_ENV_FILE": str(project / ".env")}
            for selected in (self.a, self.b, self.a):
                SELECTION["_activate_selection"](project, selected)
                remote = volume / "lifecycle/selections" / (selected["selection_id"] + ".json")
                remote.parent.mkdir(parents=True, exist_ok=True)
                remote.write_bytes(encode(selected))
                result = subprocess.run(
                    ["/bin/bash", str(project / "scripts/verify_runpod_stage_readiness.sh"),
                     "--gpu"],
                    capture_output=True, text=True, env=base_env, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                env = {**base_env, **SELECTION["_selection_exports"](remote, selected),
                       "NETWORK_VOLUME_ROOT": str(volume), "PROJECT_ROOT": str(project),
                       "DATA_ROOT": str(volume / "datasets" / selected["dataset_request_sha256"]),
                       "RUNPOD_REMOTE_SELECTION_PATH": str(remote),
                       "RUNPOD_PYTHON_BIN": str(binary / "python3"), "RUNPOD_POD_ID": "fixture-pod",
                       "RUNPOD_EXPECTED_VOLUME_ID": "fixture-volume",
                       "RUNPOD_VOLUME_ID": "fixture-volume"}
                result = subprocess.run(
                    ["/bin/bash", str(project / "scripts/verify_runpod_mounted_readiness.sh")],
                    capture_output=True, text=True, env=env, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('"verification": "checksums"', result.stdout)
            self.assertEqual((volume / "lifecycle/stage1/dataset.json").read_bytes(),
                             b"deliberately-unusable")

    def test_remote_resume_and_validation_use_the_run_bound_dataset(self):
        selected, run_id, checkpoint = self.a, "run-fixture", "checkpoint-000010"
        key = "datasets/" + selected["dataset_request_sha256"] + "/dataset-manifest.json"
        payload = json.loads(self.objects[key])
        contract = {"dataset_artifacts": CHECKPOINT["_dataset_contract"](
            key, payload, hashlib.sha256(self.objects[key]).hexdigest()
        )}
        digest = DATA["digest"](contract)
        config = ROOT / selected["stage"]["config_path"]
        config_bytes = config.read_bytes()
        identity = {"schema_version": "4.0", "run_id": run_id, "run_key": run_id,
                    "training_resume_contract_sha256": digest}
        manifest = {**identity, "training_resume_contract": contract,
                    "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest()}
        monitor = "primary_5d/selection_score"
        policy = {"monitor": monitor, "mode": "min", "selection_source": "validation"}
        row = {"path": checkpoint, "global_step": 10, "value": 1.0, "rank": 1}
        files = {name: b"checkpoint-artifact" for name in (
            "adapter.safetensors", "optimizer.pt", "scheduler.pt"
        )}
        files["resolved-config.yaml"] = config_bytes
        state = {
            **identity, "global_step": 10, "epoch": 0, "batch_index": 10,
            "training_stage": "stage2", "runtime_robust_scales": [1.0] * 14, "rng_state": {},
            "selection": {"source": "validation", "metric": monitor, "mode": "min", "value": 1.0},
            "artifact_files": {name: metadata(name, value) for name, value in files.items()},
        }
        files["trainer-state.json"] = encode(state)
        prefix = "savedModel/" + run_id + "/"
        objects = {**self.objects, **{prefix + checkpoint + "/" + k: v for k, v in files.items()},
                   prefix + "run-manifest.json": encode(manifest),
                   prefix + "resolved-config.yaml": config_bytes,
                   prefix + "checkpoint-leaderboard.json": encode({
                       **identity, **policy, "checkpoints": [row], "best_checkpoint": checkpoint,
                       "save_top_k": 5,
                   }),
                   prefix + "best-checkpoint.json": encode({**identity, **policy, **row})}

        class Reader:
            def __init__(self):
                self.calls = []

            def bytes_object(self, key, label):
                self.calls.append(key)
                return objects[key]

            def json_object(self, key, label):
                return json.loads(self.bytes_object(key, label))

            def optional_json_object(self, key, label):
                return self.json_object(key, label) if key in objects else None

            def content_length(self, key, label):
                return len(self.bytes_object(key, label))

        reader = Reader()
        for policy in ("latest", "best", "retained"):
            result = CHECKPOINT["validate_remote_checkpoint_run"](
                reader, run_id, checkpoint if policy == "retained" else None, policy, config,
                selected["dataset_request_sha256"],
            )
            self.assertEqual(result, checkpoint)
        self.assertNotIn("lifecycle/stage1/dataset.json", reader.calls)
        with self.assertRaisesRegex(ValueError, "active selection"):
            CHECKPOINT["validate_remote_checkpoint_run"](
                reader, run_id, None, "latest", config, self.b["dataset_request_sha256"]
            )
        objects[key] += b" "
        with self.assertRaisesRegex(ValueError, "different numerical dataset artifacts"):
            CHECKPOINT["validate_remote_checkpoint_run"](
                reader, run_id, None, "latest", config, selected["dataset_request_sha256"]
            )

    def test_provenance_only_resume_migration_preserves_numerical_guards(self):
        path = ROOT / "src/stock_forecasting/run_contract.py"
        tree = ast.parse(path.read_text())
        names = {
            "_canonical_payload_digest", "_validated_implementation_files",
            "_checkpoint_retention_migration_matches", "compatible_training_resume_contract_digest",
        }
        nodes = [
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        ]
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef) and node.name in names) or (
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "TRAINING_IMPLEMENTATION_PATHS"
                    for target in node.targets
                )
            ):
                nodes.append(node)
        migrations = runpy.run_path(str(
            ROOT / "src/stock_forecasting/checkpoint_resume_migrations.py"
        ))["CHECKPOINT_RETENTION_MIGRATIONS"]
        scope = {"json": json, "hashlib": hashlib, "CHECKPOINT_RETENTION_MIGRATIONS": migrations}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(path), "exec"), scope)
        migration = next(row for row in migrations if row["id"] == "dataset-scoped-provenance-v1")
        tracking_path = ROOT / "src/stock_forecasting/tracking.py"
        actual = hashlib.sha256(tracking_path.read_bytes()).hexdigest()
        self.assertEqual(migration["to_files"]["tracking.py"], actual)
        old_files = {name: "0" * 64 for name in scope["TRAINING_IMPLEMENTATION_PATHS"]}
        old_files.update(migration["from_files"])
        new_files = {**old_files, **migration["to_files"]}

        def implementation(files):
            return {"files": files, "sha256": DATA["digest"](files)}

        stored = {"data": {"dataset": "a"}, "model": {"rank": 32},
                  "training_implementation": implementation(old_files)}
        current = {**stored, "training_implementation": implementation(new_files)}
        matcher = scope["_checkpoint_retention_migration_matches"]
        self.assertTrue(matcher(stored, current))
        for field in ("data", "model"):
            changed = copy.deepcopy(current)
            changed[field]["different"] = True
            self.assertFalse(matcher(stored, changed))
        changed = {**current, "training_implementation": implementation({
            **new_files, "training.py": "f" * 64
        })}
        self.assertFalse(matcher(stored, changed))
        digest = DATA["digest"](stored)
        scope["training_resume_contract"] = lambda config: current
        self.assertEqual(scope["compatible_training_resume_contract_digest"](None, {
            "training_resume_contract": stored, "training_resume_contract_sha256": digest,
        }), digest)


if __name__ == "__main__":
    unittest.main()
