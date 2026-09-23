"""Dependency-free baseline identity shared by the local control host and GPU Pod."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path

BASELINE_SOURCES = (
    "src/stock_forecasting/baseline_build.py",
    "src/stock_forecasting/baseline_storage.py",
    "src/stock_forecasting/baseline_runtime.py",
    "src/stock_forecasting/baseline_input.py",
    "src/stock_forecasting/baselines.py",
    "src/stock_forecasting/evaluation_store.py",
    "src/stock_forecasting/metrics.py",
    "src/stock_forecasting/optimization_policy.py",
    "src/stock_forecasting/data/dataset.py",
    "src/stock_forecasting/data/adjustments.py",
    "src/stock_forecasting/data/horizons.py",
)
SHARED_TRAINING_DEFINITIONS = (
    "ResumableFixedSizeBatchSampler",
    "_RuntimeLabelDataset",
    "estimate_runtime_robust_scales",
    "_robust_scale_identity",
    "_load_cached_robust_scales",
    "resolve_runtime_robust_scales",
)


def shared_training_identity(project: Path) -> str:
    tree = ast.parse((project / "src/stock_forecasting/training.py").read_text())
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in SHARED_TRAINING_DEFINITIONS
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("ROBUST_SCALE_")
                for target in node.targets
            )
        )
    ]
    names = {getattr(node, "name", None) for node in selected}
    if not set(SHARED_TRAINING_DEFINITIONS) <= names:
        raise ValueError("Baseline shared calibration/sampling definitions are missing")
    module = ast.Module(body=selected, type_ignores=[])
    # Python 3.13 omits optional empty AST fields by default; retain the 3.12
    # representation used by persisted GPU artifacts on every control host.
    try:
        representation = ast.dump(module, show_empty=True)
    except TypeError:
        representation = ast.dump(module)
    return hashlib.sha256(representation.encode()).hexdigest()


def digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def execution_identity(project: Path) -> dict:
    """Accept only an explicitly reviewed, exact execution-only compatibility bridge."""
    implementation = {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in BASELINE_SOURCES
    }
    shared = shared_training_identity(project)
    result = {
        "implementation": implementation,
        "canonical_implementation": implementation,
        "shared_calibration_sampling": shared,
        "compatibility": None,
    }
    registry = project / "configs/baseline_execution_compatibility.json"
    if not registry.is_file():
        return result
    payload = json.loads(registry.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
        raise ValueError("Invalid baseline execution compatibility registry")
    if len(payload["entries"]) > 32:
        raise ValueError("Baseline execution compatibility registry is too large")
    for entry in payload["entries"]:
        canonical = entry.get("canonical_implementation", {})
        if not canonical or any(
            name not in BASELINE_SOURCES
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for name, value in canonical.items()
        ):
            raise ValueError("Invalid canonical baseline source hashes")
        if (
            entry.get("execution_implementation") == implementation
            and entry.get("shared_calibration_sampling") == shared
        ):
            result["canonical_implementation"] = canonical
            result["compatibility"] = entry["name"]
            break
    return result


def baseline_contract(project: Path, selection: dict) -> dict:
    parameters = json.loads((project / "configs/baseline.json").read_text())
    # Scheduling/resource limits cannot change numerical identity or force a rebuild.
    parameters.pop("resources", None)
    request = selection["dataset_request"]
    execution = execution_identity(project)
    core = {
        "schema_version": 1,
        "data": request,
        "parameters": parameters,
        "implementation": execution["canonical_implementation"],
        "shared_calibration_sampling": execution["shared_calibration_sampling"],
        "evaluation": "full-canonical-stock-date-v2",
    }
    return {"baseline_id": "baseline-" + digest(core), "contract": core}


def runtime_contract() -> tuple[Path, dict]:
    project = Path(__file__).resolve().parents[2]
    selection_path = os.environ.get("RUNPOD_SELECTION_FILE") or os.environ.get(
        "RUNPOD_REMOTE_SELECTION_PATH"
    )
    if not selection_path:
        raise ValueError("Baseline workflow requires the immutable RunPod selection")
    selection = json.loads(Path(selection_path).read_text())
    return project, baseline_contract(project, selection)


def shared_evaluation_paths() -> dict:
    """Reuse the existing full-population input arrays, independent of model identity."""
    return {
        split: {
            "membership": f"inputs/{split}/metadata.npy",
            "targets": f"inputs/{split}/targets.npy",
        }
        for split in ("validation", "test")
    }


def duplicate_evaluation_paths(payload: dict) -> dict[str, str]:
    """Enumerate only redundant job arrays and their existing canonical sources."""
    result = {}
    parameters = payload["identity"]["contract"]["parameters"]
    for model in parameters["models"]:
        learned = model in {"gbdt", "gru", "dlinear", "patchtst"}
        for seed in parameters["seeds"] if learned else [None]:
            job = f"jobs/{model}-{seed}" if learned else f"jobs/rules/{model}"
            for split, paths in shared_evaluation_paths().items():
                for kind, source in paths.items():
                    result[f"{job}/{split}/{kind}.npy"] = source
    return result


def validate_complete(payload: dict, expected: dict, *, require_shared: bool = False) -> None:
    if payload.get("state") != "complete" or payload.get("identity") != expected:
        raise ValueError("No complete baseline matches the active data/training contract")
    models = expected["contract"]["parameters"]["models"]
    if set(payload.get("models", {})) != set(models):
        raise ValueError("Baseline completion does not contain every required model")
    if any(v.get("state") != "complete" for v in payload["models"].values()):
        raise ValueError("A baseline model is incomplete")
    if not payload.get("artifacts") or not payload.get("evaluation_membership"):
        raise ValueError("Baseline completion is missing reusable artifacts or full membership")
    shared = payload.get("evaluation_data")
    if shared is not None:
        if shared != shared_evaluation_paths():
            raise ValueError("Baseline shared evaluation paths are invalid")
        for paths in shared.values():
            for relative in paths.values():
                if relative not in payload["artifacts"]:
                    raise ValueError(f"Missing shared baseline evaluation artifact: {relative}")
        if any(
            path.startswith("jobs/") and Path(path).name in ("membership.npy", "targets.npy")
            for path in payload["artifacts"]
        ):
            raise ValueError("Published baseline artifacts contain duplicate evaluation data")
    elif require_shared:
        raise ValueError(
            "Baseline result storage is not finalized; run the baseline workflow to finalize "
            "the existing results without retraining"
        )
    for name in models:
        learned = name in {"gbdt", "gru", "dlinear", "patchtst"}
        for seed in expected["contract"]["parameters"]["seeds"] if learned else [None]:
            directory = f"jobs/{name}-{seed}" if learned else f"jobs/rules/{name}"
            weight = "model.pkl" if name == "gbdt" else "model.pt" if learned else "model.json"
            required = (
                f"{directory}/{weight}",
                f"{directory}/validation-metrics.json",
                f"{directory}/test/predictions.npy",
            )
            if shared is None:
                # The numerical builder returns staged output before storage publication.
                required += (f"{directory}/test/targets.npy", f"{directory}/test/membership.npy")
            else:
                required += (f"{directory}/validation/predictions.npy",)
            for relative in required:
                if relative not in payload["artifacts"]:
                    raise ValueError(
                        f"Baseline completion lacks required reusable artifact: {relative}"
                    )
    counts = payload.get("sample_counts", {})
    if set(counts) != {"train", "validation", "test"} or any(
        type(v) is not int or v < 1 for v in counts.values()
    ):
        raise ValueError("Baseline completion must record all three full population counts")
    if payload.get("validation_membership", {}).get("samples") != counts["validation"]:
        raise ValueError("Baseline completion is missing the full validation membership")
    for name, result in payload["models"].items():
        scores = result.get("seed_results") or {"single": result.get("metrics", {})}
        if name in {"gbdt", "gru", "dlinear", "patchtst"} and set(scores) != set(
            map(str, expected["contract"]["parameters"]["seeds"])
        ):
            raise ValueError("Baseline completion is missing required random seeds")
        for metrics in scores.values():
            if (
                metrics.get("sample_membership") != payload["evaluation_membership"]
                or metrics.get("samples") != counts["test"]
            ):
                raise ValueError("Baseline metrics do not match the complete test population")
            if metrics.get("evaluation_robust_scales") != payload.get("robust_scales"):
                raise ValueError("Baseline metric calibration differs between experiments")


def validate_optimization_alignment(config, parameters):
    for name in (
        "evaluations_per_epoch",
        "early_stopping_patience_evaluations",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
        "plateau_patience_evaluations",
        "plateau_factor",
        "plateau_min_ratio",
        "plateau_min_low_lr_evaluations",
    ):
        if getattr(config.training, name) != parameters[name]:
            raise ValueError(f"Baseline runtime differs from its own optimization policy: {name}")
    if config.training.learning_rate_schedule != "validation_plateau":
        raise ValueError("The full baseline workflow requires the validation-plateau schedule")
    if (
        config.training.evaluation_max_samples is not None
        or config.validation.baseline_max_samples_per_split is not None
    ):
        raise ValueError("Prebuilt baselines require full validation and full test")
    if (
        config.data.calibration_seed != parameters["calibration_seed"]
        or config.data.label_scale_calibration_samples
        != parameters["label_scale_calibration_samples"]
    ):
        raise ValueError("Baseline runtime differs from its own train-only loss calibration")
    if (
        set(config.validation.models) - {"kronos_full"} != set(parameters["models"])
        or config.validation.seeds != parameters["seeds"]
    ):
        raise ValueError("Baseline runtime differs from its own model/seed manifest")


def validate_local_configuration(project: Path, selection: dict, parameters: dict) -> None:
    """Validate baseline-owned inputs only; never read a main-model YAML or cache."""
    for name in (
        "epochs", "batch_size", "evaluations_per_epoch", "early_stopping_patience_evaluations",
        "early_stopping_start_epoch", "plateau_patience_evaluations",
        "plateau_min_low_lr_evaluations", "label_scale_calibration_samples",
    ):
        if type(parameters.get(name)) is not int or parameters[name] < 1:
            raise ValueError(f"Invalid baseline parameter: {name}")
    for name in ("learning_rate", "plateau_factor", "plateau_min_ratio"):
        value = parameters.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid baseline parameter: {name}")
    if parameters["plateau_factor"] >= 1 or parameters["plateau_min_ratio"] >= 1:
        raise ValueError("Baseline plateau ratios must be below one")
    if parameters["early_stopping_start_epoch"] > parameters["epochs"]:
        raise ValueError("Baseline early stopping starts after its final epoch")
    for name in ("seeds", "models"):
        values = parameters.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid baseline parameter: {name}")
    preparation = selection["dataset_request"]["preparation"]
    if preparation["h_start"] not in (1, 2, 3) or preparation["window_size"] < 32:
        raise ValueError("Invalid baseline data horizon or context length")


def baseline_runtime_payload(selection: dict, parameters: dict, volume: Path) -> dict:
    """Adapt data and baseline parameters to the shared worker serialization format.

    The unused backbone section only satisfies the common configuration envelope;
    no backbone is constructed by baseline_build. Every learned baseline still
    uses its real GRU/DLinear/PatchTST/GBDT implementation and saved weights.
    """
    request = selection["dataset_request"]
    preparation = request["preparation"]
    root = volume / "datasets" / selection["dataset_request_sha256"]
    training = {
        name: parameters[name]
        for name in (
            "epochs", "learning_rate", "weight_decay", "warmup_ratio", "evaluations_per_epoch",
            "early_stopping_patience_evaluations", "early_stopping_min_delta",
            "early_stopping_start_epoch", "plateau_patience_evaluations", "plateau_factor",
            "plateau_min_ratio", "plateau_min_low_lr_evaluations",
        )
    }
    return {
        "experiment_name": "independent-full-data-baselines",
        "data": {
            "raw_path": str(root / "raw/market.parquet"),
            "bar_store_path": str(root / "prepared/bar-store"),
            "manifest_path": str(root / "dataset-manifest.json"),
            "dataset_profile": request["profile"],
            "input_length": preparation["window_size"],
            "h_start": preparation["h_start"],
            "max_abs_log_return": preparation["max_abs_log_return"],
            "effective_embargo_trading_days": preparation["effective_embargo_bars"],
            "train_fraction": 1.0,
            "max_samples": None,
            **preparation["fixed_split"],
            "calibration_seed": parameters["calibration_seed"],
            "label_scale_calibration_samples": parameters["label_scale_calibration_samples"],
        },
        "model": {"time_series_backend": "mock", "lora": {"enabled": False}},
        "training": {
            **training, "stage": "stage2", "evaluation_max_samples": None,
            "learning_rate_schedule": "validation_plateau",
            "lora_learning_rate": min(1e-5, parameters["learning_rate"]),
        },
        "validation": {
            "models": parameters["models"], "seeds": parameters["seeds"],
            "baseline_max_samples_per_split": None,
        },
        "wandb": {"enabled": False},
        "runtime": {"auto_terminate_pod": False},
    }


def require_baselines(config, *, verify_artifacts: bool = True) -> dict:
    _project, identity = runtime_contract()
    # Main-model optimization and architecture do not determine baseline validity.
    # Data identity, complete populations and persisted artifacts are checked below.
    root = (
        Path(
            os.environ.get(
                "NETWORK_VOLUME_ROOT", os.environ.get("RUNPOD_VOLUME_MOUNT_PATH", "/runpod-volume")
            )
        )
        / "baselines"
        / identity["baseline_id"]
    )
    path = root / "complete.json"
    if not path.is_file():
        raise ValueError("Build matching baselines first: bash scripts/runpod_workflow.sh baseline")
    payload = json.loads(path.read_text())
    validate_complete(payload, identity, require_shared=True)
    from stock_forecasting.data.manifest import sha256_file
    from stock_forecasting.training_paths import resolve_bar_store_path

    manifest = resolve_bar_store_path(config.data.bar_store_path) / "bar-store.json"
    if payload.get("data_identity", {}).get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Prepared data differs from the baseline's immutable bar store")
    if payload["sample_counts"] != json.loads(manifest.read_text())["split_counts"]:
        raise ValueError("Baseline population counts differ from the complete prepared splits")
    if verify_artifacts:
        for relative, metadata in payload["artifacts"].items():
            artifact = (root / relative).resolve()
            if (
                not artifact.is_relative_to(root.resolve())
                or not artifact.is_file()
                or artifact.stat().st_size != metadata["bytes"]
            ):
                raise ValueError(f"Missing or truncated baseline artifact: {relative}")
    return payload
