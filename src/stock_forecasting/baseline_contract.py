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
    "src/stock_forecasting/baseline_direction.py",
    "src/stock_forecasting/baseline_direction_data.py",
    "src/stock_forecasting/baseline_direction_tree.py",
    "src/stock_forecasting/direction_costs.py",
    "src/stock_forecasting/baselines.py",
    "src/stock_forecasting/evaluation_store.py",
    "src/stock_forecasting/metrics.py",
    "src/stock_forecasting/optimization_policy.py",
    "src/stock_forecasting/data/dataset.py",
    "src/stock_forecasting/data/adjustments.py",
    "src/stock_forecasting/data/horizons.py",
    "src/stock_forecasting/data/sample_universe.py",
    "src/stock_forecasting/data_policy.py",
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
    import runpy

    policy_tools = runpy.run_path(str(project / "src/stock_forecasting/data_policy.py"))
    policy = policy_tools["load_data_policy"](project)
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
        "evaluation": "full-clean-market-session-universe-v3",
        "data_cleaning": policy,
        "evaluation_request": policy_tools["evaluation_request"](selection, policy),
    }
    if parameters.get("direction", {}).get("enabled"):
        definitions = {}
        for relative, names in (
            (
                "models/direction.py",
                {"direction_labels", "direction_cost_features", "classification_loss"},
            ),
            ("direction_runtime.py", {"DirectionMetrics"}),
        ):
            tree = ast.parse((project / "src/stock_forecasting" / relative).read_text())
            nodes = [node for node in tree.body if getattr(node, "name", None) in names]
            if len(nodes) != len(names):
                raise ValueError("Missing shared direction definitions")
            # Source segments are Python-version-independent, excluding unrelated main-model code.
            source = (project / "src/stock_forecasting" / relative).read_text()
            definitions[relative] = [ast.get_source_segment(source, node) for node in nodes]
        core["direction_semantics"] = digest(definitions)
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
            if expected["contract"]["parameters"].get("direction", {}).get("enabled"):
                filename = "direction-probabilities.npy" if learned else "direction-classes.npy"
                for split in ("validation", "test"):
                    if f"{directory}/{split}/{filename}" not in payload["artifacts"]:
                        raise ValueError("Baseline completion lacks full direction predictions")
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
            direction = expected["contract"]["parameters"].get("direction", {})
            if direction.get("enabled"):
                start = expected["contract"]["data"].get("preparation", {}).get("h_start", 1)
                validate_direction_metrics(
                    metrics.get("direction"), direction, counts["test"], horizons=range(start, 15)
                )


def validate_direction_metrics(metrics, parameters, samples, *, horizons=None):
    """A quantile-only completion cannot satisfy a dual-direction contract."""
    if (
        not isinstance(metrics, dict)
        or metrics.get("samples") != samples
        or metrics.get("class_order") != ["down", "neutral", "up"]
        or metrics.get("task_order") != ["absolute", "alpha"]
        or metrics.get("cost_scenario") != parameters["costs"]
    ):
        raise ValueError("Baseline direction population or cost scenario is incomplete")
    for task in ("absolute", "alpha"):
        scores = metrics.get("overall", {}).get(task, {})
        actual = scores.get("horizons", {})
        if (
            not actual
            or (horizons is not None and set(actual) != {str(h) for h in horizons})
            or any(row.get("count") != samples for row in scores["horizons"].values())
        ):
            raise ValueError("Baseline direction horizons are incomplete")


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
    direction = parameters.get("direction", {})
    if not isinstance(direction, dict) or type(direction.get("enabled", False)) is not bool:
        raise ValueError("Baseline direction.enabled must be a boolean")
    if config.model.direction.enabled != bool(direction.get("enabled")):
        raise ValueError("Baseline runtime differs from its direction task configuration")
    if direction.get("enabled") and config.model.direction.costs.model_dump() != direction["costs"]:
        raise ValueError("Baseline runtime direction costs differ from its own configuration")


def validate_local_configuration(project: Path, selection: dict, parameters: dict) -> None:
    """Validate baseline-owned inputs only; never read a main-model YAML or cache."""
    for name in (
        "epochs",
        "batch_size",
        "evaluations_per_epoch",
        "early_stopping_patience_evaluations",
        "early_stopping_start_epoch",
        "plateau_patience_evaluations",
        "plateau_min_low_lr_evaluations",
        "label_scale_calibration_samples",
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
    direction = parameters.get("direction", {})
    if not isinstance(direction, dict) or type(direction.get("enabled", False)) is not bool:
        raise ValueError("Baseline direction.enabled must be a boolean")
    if direction.get("enabled"):
        for name in ("absolute_weight", "alpha_weight"):
            value = direction.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10:
                raise ValueError(f"Invalid baseline direction parameter: {name}")
        size = direction.get("loss_reference_samples")
        if (
            type(size) is not int
            or not 32 <= size <= 8192
            or not isinstance(direction.get("costs"), dict)
        ):
            raise ValueError("Invalid baseline direction references or cost scenario")
    preparation = selection["dataset_request"]["preparation"]
    if preparation["h_start"] not in (1, 2, 3) or preparation["window_size"] < 32:
        raise ValueError("Invalid baseline data horizon or context length")


def validate_direction_comparison_config(project: Path, selection: dict, parameters: dict) -> None:
    """Check only the shared task's costs before a paid main-model launch.

    This does not participate in baseline identity or baseline creation. Read the
    repository's explicit, indented scalar YAML contract without importing ML.
    """
    path = project / selection["stage"]["config_path"]
    active = section = None
    enabled = False
    overrides = {}
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            active = stripped == "model:"
            section = None
        elif active and indent == 2:
            section = "direction" if stripped == "direction:" else None
        elif active and section and indent == 4:
            if stripped.startswith("enabled:"):
                enabled = stripped.split(":", 1)[1].strip().lower() == "true"
            section = "costs" if stripped == "costs:" else "direction"
        elif active and section == "costs" and indent == 6:
            key, value = stripped.split(":", 1)
            try:
                overrides[key] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                if key != "scenario":
                    raise ValueError(f"Unsupported direction cost literal: {key}") from None
                overrides[key] = value.strip()
    if not enabled:
        return
    tree = ast.parse((project / "src/stock_forecasting/config.py").read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DirectionCostConfig"
    )
    defaults = {}
    for node in definition.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            keywords = {item.arg: item.value for item in value.keywords}
            value = keywords.get("default") or keywords["default_factory"].body
        defaults[node.target.id] = ast.literal_eval(value)
    direction = parameters.get("direction", {})
    if not direction.get("enabled") or {**defaults, **overrides} != direction.get("costs"):
        raise ValueError(
            "Main model and baseline direction cost scenarios differ; align configs/baseline.json "
            "before building the matching baseline. No Pod was created."
        )


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
            "epochs",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "evaluations_per_epoch",
            "early_stopping_patience_evaluations",
            "early_stopping_min_delta",
            "early_stopping_start_epoch",
            "plateau_patience_evaluations",
            "plateau_factor",
            "plateau_min_ratio",
            "plateau_min_low_lr_evaluations",
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
        "model": {
            "time_series_backend": "mock",
            "lora": {"enabled": False},
            "direction": {
                **parameters.get("direction", {}),
                "technical_channels": False,
                "representation_weight": 0.0,
            },
        },
        "training": {
            **training,
            "stage": "stage2",
            "evaluation_max_samples": None,
            "learning_rate_schedule": "validation_plateau",
            "lora_learning_rate": min(1e-5, parameters["learning_rate"]),
        },
        "validation": {
            "models": parameters["models"],
            "seeds": parameters["seeds"],
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
    if config.model.direction.enabled:
        direction = identity["contract"]["parameters"].get("direction", {})
        if (
            not direction.get("enabled")
            or direction.get("costs") != config.model.direction.costs.model_dump()
        ):
            raise ValueError("Main model and baseline require identical direction cost scenarios")
    from stock_forecasting.data.manifest import sha256_file
    from stock_forecasting.data.sample_universe import open_clean_dataset, validated_split_roots

    expected_sources = payload.get("data_identity", {}).get("split_sources", {})
    roots = validated_split_roots(config)
    for split in ("train", "validation", "test"):
        source = open_clean_dataset(config, split, validated_root=roots[split])
        actual = {
            "manifest_sha256": sha256_file(source.root / "bar-store.json"),
            "sample_universe": source.sample_universe_identity,
            "samples": len(source),
        }
        source.close()
        if (
            expected_sources.get(split) != actual
            or payload["sample_counts"][split] != actual["samples"]
        ):
            raise ValueError(
                f"Baseline {split} population differs from the cleaned sample universe"
            )
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
