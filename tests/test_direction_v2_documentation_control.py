"""Keep direction-v2 documentation aligned without loading the ML environment."""

from __future__ import annotations

import ast
import contextlib
import re
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "architecture",
    "loss_reference_samples",
    "class_balance_power",
    "market_residual_hidden_dim",
    "prior_smoothing",
    "market_weight_max",
    "gradient_management",
    "gradient_diagnostics_interval",
    "probability_bins",
    "max_daily_entries",
    "selection_pinball_guard_tolerance",
    "selection_pinball_guard_weight",
}
SOURCES = (
    "https://research.google/pubs/long-tail-learning-via-logit-adjustment/",
    "https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html",
)


def config_fields() -> dict[str, object]:
    """Extract Pydantic field defaults statically, never importing torch/config."""
    source = (ROOT / "src/stock_forecasting/config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direction = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DirectionConfig"
    )
    fields = {}
    for node in direction.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Field"
        ):
            value = next((key.value for key in value.keywords if key.arg == "default"), None)
        if value is not None:
            with contextlib.suppress(TypeError, ValueError):
                fields[node.target.id] = ast.literal_eval(value)
    return fields


class DirectionV2DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = {}
        for relative in ("README.md", "docs/experiment_protocol.md"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            if source.count("\n## English\n") != 1:
                raise AssertionError(f"{relative} must have one full English section after Chinese")
            chinese, english = source.split("\n## English\n", 1)
            cls.documents[relative] = {"zh": chinese, "en": english}

    def test_new_and_retained_experiments_are_documented_in_both_languages(self) -> None:
        registry = ast.parse((ROOT / "scripts/runpod_selection.py").read_text(encoding="utf-8"))
        literals = {
            node.value
            for node in ast.walk(registry)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("direction64", literals)
        self.assertIn("direction64-v2", literals)
        for relative, languages in self.documents.items():
            for language, text in languages.items():
                with self.subTest(document=relative, language=language):
                    for name in ("direction64", "direction64-v2"):
                        self.assertIn(f"`{name}`", text)
                    for group in ("a", "b"):
                        self.assertIn(f"{group}_direction64_v2.yaml", text)
                        self.assertIn(f"{group}-direction64-v2", text)
        for group in ("a", "b"):
            self.assertTrue((ROOT / f"configs/experiments/{group}_direction64.yaml").is_file())
            self.assertTrue((ROOT / f"configs/experiments/{group}_direction64_v2.yaml").is_file())

    def test_new_config_fields_have_bilingual_parameter_rows(self) -> None:
        fields = config_fields()
        self.assertTrue(fields.keys() >= SETTINGS)
        for language, text in self.documents["README.md"].items():
            rows = {
                match.group(1): match.group(2)
                for match in re.finditer(r"(?m)^\| `([a-z_]+)` \|([^\n]+)$", text)
            }
            for name in SETTINGS:
                with self.subTest(language=language, setting=name):
                    self.assertIn(name, rows)
                    overrides = {"architecture", "gradient_management", "loss_reference_samples"}
                    if name not in overrides:
                        self.assertIn(str(fields[name]), rows[name])
            self.assertIn("market_residual_v2", rows["architecture"])
            self.assertIn("legacy", rows["architecture"])
            for mode in ("none", "monitor", "pcgrad"):
                self.assertIn(mode, rows["gradient_management"])
            self.assertIn("8192", rows["loss_reference_samples"])

    def test_population_probabilities_and_balanced_metrics_cannot_be_conflated(self) -> None:
        for relative, languages in self.documents.items():
            for language, text in languages.items():
                with self.subTest(document=relative, language=language):
                    for marker in (
                        "softmax(z)",
                        "argmax(f)",
                        "balanced_decision",
                        "raw accuracy",
                        "CE(z)",
                        "log(3)",
                        "PCGrad",
                        "512 MiB",
                        "LoRA numel",
                    ):
                        self.assertIn(marker, text)
                    for source in SOURCES:
                        self.assertIn(source, text)

    def test_final_market_weight_cap_and_inference_class_indices_are_explicit(self) -> None:
        source = (ROOT / "src/stock_forecasting/cli/infer.py").read_text(encoding="utf-8")
        literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for marker in (
            "probability_source",
            "population_logits",
            "balanced_decisions",
            "history_residual_logits_without_train_prior",
        ):
            self.assertIn(marker, literals)
        for relative, languages in self.documents.items():
            for language, text in languages.items():
                with self.subTest(document=relative, language=language):
                    for marker in (
                        "w_m=min(k/p_m, market_weight_max)",
                        "sum(p_m*w_m)=1",
                        'probability_source="population_logits"',
                        "balanced_decisions.absolute",
                        "0=down",
                        "1=neutral",
                        "2=up",
                        "history_residual_logits_without_train_prior",
                    ):
                        self.assertIn(marker, text)
                    self.assertIn("capped water filling", text.lower())
                    if language == "zh":
                        self.assertIn("最終有效權重不超過", text)
                        self.assertIn("train 加權平均為 1", text)
                        self.assertIn("不是先截斷再正規化", text)
                    else:
                        self.assertIn("final effective weight", text)
                        self.assertRegex(text, r"train-weighted mean (?:is )?1")
                        self.assertIn("not clipping followed by normalization", text)

    def test_selection_formula_is_identical_and_first_full_validation_anchored(self) -> None:
        formula = (
            "selection_loss = 1 - mean_market(mean_task(mean_horizon(balanced_decision_BA)))\n"
            "                 + max(pinball/first_full_validation_anchor - 1 - 0.01, 0)*1"
        )
        for relative, languages in self.documents.items():
            for language, text in languages.items():
                with self.subTest(document=relative, language=language):
                    self.assertEqual(text.count(formula), 1)
                    for marker in ("resume", "test", "anchor", "direction/selection_loss"):
                        self.assertIn(marker, text)
                    if language == "zh":
                        self.assertRegex(text, r"第一輪\*\*完整\*\* validation")
                    else:
                        self.assertIn("first **complete full** validation", text)

    def test_fixed_calibration_and_bounded_diagnostics_are_explicit(self) -> None:
        for relative, languages in self.documents.items():
            for language, text in languages.items():
                with self.subTest(document=relative, language=language):
                    self.assertIn("8,192", text)
                    self.assertIn("seed 59", text)
                    self.assertRegex(text, r"20 (?:個\s+)?confidence bins")
                    self.assertIn("4,096", text)
                    self.assertIn("28 \u00d7 LoRA numel", text)
                    if language == "zh":
                        self.assertIn("不執行模型 forward", text.replace("未訓練", ""))
                        self.assertTrue(
                            "不需因此重訓" in text or "重用已完成的雙方向 baseline" in text,
                            f"{relative} must preserve completed baseline reuse for v2",
                        )
                    else:
                        self.assertRegex(
                            text, r"(?:no untrained-model forward|without any\s+model forward)"
                        )
                        self.assertRegex(text, r"(?:baseline retraining|baselines remain reusable)")

    def test_training_launch_examples_keep_seed_runtime_gpu_and_original_cli(self) -> None:
        for language, text in self.documents["README.md"].items():
            commands = []
            for block in re.findall(r"(?ms)^```bash\n(.*?)^```", text):
                for line in block.replace("\\\n", " ").splitlines():
                    if line.startswith("bash scripts/runpod_workflow.sh train "):
                        commands.append(shlex.split(line))
            multi_seed = [command for command in commands if "--seed" in command]
            self.assertEqual(len(multi_seed), 1, language)
            command = multi_seed[0]
            self.assertEqual(command[command.index("--experiment") + 1], "a-direction64-v2")
            self.assertEqual(
                [command[i + 1] for i, value in enumerate(command) if value == "--seed"],
                ["42", "43", "44"],
            )
            self.assertEqual(command[command.index("--maxRuntime") + 1], "72h")
            self.assertEqual(command[command.index("--gpuId") + 1], "NVIDIA GeForce RTX 5090")
            for retained in ("--launchWorkers", "--max-runtime", "--gpu-id", "--seed N"):
                self.assertIn(retained, text)
            for feature in ("readiness --baseline", "status", "runs", "resume", "download"):
                self.assertIn(feature, text)


if __name__ == "__main__":
    unittest.main()
