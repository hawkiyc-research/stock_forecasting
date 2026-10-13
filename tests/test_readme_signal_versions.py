"""Keep historical direction versions distinct from independent signal tasks."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def anchored_section(text: str, anchor: str) -> str:
    """Read an explicit documentation scope without depending on its heading text."""
    marker = f'<a id="{anchor}"></a>'
    if text.count(marker) != 1:
        raise AssertionError(f"Expected one documentation anchor: {anchor}")
    tail = text.split(marker, 1)[1]
    return re.split(r'(?m)^<a id="[^"]+"></a>', tail, maxsplit=1)[0]


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).lower()


def class_default(class_name: str, field: str):
    """Inspect scalar defaults with stdlib AST, never import the ML configuration."""
    tree = ast.parse((ROOT / "src/stock_forecasting/config.py").read_text())
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    value = next(
        node.value
        for node in definition.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == field
    )
    return ast.literal_eval(value)


class ReadmeSignalVersionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.markdown = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese, english = cls.markdown.split("\n## English\n", 1)
        cls.languages = {"zh": chinese, "en": english}

    def test_complete_chinese_precedes_complete_english(self):
        self.assertEqual(self.markdown.count("\n## English\n"), 1)
        for name in ("model-versions", "signal-tasks"):
            for language, text in self.languages.items():
                with self.subTest(section=name, language=language):
                    section = anchored_section(text, f"{name}-{language}")
                    self.assertTrue(section.strip())
                    other = "en" if language == "zh" else "zh"
                    self.assertNotIn(f'<a id="{name}-{other}"></a>', text)

    def test_version_map_matches_named_architectures_and_presets(self):
        self.assertEqual(class_default("DirectionConfig", "architecture"), "legacy")
        for version, suffix, architecture in (
            ("V1", "direction64", "legacy"),
            ("V2", "direction64-v2", "market_residual_v2"),
            ("V3", "direction64-v3", "market_ordinal_v3"),
        ):
            for group in ("a", "b"):
                preset = f"{group}-{suffix}"
                config = (ROOT / "configs/experiments" / (preset.replace("-", "_") + ".yaml"))
                source = config.read_text()
                match = re.search(r"(?m)^    architecture: ([a-z0-9_]+)$", source)
                actual = (
                    match.group(1) if match else class_default("DirectionConfig", "architecture")
                )
                self.assertEqual(actual, architecture, preset)
            for language, text in self.languages.items():
                section = anchored_section(text, f"model-versions-{language}")
                rows = [line for line in section.splitlines() if line.lstrip().startswith("|")]
                matching = [
                    row
                    for row in rows
                    if re.search(rf"\b{version}\b", row)
                    and architecture in row
                    and suffix in row
                ]
                with self.subTest(version=version, language=language):
                    self.assertTrue(matching, f"Missing explicit {version} preset/architecture row")

    def test_version_comparison_documents_operational_and_numerical_differences(self):
        dimensions = {
            "labels": r"標籤|labels?",
            "head": r"方向頭|分類頭|heads?",
            "references": r"loss[ _-]references?|參考損失|參考尺度",
            "gradients": r"pcgrad",
            "selection": r"選模|selection|checkpoint monitor",
            "baseline": r"baseline",
            "resume": r"resume|接續",
        }
        for language, text in self.languages.items():
            section = anchored_section(text, f"model-versions-{language}")
            rows = [line for line in section.splitlines() if line.lstrip().startswith("|")]
            for dimension, pattern in dimensions.items():
                with self.subTest(language=language, dimension=dimension):
                    matching = [row for row in rows if re.search(pattern, row, re.IGNORECASE)]
                    self.assertTrue(matching, f"Missing version-comparison dimension: {dimension}")
                    self.assertTrue(
                        any(
                            len([
                                cell for cell in row.strip().strip("|").split("|") if cell.strip()
                            ]) >= 4
                            for row in matching
                        ),
                        f"Version comparison needs entries for all three versions: {dimension}",
                    )

    def test_independent_tasks_are_not_described_as_old_dual_direction_models(self):
        exclusions = {
            "zh": (
                r"(?:不|無|沒有).{0,35}alpha.{0,15}(?:分類|方向頭)",
                r"(?:不|無|沒有).{0,30}(?:交易成本|成本門檻)",
                r"(?:不|無|沒有).{0,35}(?:混合|合併).{0,35}(?:loss|梯度)",
            ),
            "en": (
                r"(?:no|without|not).{0,40}alpha.{0,20}(?:classif|head)",
                r"(?:no|without|not).{0,40}(?:trading costs?|cost-based)",
                r"(?:not|no|without).{0,40}(?:mix|combin)|(?:loss|gradients).{0,30}not combined",
            ),
        }
        for language, text in self.languages.items():
            section = anchored_section(text, f"signal-tasks-{language}")
            normalized = plain(section)
            with self.subTest(language=language):
                for term in ("alpha_point", "absolute_direction", "smooth l1", "cross entropy"):
                    self.assertIn(term, normalized)
                for pattern in exclusions[language]:
                    self.assertRegex(normalized, pattern)
                for contract in (
                    "direction.enabled=false", "signal-1.0", "neutral_return_threshold=0.01"
                ):
                    self.assertIn(contract, section)

    def test_standard_stage_configs_and_omitted_experiment_do_not_switch_tasks(self):
        default = class_default("ModelConfig", "prediction_task")
        self.assertEqual(default, "quantiles")
        for stage in ("stage1", "stage2"):
            source = (ROOT / "configs" / f"{stage}_kronos_base_lora.yaml").read_text()
            match = re.search(r"(?m)^  prediction_task: ([a-z_]+)$", source)
            self.assertEqual(match.group(1) if match else default, "quantiles")
        for language, text in self.languages.items():
            scope = anchored_section(text, f"signal-tasks-{language}")
            normalized = plain(scope)
            with self.subTest(language=language):
                for value in ("stage1", "stage2", "quantiles", "--experiment"):
                    self.assertIn(value, normalized)
                self.assertRegex(
                    normalized,
                    r"(?:省略|未指定|不指定).{0,30}--experiment.{0,70}(?:既有|active selection|configure)"
                    if language == "zh"
                    else (
                        r"omit(?:ted|ting)?[^.]{0,35}--experiment[^.]{0,90}"
                        r"(?:active selection|configured selection)"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
