"""Keep historical direction versions distinct from independent signal tasks."""

from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_VARIANTS = ("signal64",)
INDIVIDUAL_VARIANTS = ("alpha-point64", "absolute-direction64")
HISTORICAL_VARIANTS = (
    "lora32",
    "lora64",
    "partial",
    "adaptive64",
    "adaptive128",
    "direction64",
    "direction64-v2",
    "direction64-v3",
)


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
        for name in ("model-versions", "signal-tasks", "historical-experiments"):
            for language, text in self.languages.items():
                with self.subTest(section=name, language=language):
                    section = anchored_section(text, f"{name}-{language}")
                    self.assertTrue(section.strip())
                    other = "en" if language == "zh" else "zh"
                    self.assertNotIn(f'<a id="{name}-{other}"></a>', text)

    def test_historical_versions_are_in_each_languages_final_appendix(self):
        for language, text in self.languages.items():
            appendix_marker = f'<a id="historical-experiments-{language}"></a>'
            self.assertEqual(text.count(appendix_marker), 1)
            current, historical = text.split(appendix_marker, 1)
            with self.subTest(language=language):
                for section in (
                    "signal-tasks", "data", "training", "operations", "baseline",
                    "train", "run-status", "download", "artifacts", "cli-reference",
                    "acceptance", "references",
                ):
                    self.assertIn(f'<a id="{section}-{language}"></a>', current)
                    self.assertNotIn(f'<a id="{section}-{language}"></a>', historical)
                self.assertIn(f'<a id="model-versions-{language}"></a>', historical)
                self.assertNotIn(f'<a id="model-versions-{language}"></a>', current)
                self.assertIn(f"](#historical-experiments-{language})", current)
                for variant in HISTORICAL_VARIANTS:
                    for group in ("a", "b"):
                        self.assertIn(f"`{group}-{variant}`", historical)

    def test_current_cli_examples_do_not_launch_or_query_historical_presets(self):
        allowed = {
            f"{group}-{variant}"
            for group in ("a", "b")
            for variant in CURRENT_VARIANTS
        }
        for language, text in self.languages.items():
            current = text.split(f'<a id="historical-experiments-{language}"></a>', 1)[0]
            blocks = re.findall(r"(?ms)^```bash\n(.*?)^```", current)
            found = set()
            for block in blocks:
                commands = block.replace("\\\n", " ")
                for command in commands.splitlines():
                    workflow = re.search(r"runpod_workflow\.sh\s+(train|baseline|runs|readiness)\b", command)
                    if not workflow:
                        continue
                    presets = re.findall(r"--experiment\s+([a-z][a-z0-9-]+)", command)
                    if workflow.group(1) in {"train", "baseline"}:
                        self.assertTrue(
                            presets, f"Launch example must select its current task: {command.strip()}"
                        )
                    for preset in presets:
                        with self.subTest(language=language, command=command.strip()):
                            self.assertIn(preset, allowed)
                        found.add(preset)
            with self.subTest(language=language):
                for variant in CURRENT_VARIANTS:
                    self.assertIn(f"a-{variant}", found)

    def test_pair_workflow_and_model_scoped_advanced_operations_are_explicit(self):
        for language, text in self.languages.items():
            current, appendix = text.split(f'<a id="historical-experiments-{language}"></a>', 1)
            advanced = anchored_section(appendix, f"single-model-{language}")
            with self.subTest(language=language):
                for variant in INDIVIDUAL_VARIANTS:
                    for group in ("a", "b"):
                        self.assertIn(f"`{group}-{variant}`", advanced)
                for marker in ("MODEL_RUN_ID", "probe-scales", "W&B", "baseline"):
                    self.assertIn(marker, advanced)
                for marker in ("pair.json", "download-result.json", "RUNPOD_RESULT_DOWNLOAD_WORKERS"):
                    self.assertIn(marker, current)
                self.assertTrue(re.search(r"三台 pod|3 pods|three pods", plain(current)))
                self.assertTrue(re.search(r"六個(?:子)?模型|(?:6|six) (?:independent |child )?models", plain(current)))
                self.assertIn("--seed 42 --seed 43 --seed 44", current)

    def test_protocol_uses_same_public_pair_workflow_in_both_languages(self):
        text = (ROOT / "docs/experiment_protocol.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("\n## English\n"), 1)
        for language in text.split("\n## English\n"):
            for command in (
                "baseline --experiment a-signal64",
                "readiness --baseline --experiment a-signal64",
                "train --experiment a-signal64 --seed 42 --seed 43 --seed 44",
            ):
                self.assertIn(command, language)
            self.assertNotRegex(language, r"(?:train|baseline) --experiment [ab]-(?:alpha-point64|absolute-direction64)")

    def test_manual_pod_deletion_has_safe_local_monitor_cleanup_commands(self):
        for language, text in self.languages.items():
            current = text.split(f'<a id="historical-experiments-{language}"></a>', 1)[0]
            for command in (
                'bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>"',
                'bash scripts/runpod_workflow.sh cleanup-monitors --pod-id "<POD_ID>" --apply',
            ):
                self.assertIn(command, current)
            self.assertIn("--guard-dir PATH", current)
            self.assertRegex(current, r"PID|process identity")

    def test_rejected_creation_is_not_published_as_a_remote_run(self):
        self.assertIn("GPU 建立請求遭拒時，只保存本機建立嘗試紀錄，不發布遠端 run", self.languages["zh"])
        self.assertIn("A rejected GPU creation request leaves only a local attempt receipt, not a remote run", self.languages["en"])
        for language, text in self.languages.items():
            section = text.split(f'<a id="train-{language}"></a>', 1)[1].split(
                f'<a id="run-status-{language}"></a>', 1
            )[0]
            self.assertIn("guard", section)

    def test_primary_preset_tables_only_recommend_current_tasks(self):
        for language, text in self.languages.items():
            current = text.split(f'<a id="historical-experiments-{language}"></a>', 1)[0]
            rows = "\n".join(
                line for line in current.splitlines() if line.lstrip().startswith("|")
            )
            with self.subTest(language=language):
                for variant in CURRENT_VARIANTS:
                    for group in ("a", "b"):
                        self.assertIn(f"`{group}-{variant}`", rows)
                for variant in HISTORICAL_VARIANTS:
                    for group in ("a", "b"):
                        preset = f"`{group}-{variant}`"
                        self.assertFalse(preset in rows, f"Historical preset in main tables: {preset}")

    def test_reordered_readme_keeps_internal_links_and_unique_anchors(self):
        explicit = re.findall(r'<a id="([^"]+)"></a>', self.markdown)
        self.assertEqual(len(explicit), len(set(explicit)), "Duplicate explicit README anchor")
        headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", self.markdown)
        automatic = {
            re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
            for heading in headings
        }
        targets = set(explicit) | automatic
        links = set(re.findall(r"\]\(#([^)]+)\)", self.markdown))
        self.assertFalse(links - targets, f"Broken README links: {sorted(links - targets)}")

    def test_all_bash_examples_have_valid_shell_syntax_without_execution(self):
        blocks = list(re.finditer(r"(?ms)^```bash\n(.*?)^```", self.markdown))
        self.assertGreater(len(blocks), 80)
        for block in blocks:
            line = self.markdown.count("\n", 0, block.start()) + 1
            result = subprocess.run(
                ["bash", "-n"], input=block.group(1), text=True, capture_output=True, timeout=5
            )
            with self.subTest(line=line):
                self.assertEqual(result.returncode, 0, result.stderr)

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
