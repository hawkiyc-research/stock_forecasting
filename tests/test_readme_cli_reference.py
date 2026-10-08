"""Check bilingual CLI documentation without importing the training environment."""

from __future__ import annotations

import ast
import re
import runpy
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTION = re.compile(r"(?<![\w-])--?[A-Za-z][A-Za-z0-9-]*")


def prose(markdown: str) -> str:
    """Do not count an example command as parameter documentation."""
    return re.sub(r"(?ms)^\s*```[^\n]*\n.*?^\s*```\s*$", "", markdown)


def parser_options(relative: str, receiver: str = "parser") -> set[str]:
    """Read argument declarations statically; never import a model CLI."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("-")
    }


def shell_options(relative: str) -> set[str]:
    """Collect literal option case arms, not internal helper invocations."""
    source = (ROOT / relative).read_text(encoding="utf-8")
    arms = re.findall(
        r"(?m)^\s*((?:--?[A-Za-z][A-Za-z0-9-]*)(?:\|--?[A-Za-z][A-Za-z0-9-]*)*)\)"
        r"(?=[ \t]*(?:$|[A-Za-z_]))",
        source,
    )
    comparisons = re.findall(r'"\$1"\s*==\s*(--?[A-Za-z][A-Za-z0-9-]*)', source)
    return {option for arm in arms for option in arm.split("|")} | set(comparisons)


class ReadmeCliReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.markdown = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese, english = cls.markdown.split("\n## English\n", 1)
        cls.languages = {"zh": prose(chinese), "en": prose(english)}

    def assert_documented(self, options: set[str], source: str) -> None:
        self.assertTrue(options, f"No options found in {source}; check the extractor")
        for language, text in self.languages.items():
            with self.subTest(source=source, language=language):
                missing = options - set(OPTION.findall(text))
                self.assertFalse(missing, f"Undocumented options: {sorted(missing)}")

    def test_python_cli_options_have_prose_in_both_languages(self) -> None:
        for relative, receiver in (
            ("src/stock_forecasting/cli/download_market_data.py", "parser"),
            ("src/stock_forecasting/cli/prepare_data.py", "parser"),
            ("src/stock_forecasting/cli/infer.py", "parser"),
            ("src/stock_forecasting/cli/probe_scales.py", "parser"),
            ("scripts/runpod_selection.py", "create"),
            ("scripts/recover_runpod_after_wake.py", "parser"),
            ("scripts/runpod_rest_v2_control.py", "gpu_list"),
            ("scripts/runpod_run_status.py", "parser"),
            ("scripts/runpod_concurrency.py", "train"),
        ):
            self.assert_documented(parser_options(relative, receiver), relative)

    def test_shell_cli_options_have_prose_in_both_languages(self) -> None:
        for relative in (
            "scripts/runpod_workflow.sh",
            "scripts/deploy_runpod_network_volume.sh",
            "scripts/sync_project_to_runpod_volume.sh",
            "scripts/verify_runpod_stage_readiness.sh",
            "scripts/create_runpod_resume_pod.sh",
            "scripts/create_runpod_validation_pod.sh",
            "scripts/download_runpod_results.sh",
            "scripts/download_runpod_probes.sh",
        ):
            self.assert_documented(shell_options(relative), relative)

    def test_gpu_reference_is_complete_in_its_own_section(self) -> None:
        required = parser_options("scripts/runpod_rest_v2_control.py", "gpu_list")
        for language, text in self.languages.items():
            section = text.split(f'<a id="gpu-catalog-{language}"></a>', 1)[1]
            section = section.split("\n##### ", 2)[1]
            with self.subTest(language=language):
                self.assertFalse(required - set(OPTION.findall(section)))
                for value in ("EU-RO-1", "EU-SE-1", "table", "json", "NONE"):
                    self.assertIn(value, section)

    def test_gpu_examples_parse_without_network_or_credentials(self) -> None:
        control = runpy.run_path(str(ROOT / "scripts/runpod_rest_v2_control.py"))
        parser = control["_parser"]()
        examples = re.findall(
            r"(?m)^\s*bash scripts/runpodctl_project\.sh gpu list[^\n]*", self.markdown
        )
        self.assertGreaterEqual(len(examples), 8)
        for example in examples:
            with self.subTest(example=example.strip()):
                parser.parse_args(shlex.split(example)[2:])

    def test_reference_links_resolve_in_the_correct_language(self) -> None:
        for language, text in self.languages.items():
            for name in ("gpu-catalog", "cli-reference", "run-status"):
                anchor = f"{name}-{language}"
                with self.subTest(anchor=anchor):
                    self.assertEqual(text.count(f'<a id="{anchor}"></a>'), 1)
                    self.assertIn(f"](#{anchor})", text)

    def test_examples_alone_do_not_satisfy_documentation(self) -> None:
        example = "```bash\ncommand --undocumented VALUE\n```\nDocument `--documented VALUE`."
        self.assertEqual(set(OPTION.findall(prose(example))), {"--documented"})

    def test_training_seed_launch_and_query_are_documented_separately(self) -> None:
        for language, text in self.languages.items():
            with self.subTest(language=language):
                self.assertRegex(text, r"(?m)^\| `--seed N` \|[^\n]*4294967295")
                query = text.split(f'<a id="run-status-{language}"></a>', 1)[1].split(
                    "\n##### ", 2
                )[1]
                self.assertIn("`--seed N`", query)
                self.assertIn("4294967295", query)
        self.assertEqual(self.markdown.count("--seed 42 --seed 43 --seed 44"), 2)

    def test_baseline_readiness_documents_both_existing_capabilities(self) -> None:
        for text in self.languages.values():
            for expected in ("Baseline: COMPLETE", "Baseline: NOT COMPLETE", "Data rules: PASS",
                             "STORAGE FINALIZATION REQUIRED", "--baseline"):
                self.assertIn(expected, text)
        self.assertNotIn("不是 baseline 訓練完成檢查", self.languages["zh"])
        self.assertNotIn("not completed baseline training", self.languages["en"])

    def test_experiment_protocol_matches_current_early_stop_and_holdout(self) -> None:
        protocol = (ROOT / "docs/experiment_protocol.md").read_text(encoding="utf-8")
        chinese, english = protocol.split("\n## English\n", 1)
        for stage in ("stage1", "stage2"):
            config = (ROOT / f"configs/{stage}_kronos_base_lora.yaml").read_text()
            for key, value in (
                ("early_stopping_start_epoch", 1),
                ("early_stopping_patience_evaluations", 5),
                ("plateau_min_low_lr_evaluations", 2),
            ):
                with self.subTest(stage=stage, key=key):
                    self.assertRegex(config, rf"(?m)^  {key}: {value}$")
        chinese_row = re.search(r"(?m)^\| early stopping \|.*$", chinese).group()
        english_row = re.search(r"(?m)^\| Early stopping \|.*$", english).group()
        for expected in ("連續 5 次未改善", "最低 LR 的兩個訓練間隔", "第 1 epoch 起生效"):
            self.assertIn(expected, chinese_row)
        for expected in ("Five consecutive", "two training intervals", "active from epoch 1"):
            self.assertIn(expected, english_row)
        for language in (chinese, english):
            self.assertIn("`test_unlocked=true`", language)
            self.assertNotIn("`test_unlocked=false`", language)
        self.assertNotIn("第 2 epoch 前生效", chinese)
        self.assertNotIn("activate before epoch 2", english)
        self.assertIn("同一 Pod\n自動執行完整 test benchmark", chinese)
        self.assertIn("the same Pod automatically runs the full test benchmark", english)


if __name__ == "__main__":
    unittest.main()
