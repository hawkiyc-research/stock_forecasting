"""The end-to-end resume-rule verification (scripts/verify_resume_rules.py) must pass: it runs
the real trainers, the real job-script candidate loops and the real volume-reset plan on a
volume-like scratch tree. Both Pod jobs run this before training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_rules_end_to_end(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "verify_resume_rules", ROOT / "scripts" / "verify_resume_rules.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--volume-root", str(tmp_path / "volume"), "--keep"]) == 0
    report = tmp_path / "volume" / "verify-resume-report.json"
    assert report.is_file()
