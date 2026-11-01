"""The end-to-end resume-rule verification (scripts/verify_resume_rules.py): it runs the real
trainers, the real job-script candidate loops and the real volume-reset plan on a volume-like
scratch tree. Opt-in (``FIN_TS_VERIFY_RESUME=1``): it takes about half a minute and is a local
check of the resume rules, not part of any Pod gate (Pod time is rent)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("FIN_TS_VERIFY_RESUME", "") != "1",
    reason="opt-in: FIN_TS_VERIFY_RESUME=1 runs the end-to-end resume verification",
)
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
