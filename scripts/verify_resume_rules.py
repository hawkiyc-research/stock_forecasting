#!/usr/bin/env python3
"""End-to-end check of the resume rules of both walk-forward trainers on a volume-like tree.

The script builds ``<volume root>/diagnostics/{xs,lgbm}/launch-*`` exactly as the Pod jobs do,
runs the REAL trainers (``run_cross_stock``, ``run_walk_forward``) on small synthetic data, kills
runs half-way the way a dead Pod does (a finished ``partial/`` without ``results.json``), runs
the REAL candidate loops of ``scripts/runpod_gpu_xs.sh`` and ``scripts/runpod_cpu_lgbm.sh``
against that tree, resumes, and checks every rule:

1. only a launch with ``partial/``, no ``results.json`` and no ``resumed_into.json`` is offered;
2. a resumed run reuses exactly the finished folds and reproduces the predictions of an
   uninterrupted run;
3. the result records the provenance (``results.json`` → ``resume``) and every source used is
   stamped with ``resumed_into.json``, after which the loops never offer it again and
   ``scripts/runpod_volume_reset.py`` keeps it whole;
4. folds of another experiment, label variant, feature spec or context (schedule, dataset,
   caches, encoder revisions), partials written before the context existed, and partials whose
   prediction rows are not this run's rows are never reused;
5. with several candidates the newest launch name supplies a fold first and each source is
   stamped with the folds it supplied.

Usage::

    python scripts/verify_resume_rules.py [--volume-root DIR] [--keep]

Without ``--volume-root`` a temporary directory is used. On a Pod, point it at a scratch
directory under the network volume (for example ``/runpod-volume/tmp/verify-resume``) so the
checks run on the volume's filesystem; the directory is removed afterwards unless ``--keep``.
Exit status 0 when every check passes, 1 otherwise; the report is also written as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stock_forecasting.cross_stock_training as xs  # noqa: E402
import stock_forecasting.walk_forward_lgbm as lgbm  # noqa: E402
from stock_forecasting.resume_sources import RESUMED_INTO_NAME, resume_candidates  # noqa: E402
from stock_forecasting.walk_forward import quarterly_folds  # noqa: E402

SCHEDULE = {
    "schema": "walk-forward-schedule-v1",
    "name": "verify",
    "calendar": "XTAI",
    "train_start": "2023-01-02",
    "data_end": "2024-07-01",
    "first_quarter": "2024Q1",
    "last_quarter": "2024Q2",
    "embargo_sessions": 15,
    "validation_months": 3,
    "periods": {"P1": ["2024-01-01", "2024-04-01"], "P2": ["2024-04-01", "2024-07-01"]},
}
HORIZONS = [1, 5, 10]
WINDOW, FEATURES = 32, 12
XS_EXPERIMENT, XS_OTHER = "t-xs-a1-no-cross", "t-xs-a4-no-gating"
XS_SEEDS = (1, 2)
XS_PARAMS = {"max_epochs": 2, "patience": 2}
LGBM_SEEDS = (1,)
LGBM_FEATURES = [f"f{i}" for i in range(6)]


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"]]


# --------------------------------------------------------------------------- synthetic data


def _sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", "2024-06-28")


def xs_cache(sessions: pd.DatetimeIndex, seed: int, symbols_per_market: int = 12):
    """A fake XsCache-like object (same layout as tests/test_cross_stock.py)."""
    rng = np.random.default_rng(seed)
    markets, suffixes = ["TPEX", "TWSE"], {"TPEX": "TWO", "TWSE": "TW"}
    count = len(sessions) * len(markets) * symbols_per_market
    windows = np.empty((count, WINDOW, 5), dtype=np.float32)
    features = np.empty((count, FEATURES), dtype=np.float32)
    ranks = np.empty((count, len(HORIZONS)), dtype=np.float32)
    alpha = np.empty((count, len(HORIZONS)), dtype=np.float32)
    cutoff = np.empty(count, dtype=np.int64)
    market_codes = np.empty(count, dtype=np.int8)
    symbol_codes = np.empty(count, dtype=np.int32)
    names = [f"{1000 + i}.{suffixes[m]}" for m in markets for i in range(symbols_per_market)]
    row = 0
    for day in sessions:
        for code, _market in enumerate(markets):
            signal = rng.standard_normal(symbols_per_market)
            noise = rng.standard_normal((symbols_per_market, len(HORIZONS)))
            a = 0.03 * signal[:, None] + 0.01 * noise
            r = pd.DataFrame(a).rank().to_numpy() / symbols_per_market
            for i in range(symbols_per_market):
                drift = 0.002 * signal[i]
                closes = 100.0 * np.exp(
                    np.cumsum(np.full(WINDOW, drift) + 0.01 * rng.standard_normal(WINDOW))
                )
                windows[row, :, 3] = closes
                windows[row, :, 0] = closes * (1 + 0.001 * rng.standard_normal(WINDOW))
                windows[row, :, 1] = closes * 1.01
                windows[row, :, 2] = closes * 0.99
                windows[row, :, 4] = rng.uniform(1e5, 1e6, WINDOW)
                features[row] = rng.standard_normal(FEATURES)
                features[row, 0] = signal[i]
                ranks[row] = r[i]
                alpha[row] = a[i]
                cutoff[row] = day.value
                market_codes[row] = code
                symbol_codes[row] = code * symbols_per_market + i
                row += 1
    return SimpleNamespace(
        split="train",
        windows=windows,
        features=features,
        ranks=ranks,
        alpha=alpha,
        cutoff_ns=cutoff,
        market_codes=market_codes,
        symbol_codes=symbol_codes,
        markets=markets,
        symbols=names,
        horizons=HORIZONS,
        feature_columns=[f"f{i}" for i in range(FEATURES)],
    )


def lgbm_table(seed: int = 0, symbols_per_market: int = 20) -> pd.DataFrame:
    """Synthetic LightGBM samples (same layout as tests/test_walk_forward_lgbm.py)."""
    rng = np.random.default_rng(seed)
    rows = []
    for market, suffix in (("TWSE", "TW"), ("TPEX", "TWO")):
        for day in _sessions():
            values = rng.standard_normal((symbols_per_market, len(LGBM_FEATURES)))
            values = values.astype(np.float32)
            noise = rng.standard_normal((symbols_per_market, len(HORIZONS)))
            alpha = 0.03 * values[:, :1] + 0.01 * noise
            ranks = pd.DataFrame(alpha).rank().to_numpy() / symbols_per_market
            for i in range(symbols_per_market):
                record = {
                    "ordinal": 0,
                    "symbol": f"{1000 + i}.{suffix}",
                    "split": "train",
                    "market": market,
                    "signal_date": day,
                }
                record.update(dict(zip(LGBM_FEATURES, values[i], strict=True)))
                for position, h in enumerate(HORIZONS):
                    record[lgbm.rank_column(h)] = np.float32(ranks[i, position])
                    record[lgbm.alpha_column(h)] = np.float32(alpha[i, position])
                rows.append(record)
    table = pd.DataFrame(rows)
    table["ordinal"] = np.arange(len(table))
    return table


# --------------------------------------------------------------------------- real job pieces


def script_candidates(script: str, volume: Path) -> list[Path]:
    """Run the resume-candidate loop of a Pod job script (verbatim) against ``volume``."""
    job = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    start = job.index("RESUME_ROOTS=()\n")
    end = job.index("done\n", start) + len("done\n")
    loop = job[start:end]
    shell = (
        f"NETWORK_VOLUME_ROOT={shlex.quote(str(volume))}\n{loop}"
        'printf "%s\\n" ${RESUME_ROOTS[@]+"${RESUME_ROOTS[@]}"}\n'
    )
    out = subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True)
    return [Path(line) for line in out.stdout.splitlines() if line.strip()]


def reset_plan_paths(volume: Path) -> set[str]:
    reset = runpy.run_path(str(ROOT / "scripts/runpod_volume_reset.py"))
    return {entry["path"] for entry in reset["build_plan"](volume)}


class Crash(RuntimeError):
    """Stands in for a Pod dying mid-run."""


def _launch(volume: Path, kind: str, stamp: str) -> Path:
    """A launch directory named like the Pod jobs name them (sortable timestamp + nonce)."""
    return volume / "diagnostics" / kind / f"launch-{stamp}"


# --------------------------------------------------------------------------- cross-stock


def xs_run(table, folds, output: Path, *, experiment=XS_EXPERIMENT, crash_after=None, **kw):
    """Run the cross-stock trainer; with ``crash_after`` the run dies after that many
    finished fold-seeds (a dead Pod: partial/ kept, no results.json)."""
    original = xs.train_fold
    calls = {"n": 0}

    def guarded(*args, **kwargs):
        if crash_after is not None and calls["n"] >= crash_after:
            raise Crash("simulated Pod death")
        calls["n"] += 1
        return original(*args, **kwargs)

    xs.train_fold = guarded
    try:
        return xs.run_cross_stock(
            table,
            experiment=experiment,
            folds=folds,
            schedule=SCHEDULE,
            output_root=output,
            seeds=XS_SEEDS,
            min_group_size=5,
            device="cpu",
            params=XS_PARAMS,
            max_groups_per_epoch=20,
            **kw,
        )
    except Crash:
        return None
    finally:
        xs.train_fold = original


def verify_xs(report: Report, volume: Path) -> None:
    sessions = _sessions()
    table = xs.table_from_caches({"train": xs_cache(sessions, 5)})
    folds = quarterly_folds(sessions, SCHEDULE)
    pairs = [[f.index, s] for f in folds for s in XS_SEEDS]
    ctx = "verify|cache-a|cache-b|cache-c"
    script = "runpod_gpu_xs.sh"
    xs_root = volume / "diagnostics" / "xs"

    # R1: a complete run and a dead one.
    complete = _launch(volume, "xs", "20260101T000000Z-1")
    xs_run(table, folds, complete, resume_context=ctx)
    dead = _launch(volume, "xs", "20260101T010000Z-2")
    xs_run(table, folds, dead, resume_context=ctx, crash_after=2)
    (xs_root / "launch-20260101T020000Z-3").mkdir()  # a launch without partial/
    report.check(
        "xs: dead run left exactly its finished partials and no results.json",
        sorted(p.name for p in (dead / "partial").iterdir())
        == ["fold-00-seed-1.npz", "fold-00-seed-2.npz"]
        and not (dead / "results.json").exists(),
    )
    offered = script_candidates(script, volume)
    report.check(
        "xs: job script offers only the unfinished, unstamped launch",
        offered == [dead],
        f"offered {[p.name for p in offered]}",
    )

    # R2 + R3: resume from the offered candidates.
    resumed = _launch(volume, "xs", "20260101T030000Z-4")
    results = xs_run(table, folds, resumed, resume_context=ctx, resume_root=offered)
    report.check(
        "xs: results.json records the source and the [fold, seed] keys it supplied",
        results["resume"] == [{"root": dead.name, "keys": [[0, 1], [0, 2]]}],
        json.dumps(results["resume"]),
    )
    a = pd.read_parquet(complete / "predictions.parquet")
    b = pd.read_parquet(resumed / "predictions.parquet")
    same = a.shape == b.shape and all(
        np.allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-6)
        for c in a.columns
        if a[c].dtype.kind == "f"
    )
    report.check("xs: resumed run reproduces the uninterrupted run's predictions", same)
    stamp_path = dead / RESUMED_INTO_NAME
    stamp = json.loads(stamp_path.read_text()) if stamp_path.is_file() else {}
    report.check(
        "xs: the source is stamped with the absorbing launch and its keys",
        stamp.get("resumed_into") == resumed.name and stamp.get("keys") == [[0, 1], [0, 2]],
        json.dumps(stamp)[:160],
    )
    report.check(
        "xs: after the stamp the job script offers nothing",
        script_candidates(script, volume) == [],
    )
    plan = reset_plan_paths(volume)
    rel = str(dead.relative_to(volume))
    report.check(
        "xs: volume reset keeps the stamped source whole (and still clears finished partial/)",
        not any(p.startswith(rel) for p in plan)
        and f"{complete.relative_to(volume)}/partial" in plan,
    )
    again = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T040000Z-5"),
        resume_context=ctx,
        resume_root=[dead],
    )
    report.check(
        "xs: a stamped source handed over explicitly is still ignored",
        again["resume"] is None and again["fits"]["count"] == len(pairs),
    )

    # R4: wrong experiment / label variant / context / stale partial / wrong rows.
    other_dead = _launch(volume, "xs", "20260101T050000Z-6")
    xs_run(table, folds, other_dead, experiment=XS_OTHER, resume_context=ctx, crash_after=2)
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T060000Z-7"),
        resume_context=ctx,
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "xs: another experiment's partials are never reused and its launch stays unstamped",
        run["resume"] is None and not (other_dead / RESUMED_INTO_NAME).exists(),
    )
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T070000Z-8"),
        experiment=XS_OTHER,
        resume_context="verify|other-dataset",
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "xs: another context (schedule / dataset / caches / encoder) is never reused",
        run["resume"] is None and not (other_dead / RESUMED_INTO_NAME).exists(),
    )
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T080000Z-9"),
        experiment=XS_OTHER,
        resume_context=ctx,
        label_variant=None,
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "xs: the same experiment and context is reused (control for the two checks above)",
        run["resume"] == [{"root": other_dead.name, "keys": [[0, 1], [0, 2]]}],
    )
    stale = _launch(volume, "xs", "20260101T090000Z-10")
    xs_run(table, folds, stale, resume_context=ctx, crash_after=2)
    for path in (stale / "partial").glob("*.npz"):  # a partial written before contexts existed
        with np.load(path, allow_pickle=False) as payload:
            arrays = {k: payload[k] for k in payload.files if k != "context"}
        np.savez(path, **arrays)
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T100000Z-11"),
        resume_context=ctx,
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "xs: partials without a context are never reused when a context is required",
        run["resume"] is None and not (stale / RESUMED_INTO_NAME).exists(),
    )
    shutil.rmtree(stale)  # it was for this check only
    tampered = _launch(volume, "xs", "20260101T110000Z-12")
    xs_run(table, folds, tampered, resume_context=ctx, crash_after=3)
    bad = tampered / "partial" / "fold-00-seed-2.npz"
    with np.load(bad, allow_pickle=False) as payload:
        arrays = {k: payload[k] for k in payload.files}
    arrays["rows"] = arrays["rows"][::-1].copy()
    np.savez(bad, **arrays)
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T120000Z-13"),
        resume_context=ctx,
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "xs: a partial whose rows are not this run's rows of that fold is skipped",
        run["resume"] == [{"root": tampered.name, "keys": [[0, 1], [1, 1]]}],
        json.dumps(run["resume"]),
    )

    # R5: two candidates; the newest launch name supplies a pair first.
    older = _launch(volume, "xs", "20260101T130000Z-14")
    xs_run(table, folds, older, resume_context=ctx, crash_after=3)
    newer = _launch(volume, "xs", "20260101T140000Z-15")
    xs_run(table, folds, newer, resume_context=ctx, crash_after=1)
    offered = script_candidates(script, volume)
    report.check(
        "xs: both unfinished launches are offered",
        sorted(p.name for p in offered) == sorted([older.name, newer.name]),
    )
    run = xs_run(
        table,
        folds,
        _launch(volume, "xs", "20260101T150000Z-16"),
        resume_context=ctx,
        resume_root=offered,
    )
    report.check(
        "xs: the newest candidate supplies a pair first, the older one the rest",
        run["resume"]
        == [
            {"root": older.name, "keys": [[0, 2], [1, 1]]},
            {"root": newer.name, "keys": [[0, 1]]},
        ],
        json.dumps(run["resume"]),
    )
    report.check(
        "xs: each source is stamped with exactly the keys it supplied",
        json.loads((older / RESUMED_INTO_NAME).read_text())["keys"] == [[0, 2], [1, 1]]
        and json.loads((newer / RESUMED_INTO_NAME).read_text())["keys"] == [[0, 1]],
    )
    report.check(
        "xs: nothing is offered once every unfinished launch is stamped",
        script_candidates(script, volume) == [],
    )
    report.check(
        "resume_candidates: missing directories and duplicates are dropped, newest first",
        resume_candidates([older, older, volume / "missing", newer]) == []
        and [p.name for p in resume_candidates([complete, volume / "missing", complete])]
        == [complete.name],
    )


# --------------------------------------------------------------------------- LightGBM


def lgbm_run(table, folds, output: Path, *, crash_after=None, features=None, **kw):
    """Run the LightGBM trainer (single process); with ``crash_after`` the run dies after
    that many folds were saved."""
    original = lgbm._save_partial
    calls = {"n": 0}

    def guarded(*args, **kwargs):
        if crash_after is not None and calls["n"] >= crash_after:
            raise Crash("simulated Pod death")
        calls["n"] += 1
        return original(*args, **kwargs)

    lgbm._save_partial = guarded
    try:
        return lgbm.run_walk_forward(
            table,
            feature_columns=features or LGBM_FEATURES,
            horizons=HORIZONS,
            folds=folds,
            schedule=SCHEDULE,
            output_root=output,
            seeds=LGBM_SEEDS,
            workers=1,
            max_rounds=20,
            early_stopping=3,
            min_group_size=10,
            **kw,
        )
    except Crash:
        return None
    finally:
        lgbm._save_partial = original


def verify_lgbm(report: Report, volume: Path) -> None:
    os.environ[lgbm.WORKERS_ENV] = "1"
    table = lgbm_table()
    folds = quarterly_folds(_sessions(), SCHEDULE)
    ctx = "verify|digest-a|digest-b|digest-c"
    script = "runpod_cpu_lgbm.sh"

    complete = _launch(volume, "lgbm", "20260101T000000Z-1")
    lgbm_run(table, folds, complete, resume_context=ctx)
    dead = _launch(volume, "lgbm", "20260101T010000Z-2")
    lgbm_run(table, folds, dead, resume_context=ctx, crash_after=1)
    report.check(
        "lgbm: dead run left exactly its finished fold and no results.json",
        sorted(p.name for p in (dead / "partial").iterdir()) == ["fold-00.npz"]
        and not (dead / "results.json").exists(),
    )
    offered = script_candidates(script, volume)
    report.check(
        "lgbm: job script offers only the unfinished, unstamped launch",
        offered == [dead],
        f"offered {[p.name for p in offered]}",
    )
    resumed = _launch(volume, "lgbm", "20260101T020000Z-3")
    results = lgbm_run(table, folds, resumed, resume_context=ctx, resume_root=offered)
    report.check(
        "lgbm: results.json records the source and the folds it supplied",
        results["resume"] == [{"root": dead.name, "keys": [0]}],
        json.dumps(results["resume"]),
    )
    a = pd.read_parquet(complete / "predictions.parquet")
    b = pd.read_parquet(resumed / "predictions.parquet")
    same = a.shape == b.shape and all(
        np.allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-6)
        for c in a.columns
        if a[c].dtype.kind == "f"
    )
    report.check("lgbm: resumed run reproduces the uninterrupted run's predictions", same)
    stamp_path = dead / RESUMED_INTO_NAME
    stamp = json.loads(stamp_path.read_text()) if stamp_path.is_file() else {}
    report.check(
        "lgbm: the source is stamped with the absorbing launch and its folds",
        stamp.get("resumed_into") == resumed.name and stamp.get("keys") == [0],
    )
    report.check(
        "lgbm: after the stamp the job script offers nothing",
        script_candidates(script, volume) == [],
    )
    plan = reset_plan_paths(volume)
    rel = str(dead.relative_to(volume))
    report.check(
        "lgbm: volume reset keeps the stamped source whole",
        not any(p.startswith(rel) for p in plan),
    )
    other = _launch(volume, "lgbm", "20260101T030000Z-4")
    lgbm_run(table, folds, other, resume_context=ctx, crash_after=1)
    run = lgbm_run(
        table,
        folds,
        _launch(volume, "lgbm", "20260101T040000Z-5"),
        resume_context="verify|other-dataset",
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "lgbm: another context (schedule / dataset) is never reused",
        run["resume"] is None and not (other / RESUMED_INTO_NAME).exists(),
    )
    run = lgbm_run(
        table,
        folds,
        _launch(volume, "lgbm", "20260101T050000Z-6"),
        resume_context=ctx,
        features=list(reversed(LGBM_FEATURES)),
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "lgbm: another feature spec is never reused",
        run["resume"] is None and not (other / RESUMED_INTO_NAME).exists(),
    )
    run = lgbm_run(
        table,
        folds,
        _launch(volume, "lgbm", "20260101T060000Z-7"),
        resume_context=ctx,
        label_variant="residual_x",
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "lgbm: another label variant is never reused",
        run["resume"] is None and not (other / RESUMED_INTO_NAME).exists(),
    )
    run = lgbm_run(
        table,
        folds,
        _launch(volume, "lgbm", "20260101T070000Z-8"),
        resume_context=ctx,
        resume_root=script_candidates(script, volume),
    )
    report.check(
        "lgbm: the same context, features and labels are reused (control)",
        run["resume"] == [{"root": other.name, "keys": [0]}]
        and (other / RESUMED_INTO_NAME).exists(),
    )
    report.check(
        "lgbm: nothing is offered once every unfinished launch is stamped",
        script_candidates(script, volume) == [],
    )


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--volume-root", type=Path, help="Scratch tree (created; removed).")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch tree.")
    args = parser.parse_args(argv)
    created_temp = args.volume_root is None
    volume = Path(tempfile.mkdtemp(prefix="verify-resume-")) if created_temp else args.volume_root
    if volume.exists() and any(volume.iterdir()):
        raise SystemExit(f"--volume-root must be a new or empty directory: {volume}")
    for kind in ("xs", "lgbm"):
        (volume / "diagnostics" / kind).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = Report()
    print(f"verify_resume_rules: scratch volume {volume}")
    try:
        verify_xs(report, volume)
        verify_lgbm(report, volume)
    finally:
        summary = {
            "volume_root": str(volume),
            "checks": report.checks,
            "passed": len(report.checks) - len(report.failed),
            "failed": len(report.failed),
            "seconds": round(time.monotonic() - started, 1),
        }
        out = (volume if args.keep else Path(tempfile.gettempdir())) / "verify-resume-report.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"verify_resume_rules: {summary['passed']} passed, {summary['failed']} failed "
            f"in {summary['seconds']} s; report {out}"
        )
        if not args.keep:
            shutil.rmtree(volume, ignore_errors=True)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
