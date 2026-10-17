#!/usr/bin/env python3
"""Decompose signal errors into date-level and within-date parts (stdlib only, read-only).

Inputs are the shared evaluation arrays and persisted GRU/GBDT predictions of the
current signal baselines, plus optional main-model prediction exports. Validation is
the decision basis; test is computed identically for the record.

Usage:
  python3 -I scripts/analyze_signal_decomposition.py --baselines DIR --output JSON \
      [--predictions DIR] [--model LABEL=RUN_ID ...] [--ensemble LABEL=A,B,C ...] \
      [--daily-benchmarks JSON]

Workers: --workers N or DECOMPOSITION_WORKERS (default bounded by CPU, memory and jobs).
One worker runs the four jobs in-process; more workers use a spawn process pool.
"""

import argparse
import array
import ast
import json
import math
import mmap
import multiprocessing
import os
import re
import struct
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ALPHA_ID = "baseline-3befc8e902e5c641b24ddf80c4ab7e11769529fa307e391de47e4416f1d885a3"
DIRECTION_ID = "baseline-b21f46ea4963562f03c648e9b0394f8a069767bca2a15a9f82e7c8ed69e04cc8"
BASELINE_MODELS = ("gru-42", "gbdt-42")
HORIZONS = 14
SPLITS = ("validation", "test")
TASKS = {"alpha": "alpha_point", "direction": "absolute_direction"}
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


LOWER = float32(math.log1p(-0.01))
UPPER = float32(math.log1p(0.01))


def open_npy(path):
    handle = Path(path).open("rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    handle.close()
    if mapped[:6] != b"\x93NUMPY":
        raise ValueError(f"Not an npy file: {path}")
    if mapped[6] == 1:
        length, start = struct.unpack("<H", mapped[8:10])[0], 10
    else:
        length, start = struct.unpack("<I", mapped[8:12])[0], 12
    header = ast.literal_eval(mapped[start : start + length].decode("latin1"))
    if header.get("fortran_order"):
        raise ValueError(f"Fortran-ordered arrays are unsupported: {path}")
    return header, start + length, mapped


def float_array(path, shape_tail):
    header, offset, mapped = open_npy(path)
    if header["descr"] != "<f4" or tuple(header["shape"][1:]) != tuple(shape_tail):
        raise ValueError(f"Unexpected array layout in {path}: {header}")
    values = array.array("f")
    values.frombytes(mapped[offset:])
    mapped.close()
    if len(values) != math.prod(header["shape"]):
        raise ValueError(f"Truncated array: {path}")
    return values, header["shape"][0]


def metadata_groups(path):
    """Return per-sample date/market group IDs without decoding every record."""
    header, offset, mapped = open_npy(path)
    layout, position = {}, 0
    for name, descr in header["descr"]:
        if not descr.startswith("<U"):
            raise ValueError(f"Unexpected metadata field {name}: {descr}")
        size = int(descr[2:]) * 4
        layout[name] = (position, size)
        position += size
    count = header["shape"][0]
    date_start, date_size = layout["date"]
    market_start, market_size = layout["market"]
    keys, names, groups = {}, [], array.array("i")
    for index in range(count):
        base = offset + index * position
        raw_date = mapped[base + date_start : base + date_start + date_size]
        raw_market = mapped[base + market_start : base + market_start + market_size]
        group = keys.get((raw_date, raw_market))
        if group is None:
            date = raw_date.decode("utf-32-le").rstrip("\x00")[:10]
            market = raw_market.decode("utf-32-le").rstrip("\x00")
            group = len(names)
            keys[(raw_date, raw_market)] = group
            names.append((date, market))
        groups.append(group)
    mapped.close()
    return groups, names, count


def smooth_l1(value):
    value = abs(value)
    return 0.5 * value * value if value < 1.0 else value - 0.5


def log_softmax(z0, z1, z2):
    top = max(z0, z1, z2)
    total = math.log(math.exp(z0 - top) + math.exp(z1 - top) + math.exp(z2 - top)) + top
    return z0 - total, z1 - total, z2 - total


class Accumulator:
    """Sum per (date/market group, horizon) cell; derive project metric definitions later."""

    def __init__(self, cells):
        self.cells = cells
        self.sums = {}

    def series(self, name):
        values = self.sums.get(name)
        if values is None:
            values = self.sums[name] = [0.0] * self.cells
        return values

    def summarize(self, context):
        names, markets, market_of = context["names"], context["markets"], context["market_of"]
        market_counts, daily_counts = context["market_counts"], context["daily_counts"]
        total = sum(market_counts)
        result = {}
        for name, sums in self.sums.items():
            by_market_h = [[0.0] * HORIZONS for _ in markets]
            daily = {}
            for group, (date, _market) in enumerate(names):
                row = by_market_h[market_of[group]]
                start = group * HORIZONS
                for h in range(HORIZONS):
                    row[h] += sums[start + h]
                daily[date] = daily.get(date, 0.0) + sum(sums[start : start + HORIZONS])
            per_market = {
                market: sum(by_market_h[m][h] / market_counts[m] for h in range(HORIZONS))
                / HORIZONS
                for m, market in enumerate(markets)
                if market_counts[m]
            }
            pooled = sum(
                sum(by_market_h[m][h] for m in range(len(markets))) / total for h in range(HORIZONS)
            ) / HORIZONS
            result[name] = {
                "aggregate": pooled,
                "market_macro": sum(per_market.values()) / len(per_market),
                "by_market": per_market,
                "daily": {
                    date: value / (daily_counts[date] * HORIZONS)
                    for date, value in sorted(daily.items())
                },
            }
        return result


def group_context(baselines, split):
    groups, names, count = metadata_groups(
        Path(baselines) / ALPHA_ID / "inputs" / split / "metadata.npy"
    )
    markets = sorted({market for _date, market in names})
    market_of = [markets.index(market) for _date, market in names]
    sizes = [0] * len(names)
    for group in groups:
        sizes[group] += 1
    market_counts = [0] * len(markets)
    daily_counts = {}
    for group, size in enumerate(sizes):
        market_counts[market_of[group]] += size
        date = names[group][0]
        daily_counts[date] = daily_counts.get(date, 0) + size
    return {
        "groups": groups, "names": names, "count": count, "markets": markets,
        "market_of": market_of, "sizes": sizes, "market_counts": market_counts,
        "daily_counts": daily_counts,
    }


def load_sources(baselines, split, task, models, ensembles):
    """Return ordered (label, values) prediction sources for one task and split."""
    baseline_id = ALPHA_ID if task == "alpha" else DIRECTION_ID
    tail = (HORIZONS,) if task == "alpha" else (HORIZONS, 3)
    sources = {}
    for model in BASELINE_MODELS:
        path = Path(baselines) / baseline_id / "jobs" / model / split / "predictions.npy"
        sources[model] = float_array(path, tail)[0]
    for label, item in models.items():
        if item["task"] == task:
            path = Path(item["directory"]) / split / "predictions.npy"
            sources[label] = float_array(path, tail)[0]
    for label, members in ensembles.items():
        if not all(member in sources for member in members):
            continue
        arrays = [sources[member] for member in members]
        combined = array.array("f", bytes(4 * len(arrays[0])))
        if task == "alpha":
            for index in range(len(combined)):
                combined[index] = sum(values[index] for values in arrays) / len(arrays)
        else:
            # Mixture of member probabilities, stored back as log-probabilities.
            for index in range(0, len(combined), 3):
                mixed = [0.0, 0.0, 0.0]
                for values in arrays:
                    lp = log_softmax(values[index], values[index + 1], values[index + 2])
                    for k in range(3):
                        mixed[k] += math.exp(lp[k]) / len(arrays)
                for k in range(3):
                    combined[index + k] = math.log(max(mixed[k], 1e-12))
        sources[label] = combined
    return sources


def alpha_job(baselines, split, models, ensembles):
    context = group_context(baselines, split)
    groups, names, count, sizes = (
        context["groups"], context["names"], context["count"], context["sizes"]
    )
    scales = json.loads((Path(baselines) / ALPHA_ID / "complete.json").read_text())["robust_scales"]
    targets, n = float_array(
        Path(baselines) / ALPHA_ID / "inputs" / split / "targets.npy", (HORIZONS,)
    )
    if n != count:
        raise ValueError("Alpha targets and metadata differ in length")
    sources = load_sources(baselines, split, "alpha", models, ensembles)
    cells = len(names) * HORIZONS
    target_sum = [0.0] * cells
    prediction_sum = {label: [0.0] * cells for label in sources}
    for index in range(count):
        base, cell = index * HORIZONS, groups[index] * HORIZONS
        for h in range(HORIZONS):
            target_sum[cell + h] += targets[base + h]
            for label, values in sources.items():
                prediction_sum[label][cell + h] += values[base + h]
    target_mean = [value / sizes[c // HORIZONS] for c, value in enumerate(target_sum)]
    prediction_mean = {
        label: [value / sizes[c // HORIZONS] for c, value in enumerate(values)]
        for label, values in prediction_sum.items()
    }
    # Label-free level: each source's own split-wide mean prediction per market and horizon.
    level = {}
    for label, means in prediction_mean.items():
        totals = [[0.0] * HORIZONS for _ in context["markets"]]
        for group, size in enumerate(sizes):
            for h in range(HORIZONS):
                totals[context["market_of"][group]][h] += means[group * HORIZONS + h] * size
        level[label] = [
            [totals[m][h] / context["market_counts"][m] for h in range(HORIZONS)]
            for m in range(len(context["markets"]))
        ]
    acc = Accumulator(cells)
    zero_s, oracle_s = acc.series("zero_return"), acc.series("oracle_date_mean")
    zero_between, zero_within = acc.series("zero_mse_between"), acc.series("zero_mse_within")
    suffixes = (
        "", "_date_only", "_oracle_date_plus_within", "_mse_between", "_mse_within",
        "_within_plus_split_level",
    )
    series = {
        label: tuple(acc.series(f"{label}{suffix}") for suffix in suffixes) for label in sources
    }
    linear_counts, linear_loss, zero_total = [0.0] * HORIZONS, [0.0] * HORIZONS, [0.0] * HORIZONS
    cross = {label: [0.0, 0.0, 0.0] for label in sources}
    for index in range(count):
        group = groups[index]
        market = context["market_of"][group]
        base, cell = index * HORIZONS, group * HORIZONS
        for h in range(HORIZONS):
            scale = scales[h]
            c = cell + h
            y = targets[base + h] / scale
            y_bar = target_mean[c] / scale
            zero = smooth_l1(y)
            zero_s[c] += zero
            oracle_s[c] += smooth_l1(y_bar - y)
            zero_between[c] += y_bar * y_bar
            zero_within[c] += (y - y_bar) * (y - y_bar)
            zero_total[h] += zero
            if abs(y) >= 1.0:
                linear_counts[h] += 1
                linear_loss[h] += zero
            for label, values in sources.items():
                p = values[base + h] / scale
                p_bar = prediction_mean[label][c] / scale
                full, date_only, plus_within, mse_between, mse_within, leveled = series[label]
                full[c] += smooth_l1(p - y)
                date_only[c] += smooth_l1(p_bar - y)
                plus_within[c] += smooth_l1(y_bar + p - p_bar - y)
                leveled[c] += smooth_l1(p - p_bar + level[label][market][h] / scale - y)
                dp, dy = p - p_bar, y - y_bar
                mse_between[c] += (p_bar - y_bar) * (p_bar - y_bar)
                mse_within[c] += (dp - dy) * (dp - dy)
                values_cross = cross[label]
                values_cross[0] += dp * dy
                values_cross[1] += dp * dp
                values_cross[2] += dy * dy
    between_corr = {}
    for label in sources:
        sxy = sxx = syy = sx = sy = weight = 0.0
        for c in range(cells):
            w = sizes[c // HORIZONS]
            x = prediction_mean[label][c] / scales[c % HORIZONS]
            y = target_mean[c] / scales[c % HORIZONS]
            weight += w
            sx += w * x
            sy += w * y
            sxy += w * x * y
            sxx += w * x * x
            syy += w * y * y
        mx, my = sx / weight, sy / weight
        between_corr[label] = (sxy / weight - mx * my) / math.sqrt(
            max(sxx / weight - mx * mx, 1e-18) * max(syy / weight - my * my, 1e-18)
        )
    return {
        "task": "alpha_point",
        "split": split,
        "samples": count,
        "groups": len(names),
        "sources": list(sources),
        "markets": dict(zip(context["markets"], context["market_counts"])),
        "metrics": acc.summarize(context),
        "zero_linear_regime": {
            "sample_share": [linear_counts[h] / count for h in range(HORIZONS)],
            "loss_share": [linear_loss[h] / zero_total[h] for h in range(HORIZONS)],
        },
        "within_date_correlation": {
            label: v[0] / math.sqrt(v[1] * v[2]) for label, v in cross.items()
        },
        "between_date_correlation": between_corr,
    }


def direction_job(baselines, split, models, ensembles):
    context = group_context(baselines, split)
    groups, names, count, sizes = (
        context["groups"], context["names"], context["count"], context["sizes"]
    )
    markets, market_of = context["markets"], context["market_of"]
    audit = json.loads(
        (Path(baselines) / DIRECTION_ID / "inputs" / "train" / "class-label-audit.json").read_text()
    )
    prior = {}
    for market in markets:
        counts = audit["counts"].get(market) or audit["counts"]["overall"]
        prior[market] = [
            [math.log(value / sum(counts[str(h + 1)])) for value in counts[str(h + 1)]]
            for h in range(HORIZONS)
        ]
    absolute, n = float_array(
        Path(baselines) / DIRECTION_ID / "inputs" / split / "absolute_targets.npy", (HORIZONS,)
    )
    if n != count:
        raise ValueError("Absolute targets and metadata differ in length")
    labels = array.array("b", (0 if v < LOWER else 2 if v > UPPER else 1 for v in absolute))
    del absolute
    sources = load_sources(baselines, split, "direction", models, ensembles)
    cells = len(names) * HORIZONS
    label_counts = [0.0] * (cells * 3)
    probability_sum = {label: [0.0] * (cells * 3) for label in sources}
    for index in range(count):
        base, cell = index * HORIZONS, groups[index] * HORIZONS
        for h in range(HORIZONS):
            label_counts[(cell + h) * 3 + labels[base + h]] += 1
            k = (base + h) * 3
            slot = (cell + h) * 3
            for label, z in sources.items():
                lp = log_softmax(z[k], z[k + 1], z[k + 2])
                target = probability_sum[label]
                target[slot] += math.exp(lp[0])
                target[slot + 1] += math.exp(lp[1])
                target[slot + 2] += math.exp(lp[2])
    # Jeffreys-smoothed in-sample date shares are an optimistic date-level upper bound.
    oracle_log = [0.0] * (cells * 3)
    mean_log = {label: [0.0] * (cells * 3) for label in sources}
    for c in range(cells):
        size = sizes[c // HORIZONS]
        for k in range(3):
            oracle_log[c * 3 + k] = math.log((label_counts[c * 3 + k] + 0.5) / (size + 1.5))
            for label in sources:
                mean_log[label][c * 3 + k] = math.log(
                    max(probability_sum[label][c * 3 + k] / size, 1e-12)
                )
    acc = Accumulator(cells)
    prior_s, oracle_s = acc.series("train_prior"), acc.series("oracle_date_prior")
    suffixes = ("", "_date_only", "_oracle_date_adjusted", "_train_prior_adjusted")
    series = {
        label: tuple(acc.series(f"{label}{suffix}") for suffix in suffixes) for label in sources
    }
    for index in range(count):
        group = groups[index]
        market_prior = prior[markets[market_of[group]]]
        base, cell = index * HORIZONS, group * HORIZONS
        for h in range(HORIZONS):
            c = cell + h
            y = labels[base + h]
            slot = c * 3
            pi = market_prior[h]
            prior_s[c] -= pi[y]
            oracle_s[c] -= oracle_log[slot + y]
            o0, o1, o2 = oracle_log[slot], oracle_log[slot + 1], oracle_log[slot + 2]
            k = (base + h) * 3
            for label, z in sources.items():
                lp = log_softmax(z[k], z[k + 1], z[k + 2])
                means = mean_log[label]
                m0, m1, m2 = means[slot], means[slot + 1], means[slot + 2]
                full, date_only, oracle_adjusted, prior_adjusted = series[label]
                full[c] -= lp[y]
                date_only[c] -= means[slot + y]
                # Bayes prior shift: replace the source's own date-average distribution.
                oracle_adjusted[c] -= log_softmax(
                    lp[0] - m0 + o0, lp[1] - m1 + o1, lp[2] - m2 + o2
                )[y]
                prior_adjusted[c] -= log_softmax(
                    lp[0] - m0 + pi[0], lp[1] - m1 + pi[1], lp[2] - m2 + pi[2]
                )[y]
    daily_up, daily_n = {}, {}
    for group in range(len(names)):
        date = names[group][0]
        slot = (group * HORIZONS + HORIZONS - 1) * 3
        daily_up[date] = daily_up.get(date, 0.0) + label_counts[slot + 2]
        daily_n[date] = daily_n.get(date, 0.0) + sizes[group]
    return {
        "task": "absolute_direction",
        "split": split,
        "samples": count,
        "groups": len(names),
        "sources": list(sources),
        "markets": dict(zip(markets, context["market_counts"])),
        "metrics": acc.summarize(context),
        "daily_up_share_14d": {date: daily_up[date] / daily_n[date] for date in sorted(daily_up)},
    }


def run_job(job):
    task, baselines, split, models, ensembles = job
    function = alpha_job if task == "alpha" else direction_job
    return function(baselines, split, models, ensembles)


def latest_export(predictions, run_id):
    """Pick the newest locally downloaded complete export of one child run."""
    newest = None
    for report in Path(predictions).glob(f"{run_id}/checkpoint-*/export-*/export.json"):
        status = json.loads((report.parent / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("run_id") != run_id or payload.get("kind") != "signal-prediction-export":
            raise ValueError(f"Export identity mismatch: {report}")
        key = (payload["created_at"], report.parent.name)
        if newest is None or key > newest[0]:
            newest = (key, report.parent, payload)
    if newest is None:
        raise FileNotFoundError(f"No complete local export for {run_id}")
    return newest[1], newest[2]


def resolve_models(baselines, predictions, pairs):
    """Map LABEL=RUN_ID pairs to verified export directories."""
    memberships = {
        task: {
            "validation": json.loads(
                (Path(baselines) / baseline / "complete.json").read_text()
            )["validation_membership"]["ordered_symbol_dates_sha256"],
            "test": json.loads((Path(baselines) / baseline / "complete.json").read_text())[
                "evaluation_membership"
            ]["ordered_symbol_dates_sha256"],
        }
        for task, baseline in (("alpha", ALPHA_ID), ("direction", DIRECTION_ID))
    }
    models = {}
    for pair in pairs:
        label, separator, run_id = pair.partition("=")
        if not separator or LABEL.fullmatch(label) is None or label in models:
            raise ValueError(f"Invalid or duplicate --model {pair!r}")
        if label in BASELINE_MODELS:
            raise ValueError(f"--model label collides with a baseline: {label}")
        directory, payload = latest_export(predictions, run_id)
        task = {value: key for key, value in TASKS.items()}.get(payload.get("prediction_task"))
        if task is None:
            raise ValueError(f"Unsupported exported task for {run_id}")
        for split in SPLITS:
            summary = payload["splits"].get(split)
            if summary is None:
                raise ValueError(f"{run_id} export lacks the {split} split")
            digest = (summary.get("sample_membership") or {}).get("ordered_symbol_dates_sha256")
            if digest != memberships[task][split]:
                raise ValueError(f"{run_id} {split} order differs from the baseline arrays")
        models[label] = {"run_id": run_id, "task": task, "directory": str(directory)}
    return models


def resolve_ensembles(models, pairs):
    ensembles = {}
    for pair in pairs:
        label, separator, members = pair.partition("=")
        names = [member for member in members.split(",") if member]
        if (
            not separator
            or LABEL.fullmatch(label) is None
            or label in ensembles
            or label in models
            or len(names) < 2
            or len(set(names)) != len(names)
            or any(name not in models for name in names)
            or len({models[name]["task"] for name in names}) != 1
        ):
            raise ValueError(f"Invalid --ensemble {pair!r}; members must share one task")
        ensembles[label] = names
    return ensembles


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def benchmark_daily(results, runs):
    """Relate recorded main-model test daily losses to date-level outcomes."""
    out = {}
    test = {item["task"]: item for item in results if item["split"] == "test"}
    for task, key in (("alpha_point", "alpha"), ("absolute_direction", "direction")):
        reference = test[task]["metrics"]
        baseline = reference["gbdt-42"]["daily"]
        names = (
            ("zero_return", "oracle_date_mean")
            if task == "alpha_point"
            else ("train_prior", "oracle_date_prior")
        )
        level, oracle = (reference[name]["daily"] for name in names)
        surprise = {d: level[d] - oracle[d] for d in oracle}
        for label, template in runs.items():
            benchmark = json.loads(Path(template.format(task=key)).read_text())
            daily = {
                d[:10]: v
                for d, v in benchmark["models"]["kronos_full"]["metrics"][
                    "daily_signal_loss"
                ].items()
            }
            dates = sorted(set(daily) & set(baseline))
            excess = [daily[d] - baseline[d] for d in dates]
            entry = {
                "dates": len(dates),
                "mean_daily_minus_gbdt": sum(excess) / len(excess),
                "corr_excess_vs_date_surprise": pearson(excess, [surprise[d] for d in dates]),
            }
            if task == "absolute_direction":
                up = test[task]["daily_up_share_14d"]
                entry["corr_excess_vs_up_share_14d"] = pearson(excess, [up[d] for d in dates])
            out.setdefault(task, {})[label] = entry
    return out


def available_memory_bytes():
    try:
        pages = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        size = int(re.search(r"page size of (\d+) bytes", pages).group(1))
        counts = re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)\.", pages)
        return size * sum(map(int, counts))
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return 2 * 1024**3


def job_peak_bytes(baselines, models, ensembles, task, split):
    """Estimate a job's resident arrays: targets, metadata groups and every source."""
    width = 1 if task == "alpha" else 3
    count = metadata_count(Path(baselines) / ALPHA_ID / "inputs" / split / "metadata.npy")
    sources = len(BASELINE_MODELS) + sum(m["task"] == task for m in models.values())
    sources += sum(models[e[0]]["task"] == task for e in ensembles.values())
    per_source = count * HORIZONS * width * 4
    # Python lists of per-sample group IDs and labels add roughly 40 bytes per sample.
    return per_source * (sources + 1) + count * 40 + 256 * 1024**2


def metadata_count(path):
    header, _offset, mapped = open_npy(path)
    mapped.close()
    return header["shape"][0]


def worker_count(requested, jobs, peak):
    if requested is None:
        configured = os.environ.get("DECOMPOSITION_WORKERS")
        requested = int(configured) if configured else None
    if requested is not None:
        if not 1 <= requested <= 16:
            raise ValueError("Worker count must be between 1 and 16")
        return min(requested, jobs)
    by_memory = max(1, int(available_memory_bytes() * 0.5) // max(peak, 1))
    return max(1, min(jobs, max(1, (os.cpu_count() or 2) - 1), by_memory))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=RUN_ID")
    parser.add_argument("--ensemble", action="append", default=[], metavar="LABEL=A,B,C")
    parser.add_argument("--daily-benchmarks", type=Path)
    parser.add_argument("--workers", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.model and args.predictions is None:
        raise SystemExit("--model requires --predictions")
    models = resolve_models(args.baselines, args.predictions, args.model) if args.model else {}
    ensembles = resolve_ensembles(models, args.ensemble)
    jobs = [
        (task, str(args.baselines), split, models, ensembles)
        for split in SPLITS
        for task in ("alpha", "direction")
    ]
    peak = max(job_peak_bytes(args.baselines, models, ensembles, j[0], j[2]) for j in jobs)
    workers = worker_count(args.workers, len(jobs), peak)
    print(f"Signal decomposition: {len(jobs)} jobs, {workers} worker(s), peak~{peak >> 20} MiB/job",
          flush=True)
    if workers == 1:
        results = [run_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(run_job, jobs))
    payload = {"schema_version": "1.0", "models": models, "ensembles": ensembles,
               "results": results}
    if args.daily_benchmarks is not None:
        runs = json.loads(args.daily_benchmarks.read_text())
        payload["main_model_test_daily"] = benchmark_daily(results, runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
