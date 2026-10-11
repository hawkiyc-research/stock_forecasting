"""Main-model-only train priors and bounded dual-direction diagnostics."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecasting.direction_runtime import DirectionMetrics
from stock_forecasting.models.direction import direction_labels
from stock_forecasting.models.ranking import MARKET_IDS

MARKET_ORDER = ("US", "TWSE", "TPEX", "unknown")
INDEX_CHUNK_SIZE = 262144


def _quotas(masses, total, capacities):
    """Round weighted allocations deterministically without exceeding a stratum."""
    masses = np.asarray(masses, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    total = min(int(total), int(capacities.sum()))
    result = np.zeros(len(masses), dtype=np.int64)
    while int(result.sum()) < total:
        remaining = total - int(result.sum())
        active = (result < capacities) & (masses > 0)
        if not active.any():
            raise ValueError("Calibration strata have no remaining positive mass")
        exact = remaining * np.where(active, masses, 0) / masses[active].sum()
        addition = np.minimum(np.floor(exact).astype(np.int64), capacities - result)
        result += addition
        remainder = total - int(result.sum())
        if remainder:
            order = np.argsort(-(exact - addition), kind="stable")
            for position in order:
                if result[position] < capacities[position] and masses[position] > 0:
                    result[position] += 1
                    remainder -= 1
                    if not remainder:
                        break
    return result


def _reservoir_ordinals(index, start, stop, codes, count, rng):
    """Native chunked random-key selection retains only a bounded reservoir."""
    kept_keys = np.empty(0, dtype=np.float64)
    kept_rows = np.empty(0, dtype=np.int64)
    if not count:
        return kept_rows
    for begin in range(start, stop, INDEX_CHUNK_SIZE):
        chunk = index[begin : min(begin + INDEX_CHUNK_SIZE, stop)]
        matches = np.isin(chunk["group"] % 256, codes)
        rows = np.asarray(chunk["ordinal"][matches], dtype=np.int64)
        keys = rng.random(len(rows))
        kept_rows = np.concatenate((kept_rows, rows))
        kept_keys = np.concatenate((kept_keys, keys))
        if len(kept_rows) > count:
            chosen = np.argpartition(kept_keys, count - 1)[:count]
            kept_rows, kept_keys = kept_rows[chosen], kept_keys[chosen]
    if len(kept_rows) != count:
        raise ValueError("Direction calibration did not fill its bounded stratum")
    return kept_rows


def calibration_indices(index, market_names, count, seed, yearly_decay):
    """Use equal market budgets and the production annual-decay distribution."""
    if len(index) < 1 or count < 4:
        raise ValueError("Direction calibration needs at least four training windows")
    if yearly_decay is not None and not 0 < yearly_decay <= 1:
        raise ValueError("Invalid direction calibration annual decay")
    codes = [
        [i for i, name in enumerate(market_names) if MARKET_IDS.get(str(name).upper(), 3) == m]
        for m in range(4)
    ]
    origin = date(1970, 1, 1)
    first = date.fromordinal(origin.toordinal() + int(index["group"][0]) // 256).year
    last = date.fromordinal(origin.toordinal() + int(index["group"][-1]) // 256).year
    strata = []
    for year in range(first, last + 1):
        bounds = [(date(y, 1, 1) - origin).days * 256 for y in (year, year + 1)]
        start, stop = map(int, np.searchsorted(index["group"], bounds))
        counts = np.zeros(4, dtype=np.int64)
        for begin in range(start, stop, INDEX_CHUNK_SIZE):
            raw = index["group"][begin : min(begin + INDEX_CHUNK_SIZE, stop)] % 256
            for market, market_codes in enumerate(codes):
                counts[market] += int(np.isin(raw, market_codes).sum())
        decay = (yearly_decay or 1.0) ** (last - year)
        strata.append((year, start, stop, counts, decay))
    capacities = np.stack([row[3] for row in strata]).sum(0)
    masses = np.stack([row[3] * row[4] for row in strata]).sum(0)
    budget = _quotas((capacities > 0).astype(float), count, capacities)
    rng = np.random.default_rng(seed)
    selected, sample_weights, details = [], [], []
    for market in range(4):
        quotas = _quotas(
            [row[3][market] * row[4] for row in strata],
            budget[market],
            [row[3][market] for row in strata],
        )
        nonempty = np.asarray([row[3][market] > 0 for row in strata], dtype=np.int64)
        if budget[market] >= int(nonempty.sum()):
            # A tiny historical stratum must not disappear from the prior estimate.
            quotas = nonempty + _quotas(
                [row[3][market] * row[4] for row in strata],
                int(budget[market] - nonempty.sum()),
                np.asarray([row[3][market] for row in strata]) - nonempty,
            )
        for row, quota in zip(strata, quotas, strict=True):
            year, start, stop, population, decay = row
            if not quota:
                continue
            ordinals = _reservoir_ordinals(index, start, stop, codes[market], int(quota), rng)
            selected.extend(ordinals.tolist())
            sample_weights.extend([float(population[market] * decay / quota)] * int(quota))
            details.append(
                {
                    "market": MARKET_ORDER[market],
                    "year": year,
                    "population": int(population[market]),
                    "per_window_weight": float(decay),
                    "samples": int(quota),
                }
            )
    order = np.argsort(selected, kind="stable")
    return (
        np.asarray(selected, dtype=np.int64)[order].tolist(),
        np.asarray(sample_weights, dtype=np.float64)[order],
        {"market_population_mass": masses.tolist(), "strata": details},
    )


def references_from_arrays(
    targets,
    labels,
    markets,
    scales,
    sample_weights,
    *,
    smoothing,
    market_weight_max,
    metadata,
):
    """Estimate data-only priors; zero-centered null quantiles are seed invariant."""
    targets = np.asarray(targets, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    markets = np.asarray(markets, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError("Train-only direction targets must have sample and horizon axes")
    size, horizons = targets.shape
    mass = np.asarray(metadata.get("market_population_mass", []), dtype=np.float64)
    if (
        labels.shape != (size, 2, horizons)
        or markets.shape != (size,)
        or scales.shape != (horizons,)
        or sample_weights.shape != (size,)
        or size < 4
        or not np.isfinite(targets).all()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
        or not np.isfinite(sample_weights).all()
        or (sample_weights <= 0).any()
        or not np.isin(labels, [0, 1, 2]).all()
        or not np.isin(markets, [0, 1, 2, 3]).all()
        or not math.isfinite(smoothing)
        or smoothing <= 0
        or not math.isfinite(market_weight_max)
        or market_weight_max < 1
        or mass.shape != (4,)
        or not np.isfinite(mass).all()
        or (mass < 0).any()
        or mass.sum() <= 0
    ):
        raise ValueError("Invalid train-only direction calibration arrays")
    counts = np.zeros((4, 2, horizons, 3), dtype=np.float64)
    market_counts = np.bincount(markets, minlength=4)
    for market in range(4):
        rows = markets == market
        if not rows.any():
            continue
        # Effective sample-sized weights preserve the configured pseudo-count strength.
        weights = sample_weights[rows] * int(rows.sum()) / sample_weights[rows].sum()
        for task in range(2):
            for horizon in range(horizons):
                counts[market, task, horizon] = np.bincount(
                    labels[rows, task, horizon], weights=weights, minlength=3
                )
    priors = (counts + smoothing) / (counts.sum(-1, keepdims=True) + 3 * smoothing)
    proportions = mass / mass.sum()
    active = mass > 0
    market_weights = np.ones(4, dtype=np.float64)
    if market_weight_max != 1:
        # Water filling satisfies E_train[w]=1 without violating the final cap.
        lower, upper = 0.0, 1.0
        for _ in range(64):
            constant = (lower + upper) / 2
            expected = np.minimum(constant, market_weight_max * proportions[active]).sum()
            if expected < 1:
                lower = constant
            else:
                upper = constant
        market_weights[active] = np.minimum(upper / proportions[active], market_weight_max)
    null = scales[None, :, None] * np.asarray([-1.0, 0.0, 1.0])[None, None, :]
    error = targets[:, :, None] - null
    quantiles = np.asarray([0.1, 0.5, 0.9])[None, None, :]
    loss = np.maximum(error * quantiles, error * (quantiles - 1)) / scales[None, :, None]
    reference = float(np.average(loss.mean((1, 2)), weights=sample_weights))
    return {
        "schema_version": 2,
        "architecture": "market_residual_v2",
        "samples": size,
        "basis": "train_zero_centered_scaled_null_quantiles_uniform_ce_v2",
        "null_quantiles_scaled": [-1.0, 0.0, 1.0],
        "references": [max(reference, 1e-8), math.log(3), math.log(3)],
        "priors": priors.tolist(),
        "market_order": list(MARKET_ORDER),
        "market_counts": market_counts.tolist(),
        "market_weights": market_weights.tolist(),
        "market_weight_basis": "capped_inverse_population_expected_train_weight_one",
        "market_weight_effective_min": float(market_weights[active].min()),
        "market_weight_effective_max": float(market_weights[active].max()),
        "market_weight_expected_train": float(np.dot(proportions, market_weights)),
        "prior_smoothing": smoothing,
        "prior_smoothing_basis": "importance_weights_rescaled_to_market_sample_count",
        "sampling": metadata,
        "pinball_guard_reference": None,
        "pinball_guard_basis": None,
    }


def calibrate_direction_references(bundle, config, dataset, device, workers, batch_plan):
    """Use bounded multiworker label reads, without running any model forward."""
    from stock_forecasting.date_market_sampler import date_market_index
    from stock_forecasting.training import ModelBatchCollator, _loader_process_options

    if dataset.split != "train":
        raise ValueError("Direction priors require the training split")
    index_path = date_market_index(dataset, requested_workers=max(1, workers.effective_workers))
    index = np.load(index_path, mmap_mode="r")
    try:
        indices, weights, metadata = calibration_indices(
            index,
            sorted({str(row["market"]) for row in dataset._index.values()}),
            min(len(dataset), config.model.direction.loss_reference_samples),
            config.data.calibration_seed,
            config.training.yearly_sampling_decay,
        )
    finally:
        index._mmap.close()
    loader = DataLoader(
        dataset,
        sampler=indices,
        batch_size=min(batch_plan.training_batch_size, 128),
        collate_fn=ModelBatchCollator.for_model(config, training=False),
        pin_memory=False,
        generator=torch.Generator().manual_seed(config.data.calibration_seed),
        **_loader_process_options(workers, persistent=False),
    )
    targets, labels, markets = [], [], []
    iterator = iter(loader)
    try:
        for batch in iterator:
            targets.append(batch["target_alpha"].float().numpy())
            labels.append(
                torch.stack(
                    [
                        direction_labels(batch[name], batch["direction_bands"])
                        for name in ("target_absolute", "target_alpha")
                    ],
                    1,
                ).numpy()
            )
            markets.append(batch["market_ids"].numpy())
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    if sum(len(value) for value in targets) != len(indices):
        raise RuntimeError("Train-only direction calibration sample is incomplete")
    metadata.update(
        {
            "calibration_seed": config.data.calibration_seed,
            "yearly_sampling_decay": config.training.yearly_sampling_decay,
            "sample_universe": dataset.sample_universe_identity,
            "horizons": list(config.data.alpha_horizons),
        }
    )
    state = references_from_arrays(
        np.concatenate(targets),
        np.concatenate(labels),
        np.concatenate(markets),
        bundle.model.alpha_head.robust_scales.detach().float().cpu().numpy(),
        weights,
        smoothing=config.model.direction.prior_smoothing,
        market_weight_max=config.model.direction.market_weight_max,
        metadata=metadata,
    )
    bundle.model.direction_head.restore_references(state)
    return bundle.model.direction_head.reference_state()


class DirectionV2Metrics(DirectionMetrics):
    """Keep population decisions comparable and report balanced decisions separately."""

    def __init__(self, horizons, *, split="validation", probability_bins=10, max_daily_entries=4096):
        super().__init__(horizons)
        if split not in ("validation", "test") or probability_bins < 2 or max_daily_entries < 1:
            raise ValueError("Invalid bounded direction evaluation configuration")
        self.split, self.bins, self.max_daily_entries = split, probability_bins, max_daily_entries
        self.balanced = DirectionMetrics(horizons)
        self.probability_slices, self.daily = {}, {}

    def append(self, output, batch):
        raw = torch.stack([output.absolute_direction_logits, output.alpha_direction_logits], 1)
        decision = torch.stack(
            [output.absolute_direction_decision_logits, output.alpha_direction_decision_logits], 1
        )
        truth = torch.stack(
            [direction_labels(batch[name], batch["direction_bands"])
             for name in ("target_absolute", "target_alpha")],
            1,
        )
        log_probability = raw.float().log_softmax(-1)
        ce = -log_probability.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
        # One host transfer avoids synchronizing independently for every diagnostic.
        values = torch.cat(
            [truth[..., None].float(), log_probability.exp(),
             decision.argmax(-1)[..., None].float(), ce[..., None]],
            -1,
        ).detach().cpu().numpy()
        self.append_probabilities(
            values[..., 0].astype(np.int64),
            values[..., 1:4],
            values[..., 4].astype(np.int64),
            batch["markets"],
            batch["cutoff_at"],
            pinball=float(output.pinball_loss),
            losses=values[..., 5],
        )

    def append_probabilities(
        self, truth, probabilities, balanced_predicted, markets, dates, *, pinball=0.0, losses=None
    ):
        truth = np.asarray(truth, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        balanced_predicted = np.asarray(balanced_predicted, dtype=np.int64)
        shape = (len(truth), 2, len(self.horizons))
        if (
            truth.shape != shape
            or balanced_predicted.shape != shape
            or probabilities.shape != (*shape, 3)
            or not np.isin(truth, [0, 1, 2]).all()
            or not np.isin(balanced_predicted, [0, 1, 2]).all()
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)
            or len(markets) != len(truth)
            or len(dates) != len(truth)
            or not len(truth)
            or not math.isfinite(pinball)
            or pinball < 0
        ):
            raise ValueError("Invalid population-probability direction evaluation batch")
        predicted = probabilities.argmax(-1)
        if losses is None:
            losses = -np.log(np.maximum(
                np.take_along_axis(probabilities, truth[..., None], -1)[..., 0], 1e-30
            ))
        losses = np.asarray(losses, dtype=np.float64)
        if losses.shape != shape or not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError("Invalid direction cross entropy")
        market_array = np.asarray([
            MARKET_ORDER[MARKET_IDS.get(str(value).upper(), 3)] for value in markets
        ])
        super().append_arrays(truth, predicted, losses, market_array, dates, pinball=pinball)
        self.balanced.append_arrays(
            truth, balanced_predicted, losses, market_array, dates, pinball=pinball
        )
        groups = {"overall": np.arange(len(truth))}
        groups.update({f"market/{m}": np.flatnonzero(market_array == m)
                       for m in np.unique(market_array)})
        for key, rows in groups.items():
            if key not in self.probability_slices:
                self.probability_slices[key] = self._probability_state()
            state = self.probability_slices[key]
            selected = probabilities[rows]
            actual, forecast = truth[rows], predicted[rows]
            confidence = selected.max(-1)
            correct = forecast == actual
            encoded = np.minimum((confidence * self.bins).astype(int), self.bins - 1)
            for task in range(2):
                for horizon in range(len(self.horizons)):
                    bins = encoded[:, task, horizon]
                    state["count"][task, horizon] += np.bincount(bins, minlength=self.bins)
                    state["confidence"][task, horizon] += np.bincount(
                        bins, weights=confidence[:, task, horizon], minlength=self.bins
                    )
                    state["correct"][task, horizon] += np.bincount(
                        bins, weights=correct[:, task, horizon], minlength=self.bins
                    )
                    state["class_ce"][task, horizon] += np.bincount(
                        actual[:, task, horizon], weights=losses[rows, task, horizon], minlength=3
                    )
            one_hot = np.eye(3)[actual]
            state["brier"] += ((selected - one_hot) ** 2).sum(-1).sum(0)
            joint = (balanced_predicted[rows, 0] == 2) & (balanced_predicted[rows, 1] == 2)
            joint_score = selected[:, :, :, 2].min(1)
            for horizon in range(len(self.horizons)):
                signals = joint[:, horizon]
                bins = np.minimum((joint_score[signals, horizon] * self.bins).astype(int),
                                  self.bins - 1)
                for column, weights in enumerate(
                    (None, actual[signals, 0, horizon] == 2,
                     (actual[signals, 0, horizon] == 2) & (actual[signals, 1, horizon] == 2))
                ):
                    state["joint_bins"][horizon, :, column] += np.bincount(
                        bins, weights=weights, minlength=self.bins
                    ).astype(np.int64)
        day_array = np.asarray([str(value)[:10] for value in dates])
        for day in np.unique(day_array):
            if day not in self.daily and len(self.daily) >= self.max_daily_entries:
                raise ValueError("Direction daily diagnostics exceeded their explicit bound")
            rows = np.flatnonzero(day_array == day)
            if day not in self.daily:
                self.daily[day] = {"population": DirectionMetrics(self.horizons),
                                   "balanced": DirectionMetrics(self.horizons)}
            entry = self.daily[day]
            for name, forecast in (("population", predicted), ("balanced", balanced_predicted)):
                entry[name].append_arrays(
                    truth[rows], forecast[rows], losses[rows], market_array[rows],
                    day_array[rows], pinball=pinball
                )

    def _probability_state(self):
        bins_shape = (2, len(self.horizons), self.bins)
        return {
            "count": np.zeros(bins_shape, dtype=np.int64),
            "confidence": np.zeros(bins_shape, dtype=np.float64),
            "correct": np.zeros(bins_shape, dtype=np.float64),
            "brier": np.zeros(bins_shape[:2], dtype=np.float64),
            "class_ce": np.zeros((*bins_shape[:2], 3), dtype=np.float64),
            "joint_bins": np.zeros((len(self.horizons), self.bins, 3), dtype=np.int64),
        }

    def _probability_summary(self, state, confusion, priors=None, *, prior_class_ce=None):
        result = {}
        for task, name in enumerate(("absolute", "alpha")):
            horizons = {}
            for position, horizon in enumerate(self.horizons):
                count = state["count"][task, position]
                total = max(int(count.sum()), 1)
                confidence, correct = (state[key][task, position] for key in ("confidence", "correct"))
                support = confusion[task, position].sum(1)
                active = support > 0
                ce = np.divide(state["class_ce"][task, position], support,
                               out=np.zeros(3), where=active)
                row = {
                    "brier": float(state["brier"][task, position] / total),
                    "ece": float(np.abs(confidence - correct).sum() / total),
                    "balanced_cross_entropy": float(ce[active].mean()) if active.any() else 0.0,
                    "confidence_bins": [
                        {"lower_inclusive": i / self.bins,
                         "upper_exclusive": (i + 1) / self.bins,
                         "includes_one": i == self.bins - 1,
                         "count": int(count[i]),
                         "mean_confidence": float(confidence[i] / max(int(count[i]), 1)),
                         "accuracy": float(correct[i] / max(int(count[i]), 1))}
                        for i in range(self.bins)
                    ],
                }
                if priors is not None or prior_class_ce is not None:
                    prior_ce = (-np.log(priors[task, position]) if priors is not None
                                else prior_class_ce[task, position])
                    reference_ce = float(np.dot(support, prior_ce) / total)
                    model_ce = float(state["class_ce"][task, position].sum() / total)
                    row["train_prior_cross_entropy"] = reference_ce
                    row["cross_entropy_skill_vs_train_prior"] = 1 - model_ce / max(reference_ce, 1e-12)
                    balanced_prior = float(prior_ce[active].mean()) if active.any() else 0.0
                    row["balanced_train_prior_cross_entropy"] = balanced_prior
                    row["balanced_cross_entropy_skill_vs_train_prior"] = (
                        1 - row["balanced_cross_entropy"] / max(balanced_prior, 1e-12)
                    )
                horizons[str(horizon)] = row
            result[name] = {"horizons": horizons}
        result["joint_up_confidence"] = {
            str(horizon): [
                {"minimum_raw_up_probability": i / self.bins,
                 "signals": int(state["joint_bins"][position, i:, 0].sum()),
                 "absolute_up_precision": float(state["joint_bins"][position, i:, 1].sum()
                     / max(int(state["joint_bins"][position, i:, 0].sum()), 1)),
                 "both_up_precision": float(state["joint_bins"][position, i:, 2].sum()
                     / max(int(state["joint_bins"][position, i:, 0].sum()), 1))}
                for i in range(self.bins)
            ]
            for position, horizon in enumerate(self.horizons)
        }
        return result

    def finish(self, head, costs):
        result = super().finish(head, costs)
        references = head.reference_state()
        if references.get("schema_version") != 2:
            raise ValueError("Main direction-v2 metrics require schema-2 train priors")
        priors = np.asarray(references["priors"], dtype=np.float64)
        if (
            priors.shape != (4, 2, len(self.horizons), 3)
            or not np.isfinite(priors).all()
            or (priors <= 0).any()
            or not np.allclose(priors.sum(-1), 1, atol=1e-6)
        ):
            raise ValueError("Train priors do not match the direction evaluation horizons")
        balanced = self.balanced.finish(None, costs)
        result["schema_version"] = 2
        result["probability_source"] = "population_logits"
        result["balanced_decision"] = balanced
        balanced["decision_source"] = "history_residual_logits_without_train_prior"
        balanced["probability_source"] = "population_logits_only"
        # The overall reference remains market conditional, not a pooled prior.
        overall_support = self.slices["overall"]["confusion"].sum(-1)
        prior_ce_sum = np.zeros_like(overall_support, dtype=np.float64)
        for market in balanced.get("market", {}):
            support = self.slices[f"market/{market}"]["confusion"].sum(-1)
            prior_ce_sum += support * -np.log(priors[MARKET_IDS.get(market.upper(), 3)])
        overall_prior_class_ce = np.divide(
            prior_ce_sum, overall_support, out=np.zeros_like(prior_ce_sum),
            where=overall_support > 0,
        )
        result["probability_diagnostics"] = {
            "bins": self.bins,
            "probabilities_are_population_predictions": True,
            "calibration_not_assumed": True,
            "overall": self._probability_summary(
                self.probability_slices["overall"], self.slices["overall"]["confusion"],
                prior_class_ce=overall_prior_class_ce,
            ),
            "market": {},
        }
        scores = {"balanced_accuracy": [], "macro_f1": []}
        for market, summary in balanced.get("market", {}).items():
            for metric in scores:
                # Equal market/task/horizon weight; retain class supports for auditing.
                scores[metric].append(float(np.mean([
                    row[metric] for task in ("absolute", "alpha")
                    for row in summary[task]["horizons"].values()
                ])))
            key = f"market/{market}"
            result["probability_diagnostics"]["market"][market] = self._probability_summary(
                self.probability_slices[key], self.slices[key]["confusion"],
                priors[MARKET_IDS.get(market.upper(), 3)],
            )
        market_macro = {key: float(np.mean(value)) for key, value in scores.items()}
        balanced["market_macro"] = market_macro
        probability_macro_keys = (
            "brier", "ece", "balanced_cross_entropy", "train_prior_cross_entropy",
            "cross_entropy_skill_vs_train_prior", "balanced_train_prior_cross_entropy",
            "balanced_cross_entropy_skill_vs_train_prior",
        )
        result["probability_diagnostics"]["market_macro"] = {
            key: float(np.mean([
                float(np.mean([row[key] for task in ("absolute", "alpha")
                               for row in summary[task]["horizons"].values()]))
                for summary in result["probability_diagnostics"]["market"].values()
            ]))
            for key in probability_macro_keys
        }
        # Confusion arrays are sufficient for block-based uncertainty estimates.
        # Avoid multiplying the detailed per-horizon JSON schema by every day.
        result["daily"] = {}
        result["daily_format"] = {
            "horizon_order": list(self.horizons),
            "confusion_axes": ["task", "horizon", "true_class", "predicted_class"],
            "joint_up_columns": ["signals", "absolute_up_correct", "both_up_correct"],
        }
        for day, values in sorted(self.daily.items()):
            population, decisions = values["population"], values["balanced"]
            entry = {"samples": population.samples, "market": {}}
            for key in population.slices:
                if key != "overall" and not key.startswith("market/"):
                    continue
                compact = {
                    "population_confusion": population.slices[key]["confusion"].tolist(),
                    "balanced_confusion": decisions.slices[key]["confusion"].tolist(),
                    "population_ce_sum": population.slices[key]["ce"].tolist(),
                    "balanced_joint_up_counts": decisions.slices[key]["joint"].tolist(),
                }
                if key == "overall":
                    entry.update(compact)
                else:
                    entry["market"][key.split("/", 1)[1]] = compact
            result["daily"][day] = entry
        anchor = references.get("pinball_guard_reference")
        if anchor is None:
            if self.split != "validation":
                raise ValueError("Holdout direction selection requires a frozen validation pinball anchor")
            anchor = result["pinball_loss"]
            basis = "pending_first_full_validation"
        else:
            basis = references.get("pinball_guard_basis", "first_full_validation")
        if not math.isfinite(anchor) or anchor <= 0:
            raise ValueError("Invalid frozen validation pinball anchor")
        tolerance = getattr(head.config, "selection_pinball_guard_tolerance", 0.01)
        weight = getattr(head.config, "selection_pinball_guard_weight", 1.0)
        if not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid direction selection pinball guard parameters")
        penalty = weight * max(result["pinball_loss"] / anchor - (1 + tolerance), 0.0)
        result["selection_loss"] = 1 - market_macro["balanced_accuracy"] + penalty
        result["selection_definition"] = {
            "direction_basis": "equal_market_task_horizon_balanced_decision_accuracy",
            "pinball_guard_reference": anchor,
            "pinball_guard_basis": basis,
            "pinball_guard_tolerance": tolerance,
            "pinball_guard_weight": weight,
            "pinball_guard_penalty": penalty,
            "fit_split": "validation",
        }
        return result
