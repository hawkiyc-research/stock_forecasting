"""Regression checks for main-only calibration and honest probability diagnostics."""

from __future__ import annotations

import json
import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stock_forecasting.direction_v2_runtime import (
    DirectionV2Metrics,
    calibration_indices,
    references_from_arrays,
)


def _index(population=100, *, rare_market=False):
    rows = []
    epoch = date(1970, 1, 1)
    ordinal = 0
    for year in (2021, 2022, 2023, 2024, 2025):
        day = (date(year, 3, 1) - epoch).days
        for code in range(3):
            count = 1 if rare_market and code == 2 else population
            for _ in range(count):
                rows.append((day * 256 + code, ordinal))
                ordinal += 1
    return np.asarray(rows, dtype=[("group", "i8"), ("ordinal", "i8")])


def _reference_data(size=12):
    target = np.asarray([[0.01, -0.03], [-0.02, 0.05], [0.0, 0.0]] * (size // 3))
    labels = np.asarray([[[2, 0], [1, 0]], [[0, 2], [0, 2]], [[1, 1], [1, 1]]] * (size // 3))
    markets = np.repeat([0, 1, 2], size // 3)
    metadata = {"market_population_mass": [100.0, 20.0, 10.0, 0.0], "calibration_seed": 59}
    return {
        "targets": target,
        "labels": labels,
        "markets": markets,
        "scales": [0.02, 0.04],
        "sample_weights": np.ones(size),
        "smoothing": 1.0,
        "market_weight_max": 3.0,
        "metadata": metadata,
    }


def _head(*, anchor=0.3, priors=None):
    priors = np.full((4, 2, 2, 3), 1 / 3) if priors is None else priors
    state = {
        "schema_version": 2,
        "references": [0.3, math.log(3), math.log(3)],
        "samples": 12,
        "priors": priors.tolist(),
        "pinball_guard_reference": anchor,
        "pinball_guard_basis": "first_full_validation" if anchor is not None else None,
    }
    return SimpleNamespace(
        reference_state=lambda: state,
        weights=lambda: torch.tensor([0.3 / math.log(3)] * 2),
        config=SimpleNamespace(selection_pinball_guard_tolerance=0.01,
                               selection_pinball_guard_weight=1.0),
    )


COSTS = SimpleNamespace(model_dump=lambda: {"scenario": "test_costs"})


def _batch():
    truth = np.asarray([[[0, 2], [0, 2]], [[1, 1], [1, 1]], [[2, 0], [2, 0]]])
    probability = np.full((*truth.shape, 3), 0.1)
    np.put_along_axis(probability, truth[..., None], 0.8, axis=-1)
    return truth, probability


def test_year_weighted_sample_is_seed_independent_of_training_and_bounded():
    index = _index()
    first = calibration_indices(index, ["US", "TWSE", "TPEX"], 300, 59, 0.8)
    second = calibration_indices(index, ["US", "TWSE", "TPEX"], 300, 59, 0.8)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == len(set(first[0])) == 300
    assert first[0] == sorted(first[0])
    for market in ("US", "TWSE", "TPEX"):
        rows = [row for row in first[2]["strata"] if row["market"] == market]
        assert sum(row["samples"] for row in rows) == 100
        assert rows[-1]["samples"] > rows[0]["samples"]
        assert rows[0]["per_window_weight"] == pytest.approx(0.8**4)


def test_rare_market_is_fully_represented_without_replacement():
    index = _index(rare_market=True)
    selected, weights, metadata = calibration_indices(index, ["US", "TWSE", "TPEX"], 300, 59, 0.8)
    assert len(selected) == len(set(selected)) == 300
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert sum(row["samples"] for row in metadata["strata"] if row["market"] == "TPEX") == 5


def test_market_names_follow_canonical_model_ids_not_index_code_order():
    index = _index()
    _, _, metadata = calibration_indices(index, ["TPEX", "US", "TWSE"], 300, 59, None)
    assert {row["market"] for row in metadata["strata"]} == {"US", "TWSE", "TPEX"}
    assert metadata["market_population_mass"] == [500.0, 500.0, 500.0, 0.0]


def test_reference_scale_uses_no_initial_model_loss_and_is_json_roundtrippable():
    arguments = _reference_data()
    reference = references_from_arrays(**arguments)
    assert reference["schema_version"] == 2
    assert reference["references"][1:] == [math.log(3), math.log(3)]
    assert reference["basis"] == "train_zero_centered_scaled_null_quantiles_uniform_ce_v2"
    assert reference["pinball_guard_reference"] is None
    assert reference["pinball_guard_basis"] is None
    assert json.loads(json.dumps(reference, allow_nan=False)) == reference
    np.testing.assert_allclose(np.asarray(reference["priors"]).sum(-1), 1)
    assert (np.asarray(reference["priors"]) > 0).all()
    assert reference == references_from_arrays(**arguments)


def test_year_weights_change_prior_without_random_model_reference():
    arguments = _reference_data()
    arguments["sample_weights"] = np.asarray([1, 4, 1] * 4)
    weighted = references_from_arrays(**arguments)
    unweighted = references_from_arrays(**_reference_data())
    assert weighted["priors"] != unweighted["priors"]
    assert weighted["references"][1:] == unweighted["references"][1:]
    population = np.asarray(arguments["metadata"]["market_population_mass"])
    np.testing.assert_allclose(np.dot(population / population.sum(), weighted["market_weights"]), 1)
    np.testing.assert_allclose(np.asarray(weighted["priors"])[3], 1 / 3)


@pytest.mark.parametrize("cap", [1.0, 3.0])
def test_effective_market_weights_respect_cap_under_extreme_imbalance(cap):
    arguments = _reference_data()
    arguments["metadata"]["market_population_mass"] = [0.99, 0.005, 0.005, 0]
    arguments["market_weight_max"] = cap
    reference = references_from_arrays(**arguments)
    weights = np.asarray(reference["market_weights"])
    assert weights.max() <= cap
    assert weights[3] == 1
    assert np.dot(arguments["metadata"]["market_population_mass"], weights) == pytest.approx(1)
    assert reference["market_weight_effective_max"] <= cap
    assert reference["market_weight_expected_train"] == pytest.approx(1)
    if cap == 1:
        np.testing.assert_array_equal(weights, np.ones(4))
    else:
        np.testing.assert_allclose(weights, [0.97 / 0.99, 3, 3, 1])


@pytest.mark.parametrize("key,value", [
    ("scales", [0.0, 0.04]),
    ("smoothing", 0.0),
    ("market_weight_max", 0.5),
    ("sample_weights", np.zeros(12)),
])
def test_invalid_train_reference_inputs_fail(key, value):
    arguments = _reference_data()
    arguments[key] = value
    with pytest.raises(ValueError):
        references_from_arrays(**arguments)


@pytest.mark.parametrize("mass", [[0, 0, 0, 0], [1, 2], [1, -1, 0, 0], [1, np.nan, 0, 0]])
def test_invalid_population_mass_is_rejected_before_normalization(mass):
    arguments = _reference_data()
    arguments["metadata"] = {"market_population_mass": mass}
    with pytest.raises(ValueError, match="train-only"):
        references_from_arrays(**arguments)


def test_every_nonempty_year_remains_represented_with_sufficient_budget():
    index = _index(population=1)
    _, _, metadata = calibration_indices(index, ["US", "TWSE", "TPEX"], 15, 59, 0.01)
    assert len(metadata["strata"]) == 15
    assert all(row["samples"] == 1 for row in metadata["strata"])


def test_population_and_balanced_decisions_remain_separate():
    truth, probability = _batch()
    metric = DirectionV2Metrics([1, 5])
    metric.append_probabilities(truth, probability, np.zeros_like(truth), ["US"] * 3,
                                ["2026-01-02"] * 3, pinball=0.3)
    result = metric.finish(_head(), COSTS)
    assert result["overall"]["absolute"]["aggregate"]["accuracy"] == 1
    assert result["balanced_decision"]["overall"]["absolute"]["aggregate"]["accuracy"] == 1 / 3
    assert result["selection_loss"] == pytest.approx(2 / 3)
    assert result["probability_source"] == "population_logits"
    probability_metrics = result["probability_diagnostics"]["market"]["US"]["absolute"]["horizons"]["1"]
    assert probability_metrics["brier"] == pytest.approx(0.06)
    assert probability_metrics["ece"] == pytest.approx(0.2)
    assert probability_metrics["balanced_cross_entropy"] == pytest.approx(-math.log(0.8))
    assert probability_metrics["cross_entropy_skill_vs_train_prior"] > 0


def test_market_macro_does_not_hide_small_market_under_large_market_count():
    truth, probability = _batch()
    truth = np.concatenate([np.tile(truth, (100, 1, 1)), truth])
    probability = np.concatenate([np.tile(probability, (100, 1, 1, 1)), probability])
    decisions = truth.copy()
    decisions[-3:] = (truth[-3:] + 1) % 3
    metric = DirectionV2Metrics([1, 5])
    metric.append_probabilities(truth, probability, decisions, ["US"] * 300 + ["TPEX"] * 3,
                                ["2026-01-02"] * 303, pinball=0.3)
    result = metric.finish(_head(), COSTS)
    assert result["balanced_decision"]["market_macro"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result["selection_loss"] == pytest.approx(0.5)


def test_population_prior_reference_keeps_market_conditioning_when_aggregated():
    truth, probability = _batch()
    priors = np.full((4, 2, 2, 3), 1 / 3)
    priors[0] = [0.8, 0.1, 0.1]
    priors[1] = [0.1, 0.8, 0.1]
    metric = DirectionV2Metrics([1, 5])
    metric.append_probabilities(truth, probability, truth, ["us", "twse", "TPEX"],
                                ["2026-01-02"] * 3, pinball=0.3)
    result = metric.finish(_head(priors=priors), COSTS)
    row = result["probability_diagnostics"]["overall"]["absolute"]["horizons"]["1"]
    assert row["train_prior_cross_entropy"] == pytest.approx(
        (-math.log(0.8) - math.log(0.8) + math.log(3)) / 3
    )
    assert set(result["market"]) == {"US", "TWSE", "TPEX"}
    assert result["probability_diagnostics"]["market_macro"]["ece"] == pytest.approx(0.2)


def test_validation_anchor_is_read_only_and_pinball_regression_adds_penalty():
    truth, probability = _batch()
    head = _head(anchor=0.3)
    metric = DirectionV2Metrics([1, 5], split="test")
    metric.append_probabilities(truth, probability, truth, ["US"] * 3, ["2026-01-02"] * 3,
                                pinball=0.33)
    state = json.dumps(head.reference_state(), sort_keys=True)
    result = metric.finish(head, COSTS)
    assert result["selection_loss"] == pytest.approx(0.09)
    assert result["selection_definition"]["pinball_guard_reference"] == 0.3
    assert json.dumps(head.reference_state(), sort_keys=True) == state


def test_first_validation_anchor_is_pending_but_test_cannot_fit_one():
    truth, probability = _batch()
    for split in ("validation", "test"):
        metric = DirectionV2Metrics([1, 5], split=split)
        metric.append_probabilities(truth, probability, truth, ["US"] * 3, ["2026-01-02"] * 3,
                                    pinball=0.3)
        head = _head(anchor=None)
        if split == "test":
            with pytest.raises(ValueError, match="frozen validation"):
                metric.finish(head, COSTS)
        else:
            result = metric.finish(head, COSTS)
            assert result["selection_definition"]["pinball_guard_basis"] == "pending_first_full_validation"
            assert head.reference_state()["pinball_guard_reference"] is None


def test_stream_partition_preserves_confusion_and_daily_counts():
    truth, probability = _batch()
    metrics = [DirectionV2Metrics([1, 5]), DirectionV2Metrics([1, 5])]
    markets = ["US", "TWSE", "TPEX"]
    dates = ["2026-01-02", "2026-01-02", "2026-01-05"]
    metrics[0].append_probabilities(truth, probability, truth, markets, dates, pinball=0.3)
    for start in range(3):
        metrics[1].append_probabilities(truth[start:start + 1], probability[start:start + 1],
                                       truth[start:start + 1], markets[start:start + 1],
                                       dates[start:start + 1], pinball=0.3)
    results = [metric.finish(_head(), COSTS) for metric in metrics]
    assert results[0]["daily"] == results[1]["daily"]
    assert results[0]["overall"] == results[1]["overall"]
    assert results[0]["balanced_decision"] == results[1]["balanced_decision"]
    assert sum(entry["samples"] for entry in results[0]["daily"].values()) == 3


def test_confidence_joint_precision_uses_population_probabilities():
    truth, probability = _batch()
    decisions = np.full_like(truth, 2)
    metric = DirectionV2Metrics([1, 5])
    metric.append_probabilities(truth, probability, decisions, ["US"] * 3,
                                ["2026-01-02"] * 3, pinball=0.3)
    result = metric.finish(_head(), COSTS)
    curve = result["probability_diagnostics"]["overall"]["joint_up_confidence"]["1"]
    assert curve[0]["signals"] == 3
    assert curve[5]["signals"] == 1
    assert curve[5]["both_up_precision"] == 1
    assert curve[5]["absolute_up_precision"] == 1


def test_daily_bound_and_invalid_probabilities_fail_explicitly():
    truth, probability = _batch()
    metric = DirectionV2Metrics([1, 5], max_daily_entries=1)
    with pytest.raises(ValueError, match="explicit bound"):
        metric.append_probabilities(truth, probability, truth, ["US"] * 3,
                                    ["2026-01-02", "2026-01-03", "2026-01-03"])
    invalid = probability.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="population-probability"):
        DirectionV2Metrics([1, 5]).append_probabilities(
            truth, invalid, truth, ["US"] * 3, ["2026-01-02"] * 3
        )
