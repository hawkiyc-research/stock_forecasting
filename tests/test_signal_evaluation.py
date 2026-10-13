"""Numerical contracts for full signal evaluation; run in the cloud test environment."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stock_forecasting.evaluation_protocol import sample_membership
from stock_forecasting.signal_evaluation import (
    SignalEvaluationStore,
    SignalMoments,
    direction_labels,
)


def metadata(n):
    return {
        "symbols": [f"{i % 4}.US" for i in range(n)],
        "dates": [f"2026-01-{1 + i // 4:02d}" for i in range(n)],
        "markets": ["US"] * n,
        "asset_types": ["stock"] * n,
        "providers": ["fixture"] * n,
    }


def test_label_boundaries_match_fp32_torch_and_simple_returns():
    from stock_forecasting.models.signal import fixed_direction_labels

    values = np.array([np.log1p(-0.01), 0.0, np.log1p(0.01), -0.02, 0.02], dtype=np.float32)
    np.testing.assert_array_equal(direction_labels(values, 0.01), [1, 1, 1, 0, 2])
    np.testing.assert_array_equal(
        direction_labels(values, 0.01), fixed_direction_labels(torch.from_numpy(values)).numpy()
    )


@pytest.mark.parametrize("task", ["alpha_point", "absolute_direction"])
def test_chunked_population_loss_matches_torch_and_membership(tmp_path, task):
    rng = np.random.default_rng(123)
    y = rng.normal(0, 0.03, (12, 14)).astype(np.float32)
    p = rng.normal(
        0, 0.03 if task == "alpha_point" else 1, (12, 14) if task == "alpha_point" else (12, 14, 3)
    ).astype(np.float32)
    absolute = y + 0.004
    scales = np.linspace(0.01, 0.08, 14).tolist()
    store = SignalEvaluationStore(tmp_path, 12, list(range(1, 15)), scales, task)
    meta = metadata(12)
    try:
        for begin, end in ((0, 5), (5, 12)):
            store.append(
                y[begin:end],
                p[begin:end],
                target_absolute=absolute[begin:end],
                past_drawdown=np.linspace(0, 0.8, 12)[begin:end],
                **{key: value[begin:end] for key, value in meta.items()},
            )
        result = store.finish()
        if task == "alpha_point":
            expected = F.smooth_l1_loss(
                torch.tensor(p) / torch.tensor(scales), torch.tensor(y) / torch.tensor(scales)
            ).item()
            assert (
                result["cross_sectional"]["by_market"]["US"]["5d"]["rank_ic_method"]
                == "spearman_average_ties_v1"
            )
            assert "neutral_return_threshold" not in result
        else:
            labels = direction_labels(absolute, 0.01)
            expected = F.cross_entropy(
                torch.tensor(p).flatten(0, 1), torch.tensor(labels).flatten()
            ).item()
            assert sum(result["per_horizon"]["5d"]["class_support"]) == 12
            assert result["neutral_return_threshold"] == 0.01
            assert result["class_order"] == ["down", "neutral", "up"]
        assert result["loss"] == pytest.approx(expected, rel=1e-6)
        assert result["primary_5d"]["selection_score"] == result["aggregate"]["selection_score"]
        assert result["sample_membership"] == sample_membership(meta["symbols"], meta["dates"])
        assert sum(v["samples"] for v in result["subgroups"]["past_drawdown"].values()) == 12
        assert result["drawdown_diagnostics"]["filters_samples"] is False
        assert "pinball" not in json.dumps(result) and "interval_coverage" not in json.dumps(result)
        assert len(result["daily_signal_loss"]) == 3
        np.testing.assert_array_equal(store.targets, y)
    finally:
        store.close()


def test_missing_class_does_not_report_valid_three_class_balanced_accuracy():
    moments = SignalMoments(list(range(1, 15)), [1.0] * 14, "absolute_direction")
    moments.add(np.zeros((10, 14)), np.zeros((10, 14, 3)))
    result = moments.result()["per_horizon"]["5d"]
    assert result["balanced_accuracy_status"] == "missing_target_class"
    assert result["predicted_class_share"] == [1.0, 0.0, 0.0]
    assert result["majority_class_accuracy"] == 1.0


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_beta_is_rejected(beta):
    with pytest.raises(ValueError, match="Invalid"):
        SignalMoments([1], [1.0], "alpha_point", beta=beta)


def test_context_drawdown_ignores_left_and_right_padding():
    from stock_forecasting.signal_training import _past_drawdown

    close = torch.tensor([[999.0, 100.0, 80.0, 90.0, 0.0], [0.0, 50.0, 60.0, 30.0, 999.0]])
    mask = torch.tensor([[False, True, True, True, False]] * 2)
    torch.testing.assert_close(_past_drawdown(close, mask), torch.tensor([0.2, 0.5]))


def test_full_coverage_duplicate_and_absent_labels_fail_closed(tmp_path):
    store = SignalEvaluationStore(tmp_path, 4, list(range(1, 15)), [1.0] * 14, "absolute_direction")
    try:
        with pytest.raises(ValueError, match="Incomplete"):
            store.finish()
        with pytest.raises(ValueError, match="absolute-return"):
            store.append(np.zeros((4, 14)), np.zeros((4, 14, 3)), **metadata(4))
        meta = metadata(4)
        meta["symbols"] = ["duplicate"] * 4
        store.append(
            np.zeros((4, 14)), np.zeros((4, 14, 3)), target_absolute=np.zeros((4, 14)), **meta
        )
        with pytest.raises(ValueError, match="Duplicate"):
            store.finish()
    finally:
        store.close()
