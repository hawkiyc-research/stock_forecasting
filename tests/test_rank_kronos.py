"""Kronos cross-sectional rank task: head, loss, batch plumbing and full evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_signal_model import batch as signal_batch
from test_signal_model import model as signal_model
from torch.utils.data import DataLoader, SequentialSampler

MIN_GROUP = 3
TASK = "cross_sectional_rank"


@pytest.fixture
def ranked_kronos(tmp_path, monkeypatch):
    from conftest import build_rank_config

    from stock_forecasting.data.rank_labels import ensure_split_rank_labels
    from stock_forecasting.data.sample_universe import validated_split_roots

    # The historical scale branch needs at least 61 observed bars, as in production.
    config = build_rank_config(tmp_path, monkeypatch, window_size=128, sessions=600)
    config.model.prediction_task = TASK
    config.model.rank_min_group_size = MIN_GROUP
    config.model.feature_mode = "combined"
    config.model.explicit_output_scale = True
    config.model.market_aware = True
    config.runtime.log_root = tmp_path / "logs"
    roots = {split: str(root) for split, root in validated_split_roots(config).items()}
    ensure_split_rank_labels(config, roots, workers=1)
    return config


def open_ranked(config, split):
    from stock_forecasting.data.sample_universe import open_clean_dataset
    from stock_forecasting.rank_data import with_rank_labels

    return with_rank_labels(open_clean_dataset(config, split), config)


def test_rank_head_scores_are_unscaled_and_loss_masks_missing_labels():
    instance = signal_model(TASK)
    target = torch.randn(4, 14, generator=torch.Generator().manual_seed(3))
    target[1] = float("nan")
    target[2, :5] = float("nan")
    output = instance(**signal_batch(), target_rank=target)
    assert output.rank_score.shape == (4, 14)
    assert output.alpha_point is None and output.absolute_direction_logits is None
    assert output.point_loss is None and output.classification_loss is None
    assert output.loss is output.rank_loss
    valid = torch.isfinite(target)
    per_horizon = [
        ((output.rank_score[valid[:, i], i] - target[valid[:, i], i]) ** 2).mean()
        for i in range(14)
    ]
    torch.testing.assert_close(output.loss, torch.stack(per_horizon).mean())
    output.loss.backward()
    grad = dict(instance.named_parameters())["alpha_head.prediction.weight"].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    instance.eval()
    with torch.inference_mode():
        before = instance(**signal_batch())
        # Gauss-rank scores are unit scale; the alpha return scales must not touch them.
        instance.alpha_head.robust_scales.mul_(10)
        after = instance(**signal_batch())
    assert before.loss is None
    torch.testing.assert_close(before.rank_score, after.rank_score, rtol=0, atol=0)


def test_rank_loss_rejects_other_tasks_and_misaligned_targets():
    point = signal_model("alpha_point").alpha_head
    with pytest.raises(ValueError, match="cross_sectional_rank"):
        point.rank_loss(torch.zeros(2, 14), torch.zeros(2, 14))
    rank = signal_model(TASK).alpha_head
    with pytest.raises(ValueError, match="horizons"):
        rank.rank_loss(torch.zeros(2, 14), torch.zeros(2, 13))


def test_rank_items_collate_move_and_reach_only_the_rank_model(ranked_kronos):
    from stock_forecasting import training

    source = open_ranked(ranked_kronos, "validation")
    try:
        indices = [0, 5, len(source) - 1]
        items = source.__getitems__(indices)
        expected = source.array_batch(indices)["rank_targets"]
        np.testing.assert_array_equal(
            torch.stack([item["target_rank"] for item in items]).numpy(), expected
        )
        batch = training.ModelBatchCollator.for_model(ranked_kronos, training=True)(items)
        assert batch["target_rank"].shape == (3, 14)
        moved = training._move_batch_to_device(batch, torch.device("cpu"))
        captured = {}

        class Recorder(torch.nn.Module):
            def forward(self, *args, **kwargs):
                captured.clear()
                captured.update(kwargs)

        bundle = SimpleNamespace(model=Recorder())
        training.forward_batch(bundle, moved, ranked_kronos, torch.device("cpu"))
        torch.testing.assert_close(captured["target_rank"], batch["target_rank"])
        moved.pop("target_rank")
        training.forward_batch(bundle, moved, ranked_kronos, torch.device("cpu"))
        assert "target_rank" not in captured
    finally:
        source.close()


def test_full_rank_evaluation_streams_labels_and_selects_on_rank_ic(ranked_kronos):
    from stock_forecasting import training
    from stock_forecasting.signal_training import evaluate_signal_loader

    source = open_ranked(ranked_kronos, "validation")
    try:
        loader = DataLoader(
            source,
            batch_size=64,
            sampler=SequentialSampler(source),
            collate_fn=training.ModelBatchCollator.for_model(ranked_kronos, training=False),
            num_workers=0,
        )
        bundle = SimpleNamespace(model=signal_model(TASK))
        result = evaluate_signal_loader(bundle, loader, ranked_kronos, torch.device("cpu"))
    finally:
        source.close()
    assert result["prediction_task"] == TASK and result["samples"] == len(source)
    assert result["rank_min_group_size"] == MIN_GROUP
    assert result["execution"]["full_population"] is True
    selection = result["rank_ic_selection"]
    assert selection["markets"] == ["TWSE", "US"]
    assert result["aggregate"]["selection_score"] == pytest.approx(
        -selection["rank_ic_market_macro"]
    )
    assert result["primary_5d"]["selection_score"] == result["aggregate"]["selection_score"]
    assert result["aggregate"]["labelled_fraction"] > 0.9


def test_rank_reports_exports_and_single_inference():
    from stock_forecasting.checkpointing import _valid_output_schema
    from stock_forecasting.cli.export_signal_predictions import prediction_shape
    from stock_forecasting.cli.infer import _signal_forecast_payload
    from stock_forecasting.validation_benchmark import _comparison_summary

    def metrics(macro, twse):
        return {
            "aggregate": {
                "selection_score": -macro,
                "rank_ic_market_macro": macro,
                "rank_mse": 0.9,
                "rank_label_correlation": macro,
            },
            "cross_sectional": {"by_market": {"TWSE": {"5d": {"rank_ic_mean": twse}}}},
        }

    summary = _comparison_summary(
        {
            "kronos_full": {"state": "complete", "metrics": metrics(0.06, 0.08)},
            "gbdt": {"state": "complete", "seed_results": {"42": metrics(0.04, 0.05)}},
        },
        TASK,
    )
    assert summary["selection_score_ranking"] == ["kronos_full", "gbdt"]
    row = next(row for row in summary["rows"] if row["model"] == "kronos_full")
    assert row["rank_ic_market_macro"] == 0.06 and row["rank_ic_by_market_5d"] == {"TWSE": 0.08}
    assert prediction_shape(TASK, 7, 14) == (7, 14)
    assert _valid_output_schema(
        {"prediction_task": TASK, "model_output_schema_version": "signal-1.0"}
    )
    config = SimpleNamespace(
        model=SimpleNamespace(prediction_task=TASK), data=SimpleNamespace(alpha_horizons=[1, 2])
    )
    output = SimpleNamespace(rank_score=torch.tensor([[0.5, -1.0]]))
    payload = _signal_forecast_payload(output, config)
    assert payload["by_horizon"] == {"1d": {"rank_score": 0.5}, "2d": {"rank_score": -1.0}}
