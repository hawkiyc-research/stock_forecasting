"""Cloud-only numerical tests; do not import this module in the local control plane."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import stock_forecasting.signal_diagnostic_cloud as diagnostic
import stock_forecasting.training as training
from stock_forecasting.signal_evaluation import SignalEvaluationStore


class RawFixture(Dataset):
    horizons = tuple(range(1, 15))
    window_size = 128
    symbol_cache_size = 4

    def __init__(self):
        dates = pd.bdate_range("2024-01-01", periods=142, tz="UTC")
        self.frame = pd.DataFrame(
            {
                "timestamp": dates,
                "open": 100 + np.arange(142) * 0.1,
                "high": 102 + np.arange(142) * 0.1,
                "low": 99 + np.arange(142) * 0.1,
                "close": 101 + np.arange(142) * 0.1,
                "volume": np.full(142, 1000.0),
                "adjusted_close": (101 + np.arange(142) * 0.1) * 1.1,
            }
        )

    def __len__(self):
        return 100

    def __getitem__(self, _index):
        return {"sample": "fixture"}

    def _sample_frames(self, _ordinal):
        return (
            "S",
            {"market": "US"},
            "B",
            self.frame,
            self.frame,
            127,
            pd.DatetimeIndex(self.frame["timestamp"].iloc[128:142]),
            self.frame.iloc[128:142],
        )

    def _aligned_rows(self, _symbol, frame, timestamps):
        return frame.set_index("timestamp", drop=False).loc[timestamps].reset_index(drop=True)

    def array_batch(self, _indices, **_kwargs):
        values = self.frame[["open", "high", "low", "close", "volume"]].iloc[:128].to_numpy()
        return {"asset": values[None], "benchmark": values[None], "targets": np.zeros((1, 14))}


def test_raw_audit_reconstructs_next_open_labels_without_changing_alpha():
    source = RawFixture()
    case = diagnostic.RawLabelAuditDataset(source, [{"source_ordinal": 0, "row": 0}])[0]
    expected = math.log(
        source.frame["adjusted_close"].iloc[128] / (source.frame["open"].iloc[128] * 1.1)
    )
    assert abs(case["recomputed_absolute_log_returns"][0] - expected) < 1e-12
    assert case["maximum_internal_alpha_disagreement"] == 0
    assert len(case["point_in_time_input_asset"]) == 128
    assert len(case["asset_vendor_raw_and_adjusted"]["values"]) == 142
    assert len(case["recomputed_alpha_log_returns"]) == 14


@dataclass
class WorkerFixture:
    effective_workers: int = 2
    symbol_cache_size_per_worker: int = 4
    active_persistent_pools: int = 1


def test_fixed_train_membership_is_independent_of_model_seed(monkeypatch):
    source = RawFixture()
    monkeypatch.setattr(diagnostic, "open_clean_dataset", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(diagnostic, "with_direction_targets", lambda dataset, _config: dataset)
    monkeypatch.setattr(training, "_requested_dataloader_workers", lambda _config: (2, "fixture"))
    monkeypatch.setattr(
        training, "plan_dataloader_workers", lambda *_args, **_kwargs: WorkerFixture()
    )
    monkeypatch.setattr(
        training,
        "resolve_runtime_batch_plan",
        lambda *_args, **_kwargs: SimpleNamespace(evaluation_batch_size=8),
    )
    monkeypatch.setattr(training, "plan_runtime_prefetch", lambda workers, **_kwargs: workers)
    monkeypatch.setattr(
        training, "_loader_process_options", lambda *_args, **_kwargs: {"num_workers": 2}
    )
    collator = SimpleNamespace(estimated_batch_bytes=lambda *_args: 1024)
    monkeypatch.setattr(
        training.ModelBatchCollator, "for_model", lambda *_args, **_kwargs: collator
    )
    first, indices = diagnostic.fixed_train_loader(
        SimpleNamespace(), None, torch.device("cpu"), 16, 59
    )
    second, again = diagnostic.fixed_train_loader(
        SimpleNamespace(), None, torch.device("cpu"), 16, 59
    )
    assert indices == again and len(indices) == 16 and indices == sorted(set(indices))
    assert first.num_workers == second.num_workers == 2
    assert source.__len__() == 100


def test_audit_pairing_is_bounded_and_uses_same_date_market_controls(tmp_path):
    store = SignalEvaluationStore(tmp_path / "store", 40, range(1, 15), [0.05] * 14, "alpha_point")
    try:
        store.append(
            np.full((40, 14), 0.1),
            np.zeros((40, 14)),
            symbols=[f"S{i}" for i in range(40)],
            dates=["2025-06-02T00:00:00+00:00"] * 40,
            markets=["US"] * 40,
            asset_types=["stock"] * 40,
            providers=["fixture"] * 40,
            past_drawdown=np.array([0.5] * 8 + [0.1] * 32),
        )
        cases = diagnostic.audit_candidates(store, 8)
        assert 0 < len(cases) <= 8
        for case in cases:
            if case["role"].startswith("high_loss"):
                control = case["matched_control_row"]
                assert control is not None
                assert store.drawdown[case["row"]] >= 0.4 and store.drawdown[control] < 0.2
                assert store.metadata[case["row"]]["date"] == store.metadata[control]["date"]
    finally:
        store.close()
