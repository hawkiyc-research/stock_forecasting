"""The dataset revision label selects the corporate-action scope and shares one provider cache."""

from __future__ import annotations

import pytest

from stock_forecasting.dataset_identity import (
    CORPORATE_ACTION_MODES,
    PROVIDER_CACHE_REVISION,
    dataset_revision_policy,
)


@pytest.mark.parametrize(
    ("label", "mode"),
    [
        ("v1", "dividends"),
        ("v1.1", "dividends"),
        ("v1-retrain", "dividends"),
        ("v2", "full"),
        ("v2.3", "full"),
        ("v2_x", "full"),
    ],
)
def test_labels_map_to_corporate_action_modes_and_share_the_v1_cache(label: str, mode: str) -> None:
    policy = dataset_revision_policy(label)

    assert policy == {
        "revision": label,
        "corporate_actions": mode,
        "cache_revision": PROVIDER_CACHE_REVISION,
    }
    assert mode in CORPORATE_ACTION_MODES
    assert PROVIDER_CACHE_REVISION == "v1"


@pytest.mark.parametrize("label", ["", "v3", "v10", "v11", "v21", "full", "dividends", None])
def test_unknown_labels_are_rejected(label: object) -> None:
    with pytest.raises(ValueError):
        dataset_revision_policy(label)
