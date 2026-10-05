"""Small same-date, same-market auxiliary ranking objective."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

MARKET_IDS = {"US": 0, "TWSE": 1, "TPEX": 2}
MAX_RANKING_CANDIDATES = 512


def market_ids(values, device) -> Tensor:
    return torch.tensor([MARKET_IDS.get(str(v).upper(), 3) for v in values], device=device)


def ranking_groups(dates, markets, symbols, device) -> tuple[Tensor, Tensor]:
    groups, securities = {}, {}
    group_ids, security_ids = [], []
    for day, market, symbol in zip(dates, markets, symbols, strict=True):
        group_ids.append(groups.setdefault((str(day), str(market)), len(groups)))
        security_ids.append(securities.setdefault(str(symbol), len(securities)))
    return torch.tensor(group_ids, device=device), torch.tensor(security_ids, device=device)


def eligible_ranking_pairs(groups: Tensor, securities: Tensor) -> Tensor | None:
    """Prepare deterministic pair membership in loader workers, without RNG.

    Larger batches retain the legacy random 512-security subset on the GPU.
    Its permutation cannot be moved to a worker without changing checkpoint RNG.
    """
    if groups.device.type != "cpu" or securities.device.type != "cpu":
        raise ValueError("Ranking membership must be prepared on CPU before pinning")
    if groups.ndim != 1 or groups.shape != securities.shape:
        raise ValueError("Ranking identities must be aligned one-dimensional tensors")
    if len(groups) > MAX_RANKING_CANDIDATES:
        return None
    pairs = torch.triu_indices(len(groups), len(groups), offset=1)
    valid = (groups[pairs[0]] == groups[pairs[1]]) & (securities[pairs[0]] != securities[pairs[1]])
    return pairs[:, valid].contiguous()


def same_date_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    scales: Tensor,
    groups: Tensor,
    securities: Tensor,
    max_pairs: int,
    *,
    eligible_pairs: Tensor | None = None,
) -> Tensor:
    """Never rank different dates/markets or duplicate padded copies of one stock."""
    if eligible_pairs is None:
        candidates = torch.arange(len(targets), device=targets.device)
        if len(candidates) > MAX_RANKING_CANDIDATES:
            candidates = candidates[
                torch.randperm(len(candidates), device=targets.device)[:MAX_RANKING_CANDIDATES]
            ]
        pairs = candidates[
            torch.triu_indices(len(candidates), len(candidates), offset=1, device=targets.device)
        ]
        valid = (groups[pairs[0]] == groups[pairs[1]]) & (
            securities[pairs[0]] != securities[pairs[1]]
        )
        pairs = pairs[:, valid]
    else:
        if (
            len(targets) > MAX_RANKING_CANDIDATES
            or eligible_pairs.ndim != 2
            or eligible_pairs.shape[0] != 2
            or eligible_pairs.device != targets.device
            or eligible_pairs.dtype != torch.long
        ):
            raise ValueError("Prepared ranking pairs are incompatible with this batch")
        pairs = eligible_pairs
    if pairs.shape[1] == 0:
        return predictions.sum() * 0
    if pairs.shape[1] > max_pairs:
        pairs = pairs[:, torch.randperm(pairs.shape[1], device=pairs.device)[:max_pairs]]
    left, right = pairs
    delta = (targets[left].float() - targets[right].float()) / scales
    if predictions.ndim == 2:
        # Independent ranking scores are dimensionless, not return forecasts.
        gap = predictions[left].float() - predictions[right].float()
    else:
        gap = (predictions[left, :, 1].float() - predictions[right, :, 1].float()) / scales
    valid = torch.isfinite(delta) & (delta.abs() > 0.01)
    losses = F.softplus(-delta.nan_to_num().sign() * gap)
    return losses.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
