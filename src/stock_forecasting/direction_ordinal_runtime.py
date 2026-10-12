"""Bounded ordinal diagnostics and probability-aware checkpoint selection."""

from __future__ import annotations

import numpy as np

from stock_forecasting.direction_v2_runtime import DirectionV2Metrics
from stock_forecasting.models.ranking import MARKET_IDS


def ordinal_reference(state):
    return {
        **state,
        "schema_version": 3,
        "architecture": "market_ordinal_v3",
        "conditional_order": ["above_lower", "above_upper_given_above_lower"],
    }


def calibrate_direction_references(bundle, config, dataset, device, workers, batch_plan):
    from stock_forecasting.direction_v2_runtime import calibrate_direction_references as calibrate

    return calibrate(
        bundle, config, dataset, device, workers, batch_plan, reference_transform=ordinal_reference
    )


def decision_audit(matrix, *, support_min, prediction_min):
    """Class-order errors distinguish opposite trades from adjacent neutral errors."""
    matrix = np.asarray(matrix, dtype=np.float64)
    total = max(float(matrix.sum()), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    supported = support / total >= support_min
    deficit = np.maximum(1 - predicted / total / prediction_min, 0)
    tails = support[[0, 2]] > 0
    opposite = np.divide(matrix[[0, 2], [2, 0]], support[[0, 2]], out=np.zeros(2), where=tails)
    distances = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    return {
        "ordinal_mae": float((matrix * distances).sum() / total),
        "opposite_error_rate": float((matrix[0, 2] + matrix[2, 0]) / total),
        "balanced_opposite_error": float(opposite[tails].mean()) if tails.any() else 0.0,
        "collapse_deficit": float(deficit[supported].mean()) if supported.any() else 0.0,
        "collapsed_classes": [
            name
            for i, name in enumerate(("down", "neutral", "up"))
            if supported[i] and predicted[i] / total < prediction_min
        ],
        "true_class_shares": (support / total).tolist(),
        "predicted_class_shares": (predicted / total).tolist(),
    }


class DirectionOrdinalMetrics(DirectionV2Metrics):
    """Stream all windows; retain V2 diagnostics and add honest population selection."""

    def finish(self, head, costs):
        references = head.reference_state()
        if (
            references.get("schema_version") != 3
            or references.get("architecture") != "market_ordinal_v3"
        ):
            raise ValueError("Ordinal metrics require ordinal-v3 train references")
        # The parent summarizes the same three population probabilities and frozen anchor.
        result = super().finish(head, costs)
        result["schema_version"] = 3
        result["loss_reference"] = references
        result["architecture"] = "market_ordinal_v3"
        result["probability_structure"] = "ordered_conditional_chain"
        result["balanced_decision"]["decision_source"] = (
            "population_log_probabilities_minus_train_log_prior"
        )
        priors = np.asarray(references["priors"], dtype=np.float64)
        cfg = head.config
        support_min, prediction_min = (
            cfg.selection_class_support_min,
            cfg.selection_prediction_share_min,
        )
        if not 0 < support_min <= 0.5 or not 0 < prediction_min <= 0.1:
            raise ValueError("Invalid ordinal class-collapse thresholds")
        components, audits = [], {}
        for market in result.get("market", {}):
            prior = priors[MARKET_IDS.get(market.upper(), 3)]
            matrix = self.slices[f"market/{market}"]["confusion"]
            probability = result["probability_diagnostics"]["market"][market]
            audits[market] = {}
            for task_index, task in enumerate(("absolute", "alpha")):
                audits[market][task] = {}
                for position, horizon in enumerate(self.horizons):
                    row = result["market"][market][task]["horizons"][str(horizon)]
                    count = max(int(matrix[task_index, position].sum()), 1)
                    frequencies = matrix[task_index, position].sum(1) / count
                    p = prior[task_index, position]
                    prior_brier = float(1 + np.dot(p, p) - 2 * np.dot(frequencies, p))
                    diagnostic = probability[task]["horizons"][str(horizon)]
                    brier_ratio = diagnostic["brier"] / max(prior_brier, 1e-12)
                    ce_ratio = row["cross_entropy"] / max(
                        diagnostic["train_prior_cross_entropy"], 1e-12
                    )
                    diagnostic.update(
                        train_prior_brier=prior_brier, brier_skill_vs_train_prior=1 - brier_ratio
                    )
                    audit = decision_audit(
                        matrix[task_index, position],
                        support_min=support_min,
                        prediction_min=prediction_min,
                    )
                    balanced_matrix = self.balanced.slices[f"market/{market}"]["confusion"]
                    audit["balanced_decision"] = decision_audit(
                        balanced_matrix[task_index, position],
                        support_min=support_min,
                        prediction_min=prediction_min,
                    )
                    audits[market][task][str(horizon)] = audit
                    components.append(
                        [
                            ce_ratio,
                            brier_ratio,
                            1 - row["macro_f1"],
                            audit["balanced_opposite_error"],
                            audit["collapse_deficit"],
                        ]
                    )
        if not components or not np.isfinite(components).all():
            raise ValueError("Ordinal selection requires finite full-market diagnostics")
        mean = np.mean(components, axis=0)
        guard = result["selection_definition"]["pinball_guard_penalty"]
        collapse = cfg.selection_collapse_weight * mean[4]
        opposite = cfg.selection_opposite_weight * mean[3]
        result["selection_loss"] = float(mean[:3].mean() + opposite + collapse + guard)
        result["decision_audit"] = {
            "market": audits,
            "market_macro": {
                "ce_prior_ratio": float(mean[0]),
                "brier_prior_ratio": float(mean[1]),
                "population_macro_f1": float(1 - mean[2]),
                "balanced_opposite_error": float(mean[3]),
                "collapse_deficit": float(mean[4]),
            },
        }
        result["selection_definition"].update(
            direction_basis="equal_market_task_horizon_population_CE_Brier_macro_F1",
            probability_score_weight=1 / 3,
            macro_f1_weight=1 / 3,
            class_support_min=support_min,
            prediction_share_min=prediction_min,
            collapse_weight=cfg.selection_collapse_weight,
            collapse_penalty=float(collapse),
            opposite_weight=cfg.selection_opposite_weight,
            opposite_penalty=float(opposite),
            baseline="fixed_train_market_task_horizon_prior_evaluated_on_validation_labels",
            balanced_accuracy_role="diagnostic_only_not_selection_objective",
        )
        return result
