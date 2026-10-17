"""Small, dependency-free probability calibration used only by diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def distribution(logits, beta=1.0, bias=(0.0, 0.0, 0.0)):
    if len(logits) != 3 or len(bias) != 3 or not 0.05 <= beta <= 20:
        raise ValueError("Expected three logits and a bounded positive inverse temperature")
    z = [beta * float(x) + float(b) for x, b in zip(logits, bias, strict=True)]
    if not all(math.isfinite(x) for x in z):
        raise ValueError("Nonfinite diagnostic logits")
    top = max(z)
    exp = [math.exp(x - top) for x in z]
    total = sum(exp)
    return [x / total for x in exp], [x - top - math.log(total) for x in z]


def probabilities(logits, beta=1.0, bias=(0.0, 0.0, 0.0)):
    return distribution(logits, beta, bias)[0]


def label(value, lower, upper):
    if not math.isfinite(value):
        raise ValueError("Nonfinite return label")
    return 0 if value < lower else 2 if value > upper else 1


def objective_gradient(points, theta, penalty, with_bias):
    """Mean log loss and its exact gradient; all fitting observations are bounded."""
    if not points:
        raise ValueError("Calibration needs observations")
    beta, *bias = theta
    loss, gradient = 0.0, [0.0] * 4
    for logits, target in points:
        z = [beta * x + b for x, b in zip(logits, bias, strict=True)]
        top = max(z)
        denominator = sum(math.exp(x - top) for x in z)
        loss += top + math.log(denominator) - z[target]
        p = [math.exp(x - top) / denominator for x in z]
        for k in range(3):
            residual = p[k] - int(target == k)
            gradient[0] += residual * logits[k]
            gradient[k + 1] += residual
    n = len(points)
    loss /= n
    gradient = [x / n for x in gradient]
    if with_bias:
        loss += penalty * sum(x * x for x in bias) / 2
        for k in range(3):
            gradient[k + 1] += penalty * bias[k]
    else:
        gradient[1:] = [0.0] * 3
    return loss, gradient


def fit_calibrator(points, *, with_bias=False, penalty=0.1, max_iterations=80):
    """Projected gradient descent with bounded backtracking and zero-sum class bias.

    The inverse temperature parameter makes this a convex objective. Fitting
    never reads the later validation scoring segment or the test split.
    """
    if not math.isfinite(penalty) or penalty <= 0 or not 1 <= max_iterations <= 500:
        raise ValueError("Use a positive fixed shrinkage and 1-500 optimizer iterations")
    theta = [1.0, 0.0, 0.0, 0.0]
    for logits, target in points:
        if (
            len(logits) != 3
            or type(target) is not int
            or not 0 <= target < 3
            or not all(math.isfinite(x) for x in logits)
        ):
            raise ValueError("Calibration observations must be finite three-class logits")
    status = "iteration_limit"
    initial, _ = objective_gradient(points, theta, penalty, with_bias)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        before, gradient = objective_gradient(points, theta, penalty, with_bias)
        step, accepted = 1.0, False
        for _ in range(24):
            candidate = [x - step * g for x, g in zip(theta, gradient, strict=True)]
            candidate[0] = min(20.0, max(0.05, candidate[0]))
            if with_bias:
                center = sum(candidate[1:]) / 3
                candidate[1:] = [x - center for x in candidate[1:]]
            else:
                candidate[1:] = [0.0] * 3
            displacement = sum((a - b) ** 2 for a, b in zip(candidate, theta, strict=True))
            after, _ = objective_gradient(points, candidate, penalty, with_bias)
            if after <= before - displacement / (2 * step) + 1e-14:
                accepted = True
                break
            step /= 2
        if not accepted:
            status = "line_search_limit"
            break
        theta = candidate
        if displacement < 1e-12:
            status = "converged"
            break
    final, _ = objective_gradient(points, theta, penalty, with_bias)
    return {
        "inverse_temperature": theta[0],
        "temperature": 1 / theta[0],
        "bias": theta[1:],
        "bias_l2_penalty": penalty if with_bias else None,
        "fit_observations": len(points),
        "iterations": iterations,
        "status": status,
        "initial_fit_objective": initial,
        "final_fit_objective": final,
    }


@dataclass
class ProbabilityMoments:
    """Streaming multiclass scores, confidence ECE and one-vs-rest reliability."""

    count: int = 0
    ce: float = 0.0
    brier: float = 0.0
    confusion: list = field(default_factory=lambda: [[0] * 3 for _ in range(3)])
    top_bins: list = field(default_factory=lambda: [[0, 0.0, 0] for _ in range(15)])
    class_bins: list = field(
        default_factory=lambda: [[[0, 0.0, 0] for _ in range(15)] for _ in range(3)]
    )

    def add(self, p, y, log_probability=None):
        if len(p) != 3 or not 0 <= y < 3 or any(not math.isfinite(x) or x < 0 for x in p):
            raise ValueError("Invalid probability or class")
        if not math.isclose(sum(p), 1.0, abs_tol=1e-9):
            raise ValueError("Probabilities must sum to one")
        predicted = max(range(3), key=lambda k: p[k])
        self.count += 1
        # The probability floor is recorded; extreme logits may otherwise underflow.
        self.ce -= math.log(max(p[y], 1e-300)) if log_probability is None else log_probability
        self.brier += sum((p[k] - int(y == k)) ** 2 for k in range(3))
        self.confusion[y][predicted] += 1
        bucket = self.top_bins[min(int(p[predicted] * 15), 14)]
        bucket[0] += 1
        bucket[1] += p[predicted]
        bucket[2] += int(predicted == y)
        for k in range(3):
            bucket = self.class_bins[k][min(int(p[k] * 15), 14)]
            bucket[0] += 1
            bucket[1] += p[k]
            bucket[2] += int(y == k)

    def merge(self, other):
        self.count += other.count
        self.ce += other.ce
        self.brier += other.brier
        for k in range(3):
            for j in range(3):
                self.confusion[k][j] += other.confusion[k][j]
        for a, b in zip(self.top_bins, other.top_bins, strict=True):
            for j in range(3):
                a[j] += b[j]
        for a, b in zip(self.class_bins, other.class_bins, strict=True):
            for aa, bb in zip(a, b, strict=True):
                for j in range(3):
                    aa[j] += bb[j]

    def result(self):
        if not self.count:
            return {"observations": 0, "status": "empty"}
        support = [sum(row) for row in self.confusion]
        predicted = [sum(row[k] for row in self.confusion) for k in range(3)]
        recall = [self.confusion[k][k] / support[k] if support[k] else None for k in range(3)]
        precision = [self.confusion[k][k] / predicted[k] if predicted[k] else 0.0 for k in range(3)]
        f1 = [
            2 * self.confusion[k][k] / (support[k] + predicted[k])
            if support[k] + predicted[k]
            else 0.0
            for k in range(3)
        ]

        def ece(bins):
            return (
                sum(abs(correct - confidence) for n, confidence, correct in bins if n) / self.count
            )

        def reliability(bins):
            return [
                {
                    "bin": i,
                    "count": n,
                    "mean_probability": confidence / n,
                    "observed_fraction": correct / n,
                }
                for i, (n, confidence, correct) in enumerate(bins)
                if n
            ]

        def binned_auc(bins):
            positive = sum(row[2] for row in bins)
            negative = self.count - positive
            if not positive or not negative:
                return None
            negatives_below, area = 0, 0.0
            for n, _confidence, yes in bins:
                no = n - yes
                area += yes * (negatives_below + no / 2)
                negatives_below += no
            return area / (positive * negative)

        return {
            "observations": self.count,
            "cross_entropy": self.ce / self.count,
            "brier_score": self.brier / self.count,
            "accuracy": sum(self.confusion[k][k] for k in range(3)) / self.count,
            "balanced_accuracy": sum(x for x in recall if x is not None)
            / sum(x is not None for x in recall),
            "macro_f1": sum(f1) / 3,
            "class_support": support,
            "predicted_class_share": [x / self.count for x in predicted],
            "class_recall": recall,
            "class_precision": precision,
            "confusion_matrix": self.confusion,
            "expected_calibration_error": ece(self.top_bins),
            "classwise_ece": [ece(bins) for bins in self.class_bins],
            "classwise_reliability": [reliability(bins) for bins in self.class_bins],
            "classwise_auc_15bin": [binned_auc(bins) for bins in self.class_bins],
            "auc_method": "15_probability_bins_with_half_credit_for_ties; approximate",
            "probability_floor": 1e-300,
            "calibration_bins": 15,
        }
