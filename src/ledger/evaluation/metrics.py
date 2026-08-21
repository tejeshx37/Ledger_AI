"""Evaluation harness: metrics on the positive (illicit/fraud) class only.

Overall accuracy is meaningless here — the positive rate is typically
2-10%, so predicting "everything is licit" scores ~90%+ accuracy while
catching nothing. Every metric below is computed against the positive
class specifically, plus the two numbers a compliance team actually cares
about operationally: how many alerts a threshold raises, and what fraction
of those alerts are false.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ledger.config.models import EvaluationConfig, TrainingConfig


def _confusion_counts(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]
) -> tuple[int, int]:
    """Return (n_alerts, n_false_alerts) for a binary prediction vector."""
    n_alerts = int(y_pred.sum())
    n_false_alerts = int(((y_pred == 1) & (y_true == 0)).sum())
    return n_alerts, n_false_alerts


def _precision_at_k(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], k: int
) -> dict[str, Any]:
    n = len(y_true)
    k_used = min(k, n)
    if k_used <= 0:
        return {"k_requested": k, "k_used": 0, "precision": None}
    top_k_idx = np.argsort(-y_score)[:k_used]
    precision = float(y_true[top_k_idx].sum() / k_used)
    return {"k_requested": k, "k_used": k_used, "precision": precision}


def _recall_at_alert_budget(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], budget: int
) -> float | None:
    total_positive = int(y_true.sum())
    if total_positive == 0:
        return None
    n = len(y_true)
    k_used = min(budget, n)
    top_k_idx = np.argsort(-y_score)[:k_used]
    return float(y_true[top_k_idx].sum() / total_positive)


def _threshold_sweep(
    y_true: npt.NDArray[np.int_], y_score: npt.NDArray[np.float64], n_steps: int
) -> list[dict[str, Any]]:
    thresholds = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        n_alerts, n_false_alerts = _confusion_counts(y_true, y_pred)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n_alerts": n_alerts,
                "false_alert_fraction": (n_false_alerts / n_alerts) if n_alerts > 0 else None,
            }
        )
    return rows


def compute_metrics(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    evaluation_config: EvaluationConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """Compute the full positive-class evaluation report for one fold.

    Raises :class:`ValueError` if ``y_true`` has only one class present —
    ROC-AUC and average precision are undefined in that case. Recent
    scikit-learn versions warn and silently return ``NaN`` instead of
    raising; a ``NaN`` quietly written into a metrics.json is not "failing
    loudly", so this is checked explicitly rather than left to sklearn.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true and y_score length mismatch: {len(y_true)} vs {len(y_score)}")
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true has only one class present; ROC-AUC and average precision are "
            "undefined. This fold's split has zero positive or zero negative examples."
        )

    threshold = training_config.alert_threshold
    y_pred = (y_score >= threshold).astype(int)
    n_alerts, n_false_alerts = _confusion_counts(y_true, y_pred)

    precision_at_k = {
        str(k): _precision_at_k(y_true, y_score, k) for k in evaluation_config.precision_at_k_values
    }

    recall_at_budget = None
    if training_config.alert_budget_per_period is not None:
        recall_at_budget = _recall_at_alert_budget(
            y_true, y_score, training_config.alert_budget_per_period
        )

    return {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "operating_threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "total_alerts": n_alerts,
        "false_alert_fraction": (n_false_alerts / n_alerts) if n_alerts > 0 else None,
        "precision_at_k": precision_at_k,
        "alert_budget": training_config.alert_budget_per_period,
        "recall_at_alert_budget": recall_at_budget,
        "threshold_sweep": _threshold_sweep(
            y_true, y_score, evaluation_config.threshold_sweep_steps
        ),
    }


__all__ = ["compute_metrics"]
