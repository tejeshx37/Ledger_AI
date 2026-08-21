"""Unit tests for the positive-class evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from ledger.config.models import EvaluationConfig, TrainingConfig
from ledger.evaluation.metrics import compute_metrics


def _crafted_case() -> tuple[np.ndarray, np.ndarray]:
    # 10 samples, 3 positive; scores perfectly separate at 0.5.
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    return y_true, y_score


def test_compute_metrics_perfect_separation() -> None:
    y_true, y_score = _crafted_case()
    metrics = compute_metrics(y_true, y_score, EvaluationConfig(), TrainingConfig())
    assert metrics["n_samples"] == 10
    assert metrics["n_positive"] == 3
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["total_alerts"] == 3
    assert metrics["false_alert_fraction"] == 0.0


def test_compute_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(
            np.array([1, 0]), np.array([0.5, 0.5, 0.5]), EvaluationConfig(), TrainingConfig()
        )


def test_compute_metrics_single_class_raises() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_score, EvaluationConfig(), TrainingConfig())


def test_precision_at_k_clips_to_sample_size() -> None:
    y_true, y_score = _crafted_case()
    config = EvaluationConfig(precision_at_k_values=[2, 5, 1000])
    metrics = compute_metrics(y_true, y_score, config, TrainingConfig())
    pak = metrics["precision_at_k"]
    assert pak["2"]["k_used"] == 2
    assert pak["2"]["precision"] == 1.0
    assert pak["1000"]["k_used"] == 10
    assert pak["1000"]["k_requested"] == 1000


def test_recall_at_alert_budget_none_when_budget_not_set() -> None:
    y_true, y_score = _crafted_case()
    metrics = compute_metrics(y_true, y_score, EvaluationConfig(), TrainingConfig())
    assert metrics["recall_at_alert_budget"] is None
    assert metrics["alert_budget"] is None


def test_recall_at_alert_budget_computed_when_set() -> None:
    y_true, y_score = _crafted_case()
    training_config = TrainingConfig(alert_budget_per_period=2)
    metrics = compute_metrics(y_true, y_score, EvaluationConfig(), training_config)
    # top 2 scores are both true positives out of 3 total positives
    assert metrics["recall_at_alert_budget"] == pytest.approx(2 / 3)
    assert metrics["alert_budget"] == 2


def test_false_alert_fraction_none_when_zero_alerts() -> None:
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.1, 0.05, 0.02, 0.01])
    training_config = TrainingConfig(alert_threshold=0.9)
    metrics = compute_metrics(y_true, y_score, EvaluationConfig(), training_config)
    assert metrics["total_alerts"] == 0
    assert metrics["false_alert_fraction"] is None


def test_threshold_sweep_has_configured_number_of_steps() -> None:
    y_true, y_score = _crafted_case()
    config = EvaluationConfig(threshold_sweep_steps=9)
    metrics = compute_metrics(y_true, y_score, config, TrainingConfig())
    assert len(metrics["threshold_sweep"]) == 9
    thresholds = [row["threshold"] for row in metrics["threshold_sweep"]]
    assert thresholds == sorted(thresholds)
    assert all(0.0 < t < 1.0 for t in thresholds)


def test_operating_threshold_reflects_training_config() -> None:
    y_true, y_score = _crafted_case()
    training_config = TrainingConfig(alert_threshold=0.75)
    metrics = compute_metrics(y_true, y_score, EvaluationConfig(), training_config)
    assert metrics["operating_threshold"] == 0.75
    # only score 0.9 and 0.8 clear a 0.75 threshold
    assert metrics["total_alerts"] == 2
