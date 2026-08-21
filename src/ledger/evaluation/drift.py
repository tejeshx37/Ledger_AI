"""Phase 7.3: Performance and input distribution drift metrics.

Provides tools to analyze model performance degradation over time bins and
calculate feature drift (Population Stability Index - PSI).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_performance_drift(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    timestamps: pd.Series | np.ndarray,
    num_bins: int = 4,
) -> list[dict[str, Any]]:
    """Measure model ROC-AUC performance across sequential time bins."""
    df = pd.DataFrame(
        {
            "y_true": np.array(y_true, dtype=int),
            "y_prob": np.array(y_prob, dtype=float),
            "timestamp": pd.to_datetime(timestamps),
        }
    )

    if len(df) == 0:
        return []

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Split into equal-sized bins based on chronological order
    df["bin"] = pd.qcut(df["timestamp"], q=num_bins, labels=False, duplicates="drop")

    drift_report = []
    for b in sorted(df["bin"].unique()):
        sub = df[df["bin"] == b]
        min_t = sub["timestamp"].min().isoformat()
        max_t = sub["timestamp"].max().isoformat()

        if len(sub["y_true"].unique()) < 2:
            auc = 0.5  # Default if only one class exists in bin
        else:
            auc = float(roc_auc_score(sub["y_true"], sub["y_prob"]))

        drift_report.append(
            {
                "bin_index": int(b),
                "start_time": min_t,
                "end_time": max_t,
                "count": len(sub),
                "roc_auc": auc,
            }
        )

    return drift_report


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions."""
    # Filter NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin boundaries based on Expected quantiles
    quantiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bins = np.percentile(expected, quantiles)
        bins = np.unique(bins)  # Drop duplicate bin edges
        if len(bins) < 2:
            # Constant feature
            return 0.0
    except Exception:
        return 0.0

    # Digitize into bins
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    # Convert to percentages with epsilon to prevent log(0)
    eps = 1e-4
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    # Normalize to 1.0
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # PSI Formula
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def calculate_input_drift(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> dict[str, Any]:
    """Calculate average and per-feature Population Stability Index (PSI)."""
    feature_psis = {}
    common_cols = [c for c in train_features.columns if c in test_features.columns]

    for col in common_cols:
        # Ignore non-numeric columns
        if not np.issubdtype(train_features[col].dtype, np.number):
            continue
        try:
            psi_val = calculate_psi(
                train_features[col].to_numpy(), test_features[col].to_numpy()
            )
            feature_psis[col] = psi_val
        except Exception:
            feature_psis[col] = 0.0

    avg_psi = float(np.mean(list(feature_psis.values()))) if feature_psis else 0.0

    return {
        "average_psi": avg_psi,
        "feature_psis": feature_psis,
        "drift_level": (
            "significant"
            if avg_psi >= 0.25
            else "moderate"
            if avg_psi >= 0.1
            else "insignificant"
        ),
    }
