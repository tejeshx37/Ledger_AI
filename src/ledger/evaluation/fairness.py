"""Phase 7.1: Bias auditing and fairness.

Implements selection rates, demographic parity difference, equalised odds difference,
and false positive rate disparity calculations over sensitive attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_bias(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sensitive_attribute: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Audit predictions for bias across groups defined by sensitive_attribute."""
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    sens_arr = np.array(sensitive_attribute)

    unique_groups = np.unique(sens_arr)
    if len(unique_groups) < 2:
        return {
            "demographic_parity_difference": 0.0,
            "equalized_odds_difference": 0.0,
            "false_positive_rate_disparity": 0.0,
            "group_metrics": {},
        }

    selection_rates = {}
    tprs = {}
    fprs = {}
    group_metrics = {}

    for group in unique_groups:
        mask = sens_arr == group
        g_true = y_true_arr[mask]
        g_pred = y_pred_arr[mask]

        if len(g_true) == 0:
            continue

        # Selection Rate: P(Y_hat = 1)
        sel_rate = float(np.mean(g_pred))
        selection_rates[group] = sel_rate

        # Confusion matrix components
        tp = int(np.sum((g_true == 1) & (g_pred == 1)))
        fn = int(np.sum((g_true == 1) & (g_pred == 0)))
        fp = int(np.sum((g_true == 0) & (g_pred == 1)))
        tn = int(np.sum((g_true == 0) & (g_pred == 0)))

        # True Positive Rate (TPR)
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # False Positive Rate (FPR)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        tprs[group] = tpr
        fprs[group] = fpr

        group_metrics[str(group)] = {
            "selection_rate": sel_rate,
            "tpr": tpr,
            "fpr": fpr,
            "count": len(g_true),
        }

    # Calculate differences across all pairs of groups
    sel_rates_list = list(selection_rates.values())
    tprs_list = list(tprs.values())
    fprs_list = list(fprs.values())

    dp_diff = float(np.max(sel_rates_list) - np.min(sel_rates_list)) if sel_rates_list else 0.0
    tpr_diff = float(np.max(tprs_list) - np.min(tprs_list)) if tprs_list else 0.0
    fpr_diff = float(np.max(fprs_list) - np.min(fprs_list)) if fprs_list else 0.0

    eq_odds_diff = float(max(tpr_diff, fpr_diff))
    fpr_disparity = float(fpr_diff)

    return {
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eq_odds_diff,
        "false_positive_rate_disparity": fpr_disparity,
        "group_metrics": group_metrics,
    }
