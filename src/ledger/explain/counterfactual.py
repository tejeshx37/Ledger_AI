"""Phase 6.4: Counterfactual explanation search.

Provides functions to determine what feature value changes would prevent an alert
from triggering (dropping its probability below the classification threshold).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from ledger.explain.models import Evidence

logger = structlog.get_logger(__name__)


def explain_counterfactual(
    detector: Any,
    account_id: str,
    evidence: Evidence,
    target_prob: float = 0.5,
) -> str:
    """Find the threshold value for the top feature to drop alert probability below target_prob."""
    if not evidence.feature_attributions:
        return "No feature attributions available to compute counterfactual."

    # Identify the top driving feature
    top_feature = max(evidence.feature_attributions, key=evidence.feature_attributions.get)

    try:
        feat_names = getattr(detector, "_feature_names", [])
        feat_idx = feat_names.index(top_feature)
    except ValueError:
        return f"Top feature '{top_feature}' not found in detector features list."

    is_gnn = hasattr(detector, "graph") and detector.graph is not None

    if not is_gnn:
        return "Counterfactual search is supported only for graph-based GNN detectors."

    if account_id not in detector.graph.nodes:
        return f"Account '{account_id}' not found in detector graph."

    dummy_df = pd.DataFrame(index=[account_id])
    for col in feat_names:
        dummy_df[col] = 0.0

    try:
        baseline_prob = float(detector.predict_proba(dummy_df)[0])
    except Exception:
        baseline_prob = evidence.confidence_score

    if baseline_prob < target_prob:
        return (
            f"The account is not currently flagged (probability {baseline_prob:.4f} < "
            f"{target_prob:.4f}). No counterfactual needed."
        )

    orig_val = float(detector.graph.nodes[account_id]["x"][feat_idx])

    # Search values between orig_val and 0.0
    cf_val = None
    # Test 20 steps along the feature dimension
    for val in np.linspace(orig_val, 0.0, 21):
        detector.graph.nodes[account_id]["x"][feat_idx] = val
        try:
            p = float(detector.predict_proba(dummy_df)[0])
            if p < target_prob:
                cf_val = val
        except Exception:
            pass
        finally:
            # Restore original value
            detector.graph.nodes[account_id]["x"][feat_idx] = orig_val

    if cf_val is not None:
        return (
            f"If the feature '{top_feature}' were reduced from {orig_val:.4f} "
            f"to {cf_val:.4f}, this alert would not have fired."
        )
    else:
        # Fallback to check setting to 0.0 directly
        detector.graph.nodes[account_id]["x"][feat_idx] = 0.0
        try:
            p = float(detector.predict_proba(dummy_df)[0])
            if p < target_prob:
                return (
                    f"If the feature '{top_feature}' were reduced from {orig_val:.4f} "
                    f"to 0.0000, this alert would not have fired."
                )
        except Exception:
            pass
        finally:
            detector.graph.nodes[account_id]["x"][feat_idx] = orig_val

        return (
            f"Reducing the top feature '{top_feature}' from {orig_val:.4f} to 0.0 "
            f"was not sufficient to drop the alert probability below {target_prob:.4f}."
        )
