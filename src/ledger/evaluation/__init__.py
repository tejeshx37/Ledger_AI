"""Evaluation harness: positive-class metrics, curves, label mapping, and
the Phase 4.4 model-comparison protocol.

Phases 7+ (bias audit, adversarial robustness, drift, ablations) extend
this package further.
"""

from ledger.evaluation.comparison import ComparisonResult, run_model_comparison
from ledger.evaluation.curves import save_pr_curve
from ledger.evaluation.labels import build_binary_target
from ledger.evaluation.metrics import compute_metrics

__all__ = [
    "compute_metrics",
    "save_pr_curve",
    "build_binary_target",
    "ComparisonResult",
    "run_model_comparison",
]
