"""Evaluation harness: positive-class metrics, curves, and label mapping.

Phases 7+ (bias audit, adversarial robustness, drift, ablations) extend
this package; Phase 3 ships the core metrics/curve/label machinery every
later phase reuses.
"""

from ledger.evaluation.curves import save_pr_curve
from ledger.evaluation.labels import build_binary_target
from ledger.evaluation.metrics import compute_metrics

__all__ = ["compute_metrics", "save_pr_curve", "build_binary_target"]
