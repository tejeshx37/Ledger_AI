"""Phase 6: Explanation layer package.

Exposes APIs to extract evidence, generate narratives, and perform counterfactual
search for alerts.
"""

from __future__ import annotations

from ledger.explain.counterfactual import explain_counterfactual
from ledger.explain.models import Evidence
from ledger.explain.narrative import generate_deterministic_narrative, get_narrative
from ledger.explain.subgraph import extract_alert_evidence

__all__ = [
    "Evidence",
    "extract_alert_evidence",
    "generate_deterministic_narrative",
    "get_narrative",
    "explain_counterfactual",
]
