"""Unit tests for Phase 6 (Explanation Layer) components.

Covers Evidence schema, deterministic narrative generation, LLM factual validation
rules, counterfactual search, and local subgraph ablation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from ledger.explain.counterfactual import explain_counterfactual
from ledger.explain.models import Evidence
from ledger.explain.narrative import generate_deterministic_narrative, validate_narrative
from ledger.explain.subgraph import extract_alert_evidence


class DummyModel:
    """Mock detector for GNN explainability testing."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph
        self._is_fitted = True
        self._feature_names = ["feat_1", "feat_2"]

    def predict_proba(self, X: Any) -> np.ndarray:
        # Simple prediction function: sum of node feature values for index
        node_id = X.index[0]
        feats = self.graph.nodes[node_id]["x"]
        # If feat_1 is high, probability is high
        score = float(1.0 / (1.0 + np.exp(-feats[0])))
        return np.array([score])


def test_evidence_serialization() -> None:
    ev = Evidence(
        alert_id="alert_123",
        account_id="acc_1",
        contributing_accounts=["acc_1", "acc_2"],
        contributing_transactions=["tx_1"],
        motif_type_detected="cycle",
        time_window=("2022-09-01T00:00:00", "2022-09-01T01:00:00"),
        feature_attributions={"feat_1": 0.8, "feat_2": 0.2},
        confidence_score=0.92,
    )

    d = ev.to_dict()
    assert d["alert_id"] == "alert_123"
    assert d["feature_attributions"]["feat_1"] == 0.8
    assert "cycle" in ev.to_json()


def test_deterministic_narrative() -> None:
    ev = Evidence(
        alert_id="alert_123",
        account_id="acc_1",
        contributing_accounts=["acc_1", "acc_2"],
        contributing_transactions=["tx_1"],
        motif_type_detected="cycle",
        time_window=("2022-09-01T00:00:00", "2022-09-01T01:00:00"),
        feature_attributions={"feat_1": 0.8, "feat_2": 0.2},
        confidence_score=0.92,
    )

    narrative = generate_deterministic_narrative(ev)
    assert "alert_123" in narrative
    assert "acc_1" in narrative
    assert "acc_2" in narrative
    assert "tx_1" in narrative
    assert "cycle" in narrative
    assert "feat_1 (80.00%)" in narrative


def test_narrative_validation() -> None:
    ev = Evidence(
        alert_id="alert_123",
        account_id="acc_1",
        contributing_accounts=["acc_1", "acc_2"],
        contributing_transactions=["tx_1"],
        motif_type_detected="cycle",
        time_window=("2022-09-01T00:00:00", "2022-09-01T01:00:00"),
        feature_attributions={"feat_1": 0.8},
        confidence_score=0.92,
    )

    # Valid narrative
    valid_text = (
        "Alert alert_123 flags account acc_1. Transactions like tx_1 involving acc_2 "
        "form a cycle. Observed on 2022-09-01."
    )
    assert validate_narrative(valid_text, ev) is True

    # Invalid narrative: contains unauthorized account (hallucination)
    invalid_text_acc = (
        "Alert alert_123 flags account acc_1. Transactions like tx_1 involving B5 "
        "form a cycle. Observed on 2022-09-01."
    )
    assert validate_narrative(invalid_text_acc, ev) is False

    # Invalid narrative: contains unauthorized date
    invalid_text_date = (
        "Alert alert_123 flags account acc_1. Transactions like tx_1 involving acc_2 "
        "form a cycle. Observed on 2022-09-15."
    )
    assert validate_narrative(invalid_text_date, ev) is False


def test_subgraph_evidence_extraction() -> None:
    # Build tiny networkx graph
    g = nx.MultiDiGraph()
    # Add nodes with feature arrays
    g.add_node("acc_1", x=np.array([2.0, 0.5]))
    g.add_node("acc_2", x=np.array([0.1, 0.1]))
    g.add_edge("acc_1", "acc_2", tx_id="tx_1", timestamp="2022-09-01T00:20:00")

    model = DummyModel(g)

    # Extract evidence
    ev = extract_alert_evidence(model, "acc_1", alert_id="alert_test")

    assert ev.alert_id == "alert_test"
    assert ev.account_id == "acc_1"
    # Removing edge tx_1 cuts the message path, which doesn't affect baseline dummy prediction here,
    # but the features should show feat_1 as highly attributed
    assert "feat_1" in ev.feature_attributions
    assert ev.confidence_score > 0.5


def test_counterfactual_explanation() -> None:
    g = nx.MultiDiGraph()
    # High value for feat_1 makes prob high
    g.add_node("acc_1", x=np.array([5.0, 1.0]))
    model = DummyModel(g)

    ev = Evidence(
        alert_id="alert_test",
        account_id="acc_1",
        contributing_accounts=["acc_1"],
        contributing_transactions=[],
        motif_type_detected=None,
        time_window=None,
        feature_attributions={"feat_1": 1.0},
        confidence_score=0.99,
    )

    # Original prob is high: sigmoid(5) = 0.993
    # Target prob is 0.5 (where feat_1 = 0, sigmoid(0) = 0.5)
    explanation = explain_counterfactual(model, "acc_1", ev, target_prob=0.6)

    assert "feat_1" in explanation
    assert "reduced" in explanation
