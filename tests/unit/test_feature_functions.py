"""Direct unit tests for the individual feature-computation functions,
using small hand-built graphs/datasets so exact values can be asserted.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from ledger.config.models import FeaturesConfig
from ledger.data.canonical import CanonicalDataset, json_text
from ledger.features.graph_features import compute_graph_topological_features
from ledger.features.motif_features import compute_motif_features
from ledger.features.temporal_features import compute_temporal_features


def _star_graph() -> nx.MultiDiGraph:
    """One hub with 3 inbound and 2 outbound edges: fan-in=3, fan-out=2."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_nodes_from(["hub", "a", "b", "c", "d", "e"])
    g.add_edge("a", "hub", tx_id="e1")
    g.add_edge("b", "hub", tx_id="e2")
    g.add_edge("c", "hub", tx_id="e3")
    g.add_edge("hub", "d", tx_id="e4")
    g.add_edge("hub", "e", tx_id="e5")
    return g


def test_motif_fan_in_fan_out_counts() -> None:
    g = _star_graph()
    config = FeaturesConfig(fan_in_threshold=3, fan_out_threshold=2)
    feats = compute_motif_features(g, config).set_index("account_id")
    assert feats.loc["hub", "motif_fan_in_count"] == 3
    assert feats.loc["hub", "motif_fan_out_count"] == 2
    assert feats.loc["hub", "motif_is_fan_in"] == 1
    assert feats.loc["hub", "motif_is_fan_out"] == 1
    assert feats.loc["a", "motif_is_fan_in"] == 0


def test_motif_chain_participation() -> None:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    feats = compute_motif_features(g, FeaturesConfig()).set_index("account_id")
    assert feats.loc["b", "motif_chain_participation"] == 1
    assert feats.loc["a", "motif_chain_participation"] == 0


def test_motif_cycle_participation() -> None:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_node("isolated")
    feats = compute_motif_features(g, FeaturesConfig()).set_index("account_id")
    assert feats.loc["a", "motif_cycle_participation"] == 1
    assert feats.loc["b", "motif_cycle_participation"] == 1
    assert feats.loc["c", "motif_cycle_participation"] == 1
    assert feats.loc["isolated", "motif_cycle_participation"] == 0


def test_motif_features_empty_graph_returns_empty_frame() -> None:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    feats = compute_motif_features(g, FeaturesConfig())
    assert len(feats) == 0
    assert "motif_fan_in_count" in feats.columns


def test_graph_topological_features_on_triangle() -> None:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    feats = compute_graph_topological_features(g).set_index("account_id")
    assert (feats["graph_component_size"] == 3).all()
    assert (feats["graph_core_number"] == 2).all()
    assert abs(feats["graph_pagerank"].sum() - 1.0) < 1e-9


def test_graph_topological_features_empty_graph() -> None:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    feats = compute_graph_topological_features(g)
    assert len(feats) == 0
    assert "graph_pagerank" in feats.columns


def _dataset_with_transactions(rows: list[dict]) -> CanonicalDataset:
    accounts = pd.DataFrame(
        {
            "account_id": ["a1", "a2"],
            "institution_id": [None, None],
            "opened_at": pd.NaT,
            "attributes": [json_text({}), json_text({})],
        }
    )
    transactions = pd.DataFrame(rows)
    labels = pd.DataFrame(columns=["entity_id", "entity_type", "label", "label_source"])
    rings = pd.DataFrame(columns=["ring_id", "member_accounts", "pattern_type", "time_window"])
    return CanonicalDataset(
        name="test", accounts=accounts, transactions=transactions, labels=labels, rings=rings
    )


def test_temporal_features_perfectly_regular_arrivals_are_minimally_bursty() -> None:
    # Equal 1-hour gaps -> std_inter_arrival == 0 -> B == -1 (the Goh &
    # Barabasi burstiness parameter's minimum, by definition of a perfectly
    # periodic process; B == 0 is a Poisson process, not "regular").
    timestamps = pd.date_range("2020-01-01", periods=5, freq="h")
    rows = [
        {"src_account": "a1", "dst_account": "a2", "timestamp": ts, "amount": 10.0}
        for ts in timestamps
    ]
    ds = _dataset_with_transactions(rows)
    feats = compute_temporal_features(ds, FeaturesConfig()).set_index("account_id")
    assert abs(feats.loc["a1", "temporal_burstiness"] - (-1.0)) < 1e-9


def test_temporal_features_no_transactions_returns_empty() -> None:
    ds = _dataset_with_transactions([])
    ds.transactions = pd.DataFrame(columns=["src_account", "dst_account", "timestamp", "amount"])
    feats = compute_temporal_features(ds, FeaturesConfig())
    assert len(feats) == 0
    assert "temporal_burstiness" in feats.columns


def test_temporal_features_single_transaction_has_zero_velocity() -> None:
    rows = [
        {
            "src_account": "a1",
            "dst_account": "a2",
            "timestamp": pd.Timestamp("2020-01-01"),
            "amount": 5.0,
        }
    ]
    ds = _dataset_with_transactions(rows)
    feats = compute_temporal_features(ds, FeaturesConfig()).set_index("account_id")
    assert feats.loc["a1", "temporal_velocity"] == 0.0
    assert feats.loc["a1", "temporal_tx_count"] == 1
