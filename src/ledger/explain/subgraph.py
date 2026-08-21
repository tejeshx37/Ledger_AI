"""Phase 6.1: Subgraph and evidence extraction.

Provides functions to identify the minimal subgraph (accounts and transactions)
and feature attributions driving a specific alert.
"""

from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import structlog

from ledger.explain.models import Evidence

logger = structlog.get_logger(__name__)


def extract_alert_evidence(
    detector: Any,
    account_id: str,
    alert_id: str | None = None,
    threshold: float = 0.5,
) -> Evidence:
    """Extract evidence for a given alert account.

    Uses ablation-based attribution over the model's k-hop neighborhood
    and features to locate the exact drivers of the prediction.
    """
    if alert_id is None:
        alert_id = f"alert_{account_id}_{int(hashlib.md5(account_id.encode()).hexdigest()[:8], 16)}"

    # 1. Compute baseline probability
    dummy_df = pd.DataFrame(index=[account_id])
    for col in getattr(detector, "_feature_names", []):
        dummy_df[col] = 0.0  # Placeholder values, GNN gets features from graph

    try:
        baseline_prob = float(detector.predict_proba(dummy_df)[0])
    except Exception as exc:
        logger.error("Failed to compute baseline prediction probability", exc=str(exc))
        baseline_prob = 0.0

    contributing_accounts = {account_id}
    contributing_transactions = []
    time_window = None
    feature_attributions = {}
    motif_type_detected = None

    # Check if this is a GNN detector
    is_gnn = hasattr(detector, "graph") and detector.graph is not None

    if is_gnn:
        original_graph = detector.graph.copy()
        num_layers = getattr(getattr(detector, "config", None), "num_layers", 2)

        # Retrieve k-hop neighborhood of the account in the graph
        try:
            undirected = detector.graph.to_undirected()
            k_hop_nodes = list(
                nx.single_source_shortest_path_length(
                    undirected, account_id, cutoff=num_layers
                ).keys()
            )
        except Exception:
            k_hop_nodes = [account_id]

        # A. Edge ablation to identify contributing transactions
        # Collect edges within the k-hop neighborhood
        edges_to_ablate = []
        for u, v, key, data in detector.graph.subgraph(k_hop_nodes).edges(keys=True, data=True):
            edges_to_ablate.append((u, v, key, data))

        for u, v, key, data in edges_to_ablate:
            # Temporarily remove edge
            if detector.graph.has_edge(u, v, key):
                detector.graph.remove_edge(u, v, key)
                try:
                    ablated_prob = float(detector.predict_proba(dummy_df)[0])
                    drop = baseline_prob - ablated_prob
                    if drop > 0.001:
                        tx_id = data.get("tx_id")
                        if tx_id:
                            contributing_transactions.append(str(tx_id))
                        contributing_accounts.add(str(u))
                        contributing_accounts.add(str(v))
                except Exception:
                    pass
                finally:
                    # Restore edge
                    detector.graph.add_edge(u, v, key, **original_graph[u][v][key])

        # B. Feature ablation to identify key node features
        if account_id in detector.graph.nodes:
            feat_names = getattr(detector, "_feature_names", [])
            original_features = detector.graph.nodes[account_id]["x"].copy()
            for idx, feat_name in enumerate(feat_names):
                # Set feature value to 0
                detector.graph.nodes[account_id]["x"][idx] = 0.0
                try:
                    ablated_prob = float(detector.predict_proba(dummy_df)[0])
                    drop = baseline_prob - ablated_prob
                    if drop > 0.001:
                        feature_attributions[feat_name] = float(drop)
                except Exception:
                    pass
                finally:
                    # Restore feature value
                    detector.graph.nodes[account_id]["x"][idx] = original_features[idx]

        # Retrieve time window of contributing transactions if available
        timestamps = []
        for u, v, data in detector.graph.edges(data=True):
            tx_id = data.get("tx_id")
            if tx_id in contributing_transactions and "timestamp" in data:
                timestamps.append(data["timestamp"])
            elif (u == account_id or v == account_id) and "timestamp" in data:
                timestamps.append(data["timestamp"])

        if timestamps:
            try:
                min_t = pd.to_datetime(min(timestamps)).isoformat()
                max_t = pd.to_datetime(max(timestamps)).isoformat()
                time_window = (min_t, max_t)
            except Exception:
                pass

        # Identify motifs if any
        # If any contributing accounts form a cycle, mark cycle motif
        try:
            sub = detector.graph.subgraph(contributing_accounts)
            cycles = list(nx.simple_cycles(sub))
            if cycles:
                motif_type_detected = "cycle"
        except Exception:
            pass

    else:
        # Non-GNN fallback (tabular feature ablation)
        feat_names = getattr(detector, "_feature_names", [])
        original_x = dummy_df.loc[[account_id]].copy()
        for feat_name in feat_names:
            # Set feature to 0
            test_df = original_x.copy()
            test_df.loc[account_id, feat_name] = 0.0
            try:
                ablated_prob = float(detector.predict_proba(test_df)[0])
                drop = baseline_prob - ablated_prob
                if drop > 0.001:
                    feature_attributions[feat_name] = float(drop)
            except Exception:
                pass

    # Normalize feature attributions so they are relative importances
    total_attr = sum(feature_attributions.values())
    if total_attr > 0:
        feature_attributions = {k: v / total_attr for k, v in feature_attributions.items()}

    # Ensure contributing_accounts is sorted list
    sorted_accounts = sorted(list(contributing_accounts))
    sorted_txs = sorted(contributing_transactions)

    return Evidence(
        alert_id=alert_id,
        account_id=account_id,
        contributing_accounts=sorted_accounts,
        contributing_transactions=sorted_txs,
        motif_type_detected=motif_type_detected,
        time_window=time_window,
        feature_attributions=feature_attributions,
        confidence_score=baseline_prob,
    )
