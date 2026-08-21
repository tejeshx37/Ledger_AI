"""Structural motif-count features: fan-in, fan-out, chain participation,
and cycle participation — the patterns layering actually leaves behind.

Cycle participation is membership in *some* directed cycle (computed via
strongly-connected-component size, which is linear-time), not enumeration
of specific cycles — enumerating all simple cycles is exponential and
intractable on a graph with hundreds of thousands of nodes.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from ledger.config.models import FeaturesConfig

_COLUMNS = [
    "account_id",
    "motif_fan_in_count",
    "motif_fan_out_count",
    "motif_is_fan_in",
    "motif_is_fan_out",
    "motif_chain_participation",
    "motif_cycle_participation",
]


def compute_motif_features(graph: nx.MultiDiGraph[str], config: FeaturesConfig) -> pd.DataFrame:
    """Return one row per node with fan-in/fan-out/chain/cycle motif columns."""
    nodes = list(graph.nodes())
    if not nodes:
        return pd.DataFrame(columns=_COLUMNS)

    simple: nx.DiGraph[str] = nx.DiGraph()
    simple.add_nodes_from(nodes)
    simple.add_edges_from(graph.edges())

    scc_size: dict[str, int] = {}
    for component in nx.strongly_connected_components(simple):
        size = len(component)
        for node in component:
            scc_size[node] = size
    self_loop_nodes = {u for u, v in simple.edges() if u == v}

    rows = []
    for node in nodes:
        in_degree = simple.in_degree(node)
        out_degree = simple.out_degree(node)
        rows.append(
            {
                "account_id": node,
                "motif_fan_in_count": in_degree,
                "motif_fan_out_count": out_degree,
                "motif_is_fan_in": int(in_degree >= config.fan_in_threshold),
                "motif_is_fan_out": int(out_degree >= config.fan_out_threshold),
                "motif_chain_participation": int(in_degree == 1 and out_degree == 1),
                "motif_cycle_participation": int(
                    scc_size.get(node, 1) > 1 or node in self_loop_nodes
                ),
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)


__all__ = ["compute_motif_features"]
