"""Graph-topological account features: degree, clustering, PageRank, core
number, and connected-component size.

Computed over the graph structure directly (not fit on train/test), since
these are deterministic functions of observed structure rather than
statistics that could leak train-only information the way a fitted scaler
would; see :mod:`ledger.features.pipeline` for how the two are combined.
"""

from __future__ import annotations

from typing import cast

import networkx as nx
import pandas as pd


def _simple_weighted_digraph(graph: nx.MultiDiGraph[str]) -> nx.DiGraph[str]:
    """Collapse parallel edges into one, weighted by their count.

    Several networkx algorithms (PageRank included) are not implemented
    for multigraphs; this collapse is what lets us run them without
    silently dropping the multi-edge structure information (it survives as
    edge weight).
    """
    simple: nx.DiGraph[str] = nx.DiGraph()
    simple.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1
        else:
            simple.add_edge(u, v, weight=1)
    return simple


def compute_graph_topological_features(graph: nx.MultiDiGraph[str]) -> pd.DataFrame:
    """Return one row per node with degree/clustering/PageRank/core-number/
    component-size columns, prefixed ``graph_``.
    """
    nodes = list(graph.nodes())
    if not nodes:
        return pd.DataFrame(
            columns=[
                "account_id",
                "graph_degree",
                "graph_clustering",
                "graph_pagerank",
                "graph_core_number",
                "graph_component_size",
            ]
        )

    simple_digraph = _simple_weighted_digraph(graph)
    undirected: nx.Graph[str] = nx.Graph(simple_digraph)
    undirected_no_selfloops = undirected.copy()
    undirected_no_selfloops.remove_edges_from(nx.selfloop_edges(undirected_no_selfloops))

    degree: dict[str, int] = dict(graph.degree())
    clustering = cast("dict[str, float]", nx.clustering(undirected))
    pagerank = cast("dict[str, float]", nx.pagerank(simple_digraph, weight="weight"))
    core_number = cast("dict[str, int]", nx.core_number(undirected_no_selfloops))

    component_size: dict[str, int] = {}
    for component in nx.connected_components(undirected):
        size = len(component)
        for node in component:
            component_size[node] = size

    return pd.DataFrame(
        {
            "account_id": nodes,
            "graph_degree": [degree.get(n, 0) for n in nodes],
            "graph_clustering": [clustering.get(n, 0.0) for n in nodes],
            "graph_pagerank": [pagerank.get(n, 0.0) for n in nodes],
            "graph_core_number": [core_number.get(n, 0) for n in nodes],
            "graph_component_size": [component_size.get(n, 1) for n in nodes],
        }
    )


__all__ = ["compute_graph_topological_features"]
