"""Unit tests for the shared graph-detector machinery: the attributed-graph
builder, PyG subgraph construction, fan-out resolution, and the hand-rolled
neighbour sampler.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ledger.config.models import ModelConfig
from ledger.models.graph_common import (
    _build_in_edge_index,
    attach_node_features,
    build_pyg_subgraph,
    effective_fanout,
    sample_khop_edges,
)


def _tiny_graph() -> nx.MultiDiGraph[str]:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    g.add_edge("a", "b", timestamp=pd.Timestamp("2020-01-01"))
    g.add_edge("b", "c", timestamp=pd.Timestamp("2020-01-02"))
    g.add_edge("a", "c", timestamp=pd.Timestamp("2020-01-03"))
    return g


def test_attach_node_features_sets_x_attribute() -> None:
    g = _tiny_graph()
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    attributed = attach_node_features(g, features)
    assert np.allclose(attributed.nodes["a"]["x"], [1.0])
    assert np.allclose(attributed.nodes["b"]["x"], [2.0])
    # original graph is untouched
    assert "x" not in g.nodes["a"]


def test_attach_node_features_missing_account_raises() -> None:
    g = _tiny_graph()
    features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["a", "b"])
    with pytest.raises(KeyError, match="c"):
        attach_node_features(g, features)


def test_effective_fanout_pads_short_list() -> None:
    config = ModelConfig(num_layers=3, fanout_per_layer=[10])
    assert effective_fanout(config) == [10, 10, 10]


def test_effective_fanout_truncates_long_list() -> None:
    config = ModelConfig(num_layers=1, fanout_per_layer=[10, 5, 2])
    assert effective_fanout(config) == [10]


def test_effective_fanout_empty_defaults_to_ten() -> None:
    config = ModelConfig(num_layers=2, fanout_per_layer=[])
    assert effective_fanout(config) == [10, 10]


def test_build_pyg_subgraph_shapes() -> None:
    torch = pytest.importorskip("torch")
    g = _tiny_graph()
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    attributed = attach_node_features(g, features)
    data, node_order = build_pyg_subgraph(attributed, ["a", "b", "c"])
    assert node_order == ["a", "b", "c"]
    assert data.x.shape == (3, 1)
    assert data.edge_index.shape == (2, 3)
    assert data.edge_time.shape == (3,)
    assert isinstance(data.edge_index, torch.Tensor)


def test_build_pyg_subgraph_drops_edges_outside_node_set() -> None:
    g = _tiny_graph()
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    attributed = attach_node_features(g, features)
    data, node_order = build_pyg_subgraph(attributed, ["a", "b"])
    assert node_order == ["a", "b"]
    # only the a->b edge survives; a->c and b->c are dropped (c excluded)
    assert data.edge_index.shape == (2, 1)


def test_build_pyg_subgraph_empty_edges_has_correct_shape() -> None:
    g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    g.add_node("a")
    g.add_node("b")
    features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["a", "b"])
    attributed = attach_node_features(g, features)
    data, _node_order = build_pyg_subgraph(attributed, ["a", "b"])
    assert data.edge_index.shape == (2, 0)
    assert data.edge_time.shape == (0,)


def test_build_in_edge_index_maps_destinations() -> None:
    # edges: 0->1, 0->2, 1->2
    in_edges = _build_in_edge_index(edge_dst=[1, 2, 2], num_nodes=3)
    assert in_edges[0] == []
    assert in_edges[1] == [0]
    assert set(in_edges[2]) == {1, 2}


def test_sample_khop_edges_respects_fanout_limit() -> None:
    # node 5 has 4 in-edges (from 0,1,2,3); fanout=2 must keep exactly 2.
    edge_dst = [5, 5, 5, 5]
    edge_src = [0, 1, 2, 3]
    in_edges = _build_in_edge_index(edge_dst, num_nodes=6)
    rng = np.random.default_rng(0)
    nodes, edges = sample_khop_edges(in_edges, edge_src, [5], fanout=[2], rng=rng)
    assert 5 in nodes
    assert len(nodes) == 3  # seed + 2 sampled predecessors
    assert len(edges) == 2


def test_sample_khop_edges_multi_hop_expands_frontier() -> None:
    # chain: 0 -> 1 -> 2 -> 3 (edge_dst[i] = i+1 receives from edge_src[i] = i)
    edge_src = [0, 1, 2]
    edge_dst = [1, 2, 3]
    in_edges = _build_in_edge_index(edge_dst, num_nodes=4)
    rng = np.random.default_rng(0)
    nodes, edges = sample_khop_edges(in_edges, edge_src, [3], fanout=[5, 5], rng=rng)
    assert set(nodes) == {3, 2, 1}
    assert len(edges) == 2


def test_sample_khop_edges_preserves_parallel_edges() -> None:
    # two distinct edges from 0 -> 1 (a transaction multigraph feature)
    edge_src = [0, 0]
    edge_dst = [1, 1]
    in_edges = _build_in_edge_index(edge_dst, num_nodes=2)
    rng = np.random.default_rng(0)
    nodes, edges = sample_khop_edges(in_edges, edge_src, [1], fanout=[5], rng=rng)
    assert len(edges) == 2
    assert set(edges) == {0, 1}
