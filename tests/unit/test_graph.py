"""Unit tests for temporal graph construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ledger.data.adapters import get_adapter
from ledger.data.graph import TemporalGraphBuilder


@pytest.fixture
def elliptic_graph(elliptic_raw_dir: Path) -> TemporalGraphBuilder:
    ds = get_adapter("elliptic").run(elliptic_raw_dir)
    return TemporalGraphBuilder(ds)


def test_build_static_graph_full(elliptic_graph: TemporalGraphBuilder) -> None:
    g = elliptic_graph.build_static_graph()
    assert g.number_of_nodes() == 40
    assert g.number_of_edges() == 80


def test_build_static_graph_window_is_subset(elliptic_graph: TemporalGraphBuilder) -> None:
    full = elliptic_graph.build_static_graph()
    windowed = elliptic_graph.build_static_graph(
        (pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-05"))
    )
    assert windowed.number_of_nodes() == full.number_of_nodes()
    assert windowed.number_of_edges() <= full.number_of_edges()
    for _, _, data in windowed.edges(data=True):
        assert pd.Timestamp("1970-01-01") <= data["timestamp"] < pd.Timestamp("1970-01-05")


def test_build_temporal_snapshots_tiles_full_range(elliptic_graph: TemporalGraphBuilder) -> None:
    snapshots = elliptic_graph.build_temporal_snapshots(pd.Timedelta(days=2))
    assert len(snapshots) > 0
    total_edges = sum(s.number_of_edges() for s in snapshots)
    full = elliptic_graph.build_static_graph()
    assert total_edges == full.number_of_edges()


def test_build_temporal_snapshots_rejects_non_positive_step(
    elliptic_graph: TemporalGraphBuilder,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        elliptic_graph.build_temporal_snapshots(pd.Timedelta(0))


def test_node_statistics_shape_and_columns(elliptic_graph: TemporalGraphBuilder) -> None:
    stats = elliptic_graph.node_statistics()
    assert len(stats) == 40
    expected_cols = {
        "account_id",
        "in_degree",
        "out_degree",
        "total_degree",
        "in_strength",
        "out_strength",
        "n_distinct_in_neighbors",
        "n_distinct_out_neighbors",
    }
    assert expected_cols == set(stats.columns)
    assert (stats["total_degree"] == stats["in_degree"] + stats["out_degree"]).all()


def test_build_pyg_data_shapes(elliptic_graph: TemporalGraphBuilder) -> None:
    torch_geometric = pytest.importorskip("torch_geometric")
    del torch_geometric
    data = elliptic_graph.build_pyg_data()
    assert data.num_nodes == 40
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 80
    assert data.x.shape[0] == 40
