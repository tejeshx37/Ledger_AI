"""Temporal graph construction over a canonical dataset.

Nodes are accounts; edges are transactions carrying ``amount``,
``timestamp``, and ``channel``. Everything here operates on
:class:`~ledger.data.canonical.CanonicalDataset` — no dataset-specific
logic — so it works identically across every adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from ledger.data.canonical import CanonicalDataset

if TYPE_CHECKING:
    import torch_geometric.data


class TemporalGraphBuilder:
    """Builds static and temporal views of a canonical dataset's transaction graph."""

    def __init__(self, dataset: CanonicalDataset) -> None:
        self.dataset = dataset
        self._full_graph = self._build_graph(dataset.transactions)

    def _build_graph(self, transactions: pd.DataFrame) -> nx.MultiDiGraph[str]:
        graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        for row in self.dataset.accounts.itertuples(index=False):
            graph.add_node(str(row.account_id), institution_id=row.institution_id)
        for row in transactions.itertuples(index=False):
            graph.add_edge(
                str(row.src_account),
                str(row.dst_account),
                tx_id=row.tx_id,
                amount=row.amount,
                timestamp=row.timestamp,
                channel=row.channel,
            )
        return graph

    def build_static_graph(
        self, window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    ) -> nx.MultiDiGraph[str]:
        """Collapse a time window ``[start, end)`` into one static graph.

        ``window=None`` returns the full graph over every transaction,
        including those with an unknown (null) timestamp.
        """
        if window is None:
            return self._full_graph.copy()
        start, end = window
        transactions = self.dataset.transactions
        mask = (
            transactions["timestamp"].notna()
            & (transactions["timestamp"] >= start)
            & (transactions["timestamp"] < end)
        )
        return self._build_graph(transactions[mask])

    def build_temporal_snapshots(self, step: pd.Timedelta) -> list[nx.MultiDiGraph[str]]:
        """Return one static graph per fixed-width window of size ``step``,
        tiling the dataset's full observed timestamp range.

        Transactions with a null timestamp are excluded from every
        snapshot (they cannot be placed in time) but are still present in
        :meth:`build_static_graph` with ``window=None``.
        """
        if step <= pd.Timedelta(0):
            raise ValueError(f"step must be a positive Timedelta, got {step}")
        timestamps = self.dataset.transactions["timestamp"].dropna()
        if timestamps.empty:
            return []
        start, end = timestamps.min(), timestamps.max()
        boundaries = [start]
        while boundaries[-1] <= end:
            boundaries.append(boundaries[-1] + step)
        return [
            self.build_static_graph((boundaries[i], boundaries[i + 1]))
            for i in range(len(boundaries) - 1)
        ]

    def node_statistics(self, graph: nx.MultiDiGraph[str] | None = None) -> pd.DataFrame:
        """Degree, in/out strength, and neighbourhood-size statistics per node."""
        g = graph if graph is not None else self._full_graph
        rows: list[dict[str, Any]] = []
        for node in g.nodes():
            in_edges = list(g.in_edges(node, data=True))
            out_edges = list(g.out_edges(node, data=True))
            in_degree = len(in_edges)
            out_degree = len(out_edges)
            in_strength = sum(float(d["amount"]) for *_, d in in_edges if pd.notna(d.get("amount")))
            out_strength = sum(
                float(d["amount"]) for *_, d in out_edges if pd.notna(d.get("amount"))
            )
            rows.append(
                {
                    "account_id": node,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "total_degree": in_degree + out_degree,
                    "in_strength": in_strength,
                    "out_strength": out_strength,
                    "n_distinct_in_neighbors": len({u for u, _, _ in in_edges}),
                    "n_distinct_out_neighbors": len({v for _, v, _ in out_edges}),
                }
            )
        return pd.DataFrame(rows)

    def build_pyg_data(
        self, node_features: pd.DataFrame | None = None
    ) -> torch_geometric.data.Data:
        """Build a PyTorch Geometric ``Data`` object from the full graph.

        Requires the optional ``ml`` extra (``torch``, ``torch-geometric``);
        raises :class:`ImportError` with an actionable message if it is not
        installed, rather than failing on an unrelated ``NameError`` deep
        inside torch_geometric.

        ``node_features`` must be indexed by ``account_id`` if given;
        defaults to :meth:`node_statistics` (degree/strength features) when
        omitted.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "build_pyg_data requires the optional 'ml' extra. Install with "
                "`pip install -e '.[ml]'`."
            ) from exc

        nodes = list(self._full_graph.nodes())
        node_index = {n: i for i, n in enumerate(nodes)}
        transactions = self.dataset.transactions

        src = transactions["src_account"].map(node_index).to_numpy()
        dst = transactions["dst_account"].map(node_index).to_numpy()
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        amount = transactions["amount"].fillna(0.0).to_numpy(dtype="float32")
        edge_attr = torch.tensor(amount, dtype=torch.float).unsqueeze(-1)

        features = node_features if node_features is not None else self.node_statistics()
        feature_cols = [c for c in features.columns if c != "account_id"]
        aligned = features.set_index("account_id").reindex(nodes)[feature_cols].fillna(0.0)
        x = torch.tensor(aligned.to_numpy(dtype="float32"))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


__all__ = ["TemporalGraphBuilder"]
