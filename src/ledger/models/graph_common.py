"""Shared machinery for graph-based detectors (GraphSAGE, TGAT — Phase 4).

Both models message-pass over the SAME attributed graph: every account is a
node carrying its feature vector as node attribute ``"x"``; every
transaction is an edge carrying its timestamp. The graph is supplied at
construction time, not through ``fit``/``predict_proba``'s ``X`` — the
Detector ABC's ``fit(X, y)`` / ``predict_proba(X)`` contract (Phase 3)
stays exactly the row-aligned tabular contract XGBoost uses; graph context
is injected once, when the detector is built (see
:mod:`ledger.models.registry`).

``fit`` trains only on the subgraph induced by the labelled (training)
node set — edges reaching into val/test nodes are dropped before a single
gradient is computed. ``predict_proba`` then message-passes over the FULL
graph (every edge, including ones into nodes the optimiser never touched).
That asymmetry is what makes the model genuinely inductive: a held-out
account is scored using its real neighbourhood even though the trained
weights never backpropagated through it — the deployment condition the
project brief calls out explicitly.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

from ledger.config.models import FeaturesConfig, ModelConfig, TrainingConfig
from ledger.features.motif_features import compute_motif_features
from ledger.models.base import Detector

MOTIF_COLUMNS = [
    "motif_is_fan_in",
    "motif_is_fan_out",
    "motif_chain_participation",
    "motif_cycle_participation",
]


class GraphDetectorNotFittedError(RuntimeError):
    """Raised when predict_proba/feature_importance is called before fit()."""


def attach_node_features(
    graph: nx.MultiDiGraph[str], features: pd.DataFrame
) -> nx.MultiDiGraph[str]:
    """Return a copy of ``graph`` with each node's feature vector attached
    as node attribute ``"x"`` (a 1-D float32 array), sourced from
    ``features`` (indexed by account_id, as produced by
    :class:`ledger.features.pipeline.FeaturePipeline`).

    Every node in ``graph`` must have a row in ``features``; raises
    :class:`KeyError` rather than silently zero-filling an unmapped node.
    """
    graph = graph.copy()
    feature_cols = list(features.columns)
    for node in graph.nodes():
        if node not in features.index:
            raise KeyError(f"account_id {node!r} has no row in the feature matrix")
        graph.nodes[node]["x"] = features.loc[node, feature_cols].to_numpy(dtype="float32")
    return graph


def effective_fanout(config: ModelConfig) -> list[int]:
    """Resolve ``fanout_per_layer`` to exactly ``num_layers`` entries,
    repeating the last configured value if the list is shorter than that.
    """
    fanout = list(config.fanout_per_layer) or [10]
    if len(fanout) >= config.num_layers:
        return fanout[: config.num_layers]
    return fanout + [fanout[-1]] * (config.num_layers - len(fanout))


def build_pyg_subgraph(graph: nx.MultiDiGraph[str], node_ids: list[str]) -> tuple[Any, list[str]]:
    """Build a PyG ``Data`` object for the subgraph induced by ``node_ids``
    (an edge is kept only if both endpoints are in ``node_ids``).

    Returns ``(data, node_order)`` where ``node_order[i]`` is the
    account_id of row ``i`` of ``data.x`` — every downstream tensor
    (including model output) is indexed against this order. ``data`` also
    carries ``edge_time`` (Unix-epoch seconds per edge, 0.0 for an unknown
    timestamp) for time-aware encoders such as TGAT.

    Requires the optional ``ml`` extra (``torch``, ``torch-geometric``).
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "Graph detectors require the optional 'ml' extra. Install with "
            "`pip install -e '.[ml]'`."
        ) from exc

    node_order = list(node_ids)
    index = {n: i for i, n in enumerate(node_order)}
    x = torch.tensor(np.stack([graph.nodes[n]["x"] for n in node_order]), dtype=torch.float)

    node_set = set(node_order)
    src: list[int] = []
    dst: list[int] = []
    ts: list[float] = []
    for u, v, edge_data in graph.edges(data=True):
        if u in node_set and v in node_set:
            src.append(index[u])
            dst.append(index[v])
            timestamp = edge_data.get("timestamp")
            ts.append(float(timestamp.timestamp()) if pd.notna(timestamp) else 0.0)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_time = torch.tensor(ts, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_time = torch.zeros((0,), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.edge_time = edge_time
    return data, node_order


def _build_in_edge_index(edge_dst: list[int], num_nodes: int) -> list[list[int]]:
    """``in_edges[v]`` = indices of every edge whose destination is ``v``."""
    in_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    for e, v in enumerate(edge_dst):
        in_edges[v].append(e)
    return in_edges


def sample_khop_edges(
    in_edges: list[list[int]],
    edge_src: list[int],
    seed_indices: list[int],
    fanout: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Sample a ``len(fanout)``-hop neighbourhood around ``seed_indices``,
    keeping up to ``fanout[hop]`` in-edges per node at each hop.

    A hand-rolled replacement for PyG's ``NeighborLoader``: that loader's
    sampling backend requires the compiled ``pyg-lib``/``torch-sparse``
    packages, neither of which has a prebuilt wheel for every supported
    torch build, and building either from source is not something a
    training run should depend on. This samples *edge indices* rather than
    deduplicated ``(u, v)`` node pairs specifically so parallel edges
    between the same two accounts (a real feature of a transaction
    multigraph — two accounts can transact more than once) are preserved
    individually, each keeping its own timestamp.

    Returns ``(node_ids, edge_ids)``: every node index touched (seeds
    first, in seed order) and the de-duplicated edge indices sampled.
    """
    all_nodes = list(seed_indices)
    all_nodes_set = set(seed_indices)
    sampled_edges: set[int] = set()
    frontier = list(seed_indices)
    for layer_fanout in fanout:
        next_frontier: list[int] = []
        for v in frontier:
            candidates = in_edges[v]
            if not candidates:
                continue
            if len(candidates) > layer_fanout:
                chosen_pos = rng.choice(len(candidates), size=layer_fanout, replace=False)
                chosen = [candidates[i] for i in chosen_pos]
            else:
                chosen = candidates
            for e in chosen:
                sampled_edges.add(e)
                u = edge_src[e]
                if u not in all_nodes_set:
                    all_nodes_set.add(u)
                    all_nodes.append(u)
                next_frontier.append(u)
        frontier = next_frontier
    return all_nodes, list(sampled_edges)


@dataclass
class GraphDetector(Detector):
    """Shared fit/predict/save/load/feature_importance/metadata for
    GNN-based detectors. Subclasses provide the encoder (message-passing
    layers) via :meth:`_build_encoder` and a name via :attr:`architecture_name`.
    """

    config: ModelConfig
    training_config: TrainingConfig
    features_config: FeaturesConfig
    graph: nx.MultiDiGraph[str]
    _encoder: Any = field(default=None, repr=False)
    _classifier_head: Any = field(default=None, repr=False)
    _motif_head: Any = field(default=None, repr=False)
    _feature_names: list[str] | None = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    @abstractmethod
    def _build_encoder(self, in_dim: int) -> Any:
        """Return an ``nn.Module`` with ``forward(x, edge_index, edge_time) ->
        Tensor[N, hidden_dim]`` and ``first_layer_input_weight() -> Tensor``
        (used by :meth:`feature_importance`).
        """

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Short architecture name recorded in :meth:`metadata`."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GraphDetector:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0:
            raise ValueError("cannot fit: training fold has zero positive examples")
        if n_neg == 0:
            raise ValueError("cannot fit: training fold has zero negative examples")

        self._feature_names = list(X.columns)
        train_ids = list(y.index)

        data, node_order = build_pyg_subgraph(self.graph, train_ids)
        data.y = torch.tensor(y.loc[node_order].to_numpy(), dtype=torch.float)

        train_subgraph = self.graph.subgraph(node_order)
        motif_labels = (
            compute_motif_features(train_subgraph, self.features_config)
            .set_index("account_id")
            .loc[node_order, MOTIF_COLUMNS]
        )
        data.motif_y = torch.tensor(motif_labels.to_numpy(), dtype=torch.float)

        # Internal early-stopping split carved out of the given training
        # labels only — val/test data is never touched here.
        rng = np.random.default_rng(self.config.random_seed)
        n = len(node_order)
        perm = rng.permutation(n)
        n_holdout = max(1, int(round(0.15 * n))) if n >= 10 else 0
        holdout_idx = set(perm[:n_holdout].tolist())
        fit_mask = torch.ones(n, dtype=torch.bool)
        for i in holdout_idx:
            fit_mask[i] = False

        in_dim = int(data.x.shape[1])
        self._encoder = self._build_encoder(in_dim)
        self._classifier_head = nn.Linear(self.config.hidden_dim, 1)
        self._motif_head = (
            nn.Linear(self.config.hidden_dim, len(MOTIF_COLUMNS))
            if self.config.motif_loss_weight > 0
            else None
        )

        params = list(self._encoder.parameters()) + list(self._classifier_head.parameters())
        if self._motif_head is not None:
            params += list(self._motif_head.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        pos_weight = torch.tensor(n_neg / n_pos)

        fanout = effective_fanout(self.config)
        fit_indices = [i for i in range(n) if bool(fit_mask[i])]
        batch_size = max(1, min(self.training_config.batch_size, len(fit_indices)))
        edge_src = data.edge_index[0].tolist()
        edge_dst = data.edge_index[1].tolist()
        in_edges = _build_in_edge_index(edge_dst, n)

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for _epoch in range(self.training_config.max_epochs):
            self._encoder.train()
            self._classifier_head.train()
            shuffled = [fit_indices[i] for i in rng.permutation(len(fit_indices))]
            for start in range(0, len(shuffled), batch_size):
                batch_seeds = shuffled[start : start + batch_size]
                node_ids, edge_ids = sample_khop_edges(in_edges, edge_src, batch_seeds, fanout, rng)
                local_index = {node: i for i, node in enumerate(node_ids)}

                node_idx_tensor = torch.tensor(node_ids, dtype=torch.long)
                x_batch = data.x[node_idx_tensor]
                if edge_ids:
                    edge_idx_tensor = torch.tensor(edge_ids, dtype=torch.long)
                    src_batch = torch.tensor(
                        [local_index[edge_src[e]] for e in edge_ids], dtype=torch.long
                    )
                    dst_batch = torch.tensor(
                        [local_index[edge_dst[e]] for e in edge_ids], dtype=torch.long
                    )
                    edge_index_batch = torch.stack([src_batch, dst_batch])
                    edge_time_batch = data.edge_time[edge_idx_tensor]
                else:
                    edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
                    edge_time_batch = torch.zeros((0,), dtype=torch.float)

                seed_local = torch.tensor([local_index[s] for s in batch_seeds], dtype=torch.long)

                optimizer.zero_grad()
                emb = self._encoder(x_batch, edge_index_batch, edge_time_batch)
                seed_emb = emb[seed_local]
                logits = self._classifier_head(seed_emb).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits, data.y[batch_seeds], pos_weight=pos_weight
                )
                if self._motif_head is not None:
                    motif_logits = self._motif_head(seed_emb)
                    motif_loss = F.binary_cross_entropy_with_logits(
                        motif_logits, data.motif_y[batch_seeds]
                    )
                    loss = loss + self.config.motif_loss_weight * motif_loss
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

            val_loss = self._holdout_loss(data, fit_mask, pos_weight)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = self._snapshot_state()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.training_config.early_stopping_patience:
                    break

        if best_state is not None:
            self._restore_state(best_state)

        self._is_fitted = True
        return self

    def _snapshot_state(self) -> dict[str, Any]:
        state = {
            "encoder": {k: v.clone() for k, v in self._encoder.state_dict().items()},
            "classifier_head": {
                k: v.clone() for k, v in self._classifier_head.state_dict().items()
            },
        }
        if self._motif_head is not None:
            state["motif_head"] = {k: v.clone() for k, v in self._motif_head.state_dict().items()}
        return state

    def _restore_state(self, state: dict[str, Any]) -> None:
        self._encoder.load_state_dict(state["encoder"])
        self._classifier_head.load_state_dict(state["classifier_head"])
        if self._motif_head is not None and "motif_head" in state:
            self._motif_head.load_state_dict(state["motif_head"])

    def _holdout_loss(self, data: Any, fit_mask: Any, pos_weight: Any) -> float:
        import torch
        import torch.nn.functional as F

        holdout_mask = ~fit_mask
        if int(holdout_mask.sum().item()) == 0:
            return 0.0
        self._encoder.eval()
        self._classifier_head.eval()
        with torch.no_grad():
            emb = self._encoder(data.x, data.edge_index, data.edge_time)
            logits = self._classifier_head(emb[holdout_mask]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, data.y[holdout_mask], pos_weight=pos_weight
            )
        return float(loss.item())

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        import torch

        if not self._is_fitted or self._encoder is None:
            raise GraphDetectorNotFittedError("predict_proba() called before fit()")

        all_node_ids = list(self.graph.nodes())
        data, node_order = build_pyg_subgraph(self.graph, all_node_ids)
        self._encoder.eval()
        self._classifier_head.eval()
        with torch.no_grad():
            emb = self._encoder(data.x, data.edge_index, data.edge_time)
            logits = self._classifier_head(emb).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()

        scores_by_id = pd.Series(probs, index=node_order)
        missing = [i for i in X.index if i not in scores_by_id.index]
        if missing:
            raise KeyError(f"account_id(s) not present in the detector's graph: {missing[:5]}")
        result: npt.NDArray[Any] = scores_by_id.loc[list(X.index)].to_numpy()
        return result

    def feature_importance(self) -> pd.Series:
        """Weight-based proxy: absolute first-layer input weight summed per
        feature. A real but imperfect signal — correlated features can
        understate each other's individual contribution — not a substitute
        for a permutation- or gradient-based importance measure.
        """
        if not self._is_fitted or self._encoder is None or self._feature_names is None:
            raise GraphDetectorNotFittedError("feature_importance() called before fit()")
        weight = self._encoder.first_layer_input_weight()
        importance = weight.abs().sum(dim=0).detach().numpy()
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.config.name,
            "architecture": self.architecture_name,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "fanout_per_layer": effective_fanout(self.config),
            "motif_loss_weight": self.config.motif_loss_weight,
            "random_seed": self.config.random_seed,
            "learning_rate": self.training_config.learning_rate,
            "weight_decay": self.training_config.weight_decay,
            "batch_size": self.training_config.batch_size,
            "max_epochs": self.training_config.max_epochs,
            "n_features": len(self._feature_names) if self._feature_names else None,
        }

    def save(self, path: Path | str) -> Path:
        import joblib

        if not self._is_fitted:
            raise GraphDetectorNotFittedError("save() called before fit()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> GraphDetector:
        import joblib

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded


__all__ = [
    "MOTIF_COLUMNS",
    "GraphDetector",
    "GraphDetectorNotFittedError",
    "attach_node_features",
    "effective_fanout",
    "build_pyg_subgraph",
]
