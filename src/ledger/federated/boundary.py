"""Phase 5.2: Boundary node embedding exchange.

This module provides the thread-safe registry where banks publish and read
pseudonymous external embeddings. It also wraps the SAGE and TGAT GNN encoders
to substitute boundary node representations with their received embeddings.
"""

from __future__ import annotations

import threading
from typing import Any

import networkx as nx
import numpy as np
import structlog

from ledger.federated.partitioning import get_pseudonym

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, SAGEConv

    from ledger.models.graphsage import _SAGEEncoder, GraphSAGEDetector
    from ledger.models.tgat import _TGATEncoder, TGATDetector

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Stub parents to prevent name error on load
    class nn:  # type: ignore[no-redef]
        Module = object

    class _SAGEEncoder:  # type: ignore[no-redef]
        pass

    class _TGATEncoder:  # type: ignore[no-redef]
        pass

    class GraphSAGEDetector:  # type: ignore[no-redef]
        pass

    class TGATDetector:  # type: ignore[no-redef]
        pass


class BoundaryEmbeddingRegistry:
    """Thread-safe in-memory registry holding the latest boundary embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings: dict[str, np.ndarray] = {}

    def clear(self) -> None:
        """Clear all stored embeddings."""
        with self._lock:
            self._embeddings.clear()

    def publish(self, pseudonym: str, embedding: np.ndarray) -> None:
        """Publish a boundary embedding vector for a pseudonym."""
        with self._lock:
            self._embeddings[pseudonym] = embedding.copy()

    def get_all(self) -> dict[str, np.ndarray]:
        """Return a copy of all currently registered pseudonymous embeddings."""
        with self._lock:
            return {k: v.copy() for k, v in self._embeddings.items()}


# Global singleton registry
GLOBAL_BOUNDARY_REGISTRY = BoundaryEmbeddingRegistry()


def get_external_transacting_local_accounts(graph: nx.MultiDiGraph) -> set[str]:
    """Identify local accounts in graph that transact with external boundary nodes.

    A node is local if its account_id does not start with 'pseudo_';
    it transacts externally if it has an edge to a node starting with 'pseudo_'.
    """
    local_external = set()
    for u, v in graph.edges():
        u_str = str(u)
        v_str = str(v)
        if u_str.startswith("pseudo_") and not v_str.startswith("pseudo_"):
            local_external.add(v_str)
        elif not u_str.startswith("pseudo_") and v_str.startswith("pseudo_"):
            local_external.add(u_str)
    return local_external


if _TORCH_AVAILABLE:

    class BoundarySAGEEncoder(nn.Module):
        """SAGE Conv encoder that incorporates external boundary node embeddings."""

        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            self.proj = nn.Linear(in_dim, hidden_dim)
            self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.dropout = dropout

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_time: torch.Tensor | None = None,
            boundary_embeddings: dict[str, torch.Tensor] | None = None,
            node_order: list[str] | None = None,
        ) -> torch.Tensor:
            del edge_time
            h = self.proj(x)
            if boundary_embeddings and node_order:
                for i, node_id in enumerate(node_order):
                    if node_id in boundary_embeddings:
                        h[i] = boundary_embeddings[node_id]

            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index)
                if i < len(self.convs) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        def first_layer_input_weight(self) -> torch.Tensor:
            return self.proj.weight

    class BoundaryTGATEncoder(nn.Module):
        """TGAT encoder that incorporates external boundary node embeddings."""

        def __init__(
            self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, time_dim: int
        ) -> None:
            super().__init__()
            self.time_dim = time_dim
            self.time_lin = nn.Linear(1, time_dim)
            self.proj = nn.Linear(in_dim, hidden_dim)
            self.convs = nn.ModuleList(
                [
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        edge_dim=time_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.dropout = dropout
            self.last_attention: list[tuple[Any, Any]] = []

        def _time_encode(self, edge_time: torch.Tensor) -> torch.Tensor:
            if edge_time.numel() == 0:
                return edge_time.new_zeros((0, self.time_dim))
            reference = edge_time.max()
            recency = reference - edge_time
            scale = recency.max().clamp(min=1.0)
            normalized = (recency / scale).unsqueeze(-1)
            raw = self.time_lin(normalized)
            return torch.cat([raw[:, :1], torch.sin(raw[:, 1:])], dim=-1)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_time: torch.Tensor,
            boundary_embeddings: dict[str, torch.Tensor] | None = None,
            node_order: list[str] | None = None,
            return_attention: bool = False,
        ) -> torch.Tensor:
            edge_attr = self._time_encode(edge_time)
            self.last_attention = []

            h = self.proj(x)
            if boundary_embeddings and node_order:
                for i, node_id in enumerate(node_order):
                    if node_id in boundary_embeddings:
                        h[i] = boundary_embeddings[node_id]

            for i, conv in enumerate(self.convs):
                if return_attention:
                    h, (ei, alpha) = conv(
                        h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                    )
                    self.last_attention.append((ei.detach(), alpha.detach()))
                else:
                    h = conv(h, edge_index, edge_attr=edge_attr)
                if i < len(self.convs) - 1:
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        def first_layer_input_weight(self) -> torch.Tensor:
            return self.proj.weight

    class BoundaryGraphSAGEDetector(GraphSAGEDetector):
        """GraphSAGE detector wrapped to use BoundarySAGEEncoder."""

        def _build_encoder(self, in_dim: int) -> Any:
            return BoundarySAGEEncoder(
                in_dim, self.config.hidden_dim, self.config.num_layers, self.config.dropout
            )

    class BoundaryTGATDetector(TGATDetector):
        """TGAT detector wrapped to use BoundaryTGATEncoder."""

        def _build_encoder(self, in_dim: int) -> Any:
            return BoundaryTGATEncoder(
                in_dim,
                self.config.hidden_dim,
                self.config.num_layers,
                self.config.dropout,
                self.config.time_encoding_dim,
            )


def publish_local_embeddings(
    detector: Any,
    round_idx: int,
    key_env_var: str,
    privacy_config: Any,
    registry: BoundaryEmbeddingRegistry = GLOBAL_BOUNDARY_REGISTRY,
) -> None:
    """Extract, clip, noise, and publish boundary embeddings for transacting accounts."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to publish embeddings.")

    if not getattr(detector, "_is_fitted", False) or detector._encoder is None:
        return

    local_external = get_external_transacting_local_accounts(detector.graph)
    if not local_external:
        return

    all_node_ids = list(detector.graph.nodes())
    from ledger.models.graph_common import build_pyg_subgraph

    data, node_order = build_pyg_subgraph(detector.graph, all_node_ids)

    detector._encoder.eval()
    with torch.no_grad():
        current_embeddings = registry.get_all()
        torch_embeddings = {
            k: torch.tensor(v, dtype=torch.float, device=data.x.device)
            for k, v in current_embeddings.items()
        }
        emb = detector._encoder(
            data.x,
            data.edge_index,
            data.edge_time,
            boundary_embeddings=torch_embeddings,
            node_order=node_order,
        )

    emb_np = emb.cpu().numpy()
    node_to_idx = {node: i for i, node in enumerate(node_order)}

    for node in local_external:
        if node in node_to_idx:
            idx = node_to_idx[node]
            vec = emb_np[idx]

            if privacy_config.dp_enabled:
                norm = np.linalg.norm(vec)
                if norm > privacy_config.gradient_clip_norm:
                    vec = vec * (privacy_config.gradient_clip_norm / (norm + 1e-9))

                noise_scale = privacy_config.noise_multiplier * privacy_config.gradient_clip_norm
                vec = vec + np.random.normal(0, noise_scale, size=vec.shape)

            pseudo_id = get_pseudonym(node, key_env_var)
            registry.publish(pseudo_id, vec)
