"""Temporal GAT detector (Phase 4.2): graph attention with a learned
time encoding, so the model can weight recent structure more heavily and
attention weights can be read back out for explanation (Phase 6).

Each edge's raw timestamp is converted to a recency value (relative to the
most recent edge in whatever graph is being processed) and passed through
a small Time2Vec-style encoding (one linear/"trend" term plus periodic
terms) into every ``GATConv`` layer's ``edge_dim`` input, so attention
coefficients are a function of both node features and how recent a
transaction is — not just static graph structure the way plain GAT or
GraphSAGE would see it.

The encoder is a module-level class (not defined inside a method) guarded
by an import try/except — see :mod:`ledger.models.graphsage` for why:
importing :mod:`ledger.models` must not hard-fail without the optional
``ml`` extra, and a class defined inside a function/method body cannot be
pickled (no stable module-qualified name), which would break
:meth:`~ledger.models.graph_common.GraphDetector.save`.
"""

from __future__ import annotations

from typing import Any, cast

from ledger.models.graph_common import (
    GraphDetector,
    GraphDetectorNotFittedError,
    build_pyg_subgraph,
    effective_fanout,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# A fixed, non-configured architectural choice (not a threshold or
# hyperparameter the project brief calls out as tunable): averaging
# several attention heads is standard GAT practice and keeps each layer's
# output dimension equal to hidden_dim regardless of head count.
_ATTENTION_HEADS = 4


if _TORCH_AVAILABLE:

    class _TGATEncoder(nn.Module):
        def __init__(
            self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, time_dim: int
        ) -> None:
            super().__init__()
            self.time_dim = time_dim
            self.time_lin = nn.Linear(1, time_dim)
            dims = [in_dim] + [hidden_dim] * num_layers
            self.convs = nn.ModuleList(
                [
                    GATConv(
                        dims[i],
                        dims[i + 1],
                        heads=_ATTENTION_HEADS,
                        concat=False,
                        edge_dim=time_dim,
                        dropout=dropout,
                    )
                    for i in range(num_layers)
                ]
            )
            self.dropout = dropout
            self.last_attention: list[tuple[Any, Any]] = []

        def _time_encode(self, edge_time: Any) -> Any:
            if edge_time.numel() == 0:
                return edge_time.new_zeros((0, self.time_dim))
            reference = edge_time.max()
            recency = reference - edge_time
            scale = recency.max().clamp(min=1.0)
            normalized = (recency / scale).unsqueeze(-1)
            raw = self.time_lin(normalized)
            return torch.cat([raw[:, :1], torch.sin(raw[:, 1:])], dim=-1)

        def forward(
            self, x: Any, edge_index: Any, edge_time: Any, return_attention: bool = False
        ) -> Any:
            edge_attr = self._time_encode(edge_time)
            self.last_attention = []
            for i, conv in enumerate(self.convs):
                if return_attention:
                    x, (ei, alpha) = conv(
                        x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                    )
                    self.last_attention.append((ei.detach(), alpha.detach()))
                else:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def first_layer_input_weight(self) -> Any:
            first_conv = cast(GATConv, self.convs[0])
            return first_conv.lin.weight


class TGATDetector(GraphDetector):
    """Flat features plus graph structure via time-aware graph attention."""

    @property
    def architecture_name(self) -> str:
        return "tgat"

    def _build_encoder(self, in_dim: int) -> Any:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "TGATDetector requires the optional 'ml' extra (torch, "
                "torch-geometric). Install with `pip install -e '.[ml]'`."
            )
        return _TGATEncoder(
            in_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.dropout,
            self.config.time_encoding_dim,
        )

    def attention_weights(self, account_ids: list[str]) -> list[dict[str, Any]]:
        """Return this model's last-layer attention weight for every edge
        in the subgraph induced by ``account_ids`` (from ``self.graph`` —
        the same attributed graph supplied at construction), as
        ``[{"src": account_id, "dst": account_id, "attention": float}, ...]``
        averaged across attention heads.

        Raises :class:`~ledger.models.graph_common.GraphDetectorNotFittedError`
        if called before :meth:`fit`. Intended for Phase 6's explanation
        layer: the minimal subgraph and features driving a specific alert.
        """
        if not self._is_fitted or self._encoder is None:
            raise GraphDetectorNotFittedError("attention_weights() called before fit()")

        data, node_order = build_pyg_subgraph(self.graph, account_ids)
        self._encoder.eval()
        with torch.no_grad():
            self._encoder(data.x, data.edge_index, data.edge_time, return_attention=True)

        if not self._encoder.last_attention:
            return []
        edge_index, alpha = self._encoder.last_attention[-1]
        mean_alpha = alpha.mean(dim=1) if alpha.dim() > 1 else alpha
        rows = []
        for e in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, e].item())
            dst_idx = int(edge_index[1, e].item())
            rows.append(
                {
                    "src": node_order[src_idx],
                    "dst": node_order[dst_idx],
                    "attention": float(mean_alpha[e].item()),
                }
            )
        return rows


__all__ = ["TGATDetector", "effective_fanout"]
