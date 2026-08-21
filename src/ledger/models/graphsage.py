"""GraphSAGE detector (Phase 4.1): inductive node classification via
learned neighbour aggregation (Hamilton, Ying & Leskovec 2017).

Configurable depth (``model.num_layers``) and per-layer neighbour fan-out
(``model.fanout_per_layer``); mini-batch trained via the sampler in
:mod:`ledger.models.graph_common`. GraphSAGE is inductive by construction —
its aggregator weights are shared across every node rather than learning a
per-node embedding lookup table, so the same trained weights generalise to
accounts and edges the optimiser never touched. LEDGER additionally
enforces this at the data level too: :meth:`~ledger.models.graph_common.GraphDetector.fit`
only ever message-passes over the training-node-induced subgraph (see
:mod:`ledger.models.graph_common`).

The encoder is a module-level class (not defined inside a method) guarded
by an import try/except, so importing :mod:`ledger.models` never hard-fails
without the optional ``ml`` extra installed, while a fitted detector still
pickles cleanly (a class defined inside a function/method body has no
stable module-qualified name, so ``pickle``/``joblib`` cannot serialise
instances of it).
"""

from __future__ import annotations

from typing import Any, cast

from ledger.models.graph_common import GraphDetector, effective_fanout

try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _SAGEEncoder(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            dims = [in_dim] + [hidden_dim] * num_layers
            self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)])
            self.dropout = dropout

        def forward(self, x: Any, edge_index: Any, edge_time: Any) -> Any:
            del edge_time  # GraphSAGE has no notion of edge recency
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def first_layer_input_weight(self) -> Any:
            # SAGEConv.lin_r is the "root" (self) linear transform,
            # applied directly to a node's own raw feature vector —
            # lin_l instead transforms the aggregated neighbour
            # features, so it is not a per-input-feature weight.
            first_conv = cast(SAGEConv, self.convs[0])
            return first_conv.lin_r.weight


class GraphSAGEDetector(GraphDetector):
    """Flat features plus graph structure via mean/max-style SAGE aggregation."""

    @property
    def architecture_name(self) -> str:
        return "graphsage"

    def _build_encoder(self, in_dim: int) -> Any:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "GraphSAGEDetector requires the optional 'ml' extra (torch, "
                "torch-geometric). Install with `pip install -e '.[ml]'`."
            )
        return _SAGEEncoder(
            in_dim, self.config.hidden_dim, self.config.num_layers, self.config.dropout
        )


__all__ = ["GraphSAGEDetector", "effective_fanout"]
