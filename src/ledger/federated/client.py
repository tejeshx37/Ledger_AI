"""Phase 5.3: Flower Federated Client.

This module implements the NumPyClient for Flower, wrapping a local detector's training
loop, setting global weights, and returning local updates (clipped/noised for DP).
It also implements proximal regularization for the FedProx strategy.
"""

from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
import structlog

from ledger.config.settings import Settings

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_weights(detector: Any) -> list[np.ndarray]:
    """Extract model parameters from detector components (encoder, classifier, motif heads)."""
    weights = []
    if detector._encoder is not None:
        for p in detector._encoder.parameters():
            weights.append(p.detach().cpu().numpy())
    if detector._classifier_head is not None:
        for p in detector._classifier_head.parameters():
            weights.append(p.detach().cpu().numpy())
    if detector._motif_head is not None:
        for p in detector._motif_head.parameters():
            weights.append(p.detach().cpu().numpy())
    return weights


def set_weights(detector: Any, weights: list[np.ndarray]) -> None:
    """Load parameter weights back into detector components in-place."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to set weights.")

    idx = 0
    if detector._encoder is not None:
        for p in detector._encoder.parameters():
            p.data.copy_(torch.tensor(weights[idx], device=p.device))
            idx += 1
    if detector._classifier_head is not None:
        for p in detector._classifier_head.parameters():
            p.data.copy_(torch.tensor(weights[idx], device=p.device))
            idx += 1
    if detector._motif_head is not None:
        for p in detector._motif_head.parameters():
            p.data.copy_(torch.tensor(weights[idx], device=p.device))
            idx += 1


def local_train(
    detector: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    settings: Settings,
    global_weights: list[np.ndarray],
    registry: Any,
) -> None:
    """Run PyTorch GNN optimization for local epochs, optionally adding FedProx proximal loss."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to run local training.")

    from ledger.features.motif_features import compute_motif_features
    from ledger.models.graph_common import (
        MOTIF_COLUMNS,
        _build_in_edge_index,
        build_pyg_subgraph,
        effective_fanout,
        sample_khop_edges,
    )

    if not getattr(detector, "_is_fitted", False) or detector._encoder is None:
        in_dim = X_train.shape[1]
        detector._encoder = detector._build_encoder(in_dim)
        detector._classifier_head = torch.nn.Linear(settings.model.hidden_dim, 1)
        detector._motif_head = (
            torch.nn.Linear(settings.model.hidden_dim, len(MOTIF_COLUMNS))
            if settings.model.motif_loss_weight > 0
            else None
        )
        detector._feature_names = list(X_train.columns)
        detector._is_fitted = True

    set_weights(detector, global_weights)

    train_ids = list(y_train.index)
    data, node_order = build_pyg_subgraph(detector.graph, train_ids)
    data.y = torch.tensor(y_train.loc[node_order].to_numpy(), dtype=torch.float)

    train_subgraph = detector.graph.subgraph(node_order)
    motif_labels = (
        compute_motif_features(train_subgraph, detector.features_config)
        .set_index("account_id")
        .loc[node_order, MOTIF_COLUMNS]
    )
    data.motif_y = torch.tensor(motif_labels.to_numpy(), dtype=torch.float)

    params = list(detector._encoder.parameters()) + list(detector._classifier_head.parameters())
    if detector._motif_head is not None:
        params += list(detector._motif_head.parameters())

    optimizer = torch.optim.Adam(
        params,
        lr=detector.training_config.learning_rate,
        weight_decay=detector.training_config.weight_decay,
    )

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / (n_pos + 1e-9))

    fanout = effective_fanout(detector.config)
    n = len(node_order)
    batch_size = max(1, min(detector.training_config.batch_size, n))

    edge_src = data.edge_index[0].tolist()
    edge_dst = data.edge_index[1].tolist()
    in_edges = _build_in_edge_index(edge_dst, n)

    rng = np.random.default_rng(detector.config.random_seed)

    received_embeddings = registry.get_all()
    torch_embeddings = {
        k: torch.tensor(v, dtype=torch.float, device=data.x.device)
        for k, v in received_embeddings.items()
    }

    detector._encoder.train()
    detector._classifier_head.train()
    if detector._motif_head is not None:
        detector._motif_head.train()

    for _epoch in range(settings.federated.local_epochs):
        shuffled = [i for i in rng.permutation(n)]
        for start in range(0, len(shuffled), batch_size):
            batch_seeds = shuffled[start : start + batch_size]
            node_ids, edge_ids = sample_khop_edges(in_edges, edge_src, batch_seeds, fanout, rng)
            local_index = {node: i for i, node in enumerate(node_ids)}

            node_idx_tensor = torch.tensor(node_ids, dtype=torch.long, device=data.x.device)
            x_batch = data.x[node_idx_tensor]

            if edge_ids:
                edge_idx_tensor = torch.tensor(edge_ids, dtype=torch.long, device=data.x.device)
                src_batch = torch.tensor(
                    [local_index[edge_src[e]] for e in edge_ids],
                    dtype=torch.long,
                    device=data.x.device,
                )
                dst_batch = torch.tensor(
                    [local_index[edge_dst[e]] for e in edge_ids],
                    dtype=torch.long,
                    device=data.x.device,
                )
                edge_index_batch = torch.stack([src_batch, dst_batch])
                edge_time_batch = data.edge_time[edge_idx_tensor]
            else:
                edge_index_batch = torch.zeros((2, 0), dtype=torch.long, device=data.x.device)
                edge_time_batch = torch.zeros((0,), dtype=torch.float, device=data.x.device)

            seed_local = torch.tensor(
                [local_index[s] for s in batch_seeds], dtype=torch.long, device=data.x.device
            )

            optimizer.zero_grad()
            emb = detector._encoder(
                x_batch,
                edge_index_batch,
                edge_time_batch,
                boundary_embeddings=torch_embeddings,
                node_order=node_ids,
            )
            seed_emb = emb[seed_local]
            logits = detector._classifier_head(seed_emb).squeeze(-1)

            loss = F.binary_cross_entropy_with_logits(
                logits,
                data.y[batch_seeds].to(logits.device),
                pos_weight=pos_weight.to(logits.device),
            )

            if detector._motif_head is not None:
                motif_logits = detector._motif_head(seed_emb)
                motif_loss = F.binary_cross_entropy_with_logits(
                    motif_logits, data.motif_y[batch_seeds].to(motif_logits.device)
                )
                loss = loss + detector.config.motif_loss_weight * motif_loss

            if settings.federated.strategy == "fedprox" and settings.federated.fedprox_mu > 0:
                prox_loss = 0.0
                idx = 0
                if detector._encoder is not None:
                    for p in detector._encoder.parameters():
                        g_w = torch.tensor(global_weights[idx], device=p.device)
                        prox_loss += torch.sum((p - g_w) ** 2)
                        idx += 1
                if detector._classifier_head is not None:
                    for p in detector._classifier_head.parameters():
                        g_w = torch.tensor(global_weights[idx], device=p.device)
                        prox_loss += torch.sum((p - g_w) ** 2)
                        idx += 1
                if detector._motif_head is not None:
                    for p in detector._motif_head.parameters():
                        g_w = torch.tensor(global_weights[idx], device=p.device)
                        prox_loss += torch.sum((p - g_w) ** 2)
                        idx += 1

                loss = loss + 0.5 * settings.federated.fedprox_mu * prox_loss

            loss.backward()
            optimizer.step()


class LedgerFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapper for local training of AML graph model."""

    def __init__(
        self,
        client_idx: int,
        detector: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        settings: Settings,
        registry: Any,
    ) -> None:
        self.client_idx = client_idx
        self.detector = detector
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.settings = settings
        self.registry = registry

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        return get_weights(self.detector)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any] | None = None
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        local_train(
            self.detector,
            self.X_train,
            self.y_train,
            self.settings,
            parameters,
            self.registry,
        )

        local_params = get_weights(self.detector)

        if self.settings.privacy.dp_enabled:
            from ledger.federated.privacy import apply_model_dp

            local_params = apply_model_dp(
                local_params,
                parameters,
                self.settings.privacy.gradient_clip_norm,
                self.settings.privacy.noise_multiplier,
            )

        return local_params, len(self.X_train), {}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any] | None = None
    ) -> tuple[float, int, dict[str, Any]]:
        set_weights(self.detector, parameters)

        if self.X_val is not None and not self.X_val.empty and self.y_val is not None:
            probs = self.detector.predict_proba(self.X_val)
            y_val_np = self.y_val.to_numpy()
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            loss = float(-np.mean(y_val_np * np.log(probs) + (1 - y_val_np) * np.log(1 - probs)))

            from sklearn.metrics import roc_auc_score

            try:
                auc = float(roc_auc_score(y_val_np, probs))
            except Exception:
                auc = 0.5

            return loss, len(self.X_val), {"roc_auc": auc}
        else:
            return 0.0, 0, {}
