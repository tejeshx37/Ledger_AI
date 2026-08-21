"""Phase 5.5: Federated Learning experiment orchestration.

Runs three conditions:
1. Isolated: Each bank trains and scores alone on its local subgraph.
2. Federated: Collaborative training via FedAvg/FedProx strategy.
3. Pooled: Centralized training (legally impossible, acts as upper bound).

Reports overall metrics, ring-level recall by bank span, privacy epsilon, and
communication cost.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
import torch
from flwr.common import parameters_to_ndarrays

from ledger.config.settings import Settings
from ledger.data.canonical import CanonicalDataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.data.splitting import split_dataset
from ledger.evaluation.labels import build_binary_target
from ledger.features.pipeline import FeaturePipeline
from ledger.federated.boundary import (
    GLOBAL_BOUNDARY_REGISTRY,
    BoundaryGraphSAGEDetector,
    BoundaryTGATDetector,
    publish_local_embeddings,
)
from ledger.federated.client import LedgerFlowerClient, get_weights, set_weights
from ledger.federated.partitioning import compute_laundering_rings, partition_dataset
from ledger.federated.privacy import PrivacyAccountant, generate_pairwise_masks
from ledger.federated.server import LedgerFlowerStrategy
from ledger.models.graph_common import attach_node_features
from ledger.models.registry import GRAPH_MODEL_NAMES, get_detector
from ledger.utils.manifest import RunManifest
from ledger.utils.seeding import seed_everything

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ExperimentResult:
    """Paths and performance tables from the Phase 5 experiment."""

    run_id: str
    table_csv_path: Path
    table_json_path: Path
    manifest_path: Path
    results: list[dict[str, Any]]


def run_single_model_training(
    settings: Settings,
    dataset: CanonicalDataset,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> tuple[Any, FeaturePipeline, pd.DataFrame, pd.Series, np.ndarray]:
    """Train a standalone model on a dataset partition (used for Isolated and Pooled)."""
    y_all = build_binary_target(dataset)
    y_index = set(y_all.index)

    tr_ids = [i for i in train_ids if i in y_index]
    va_ids = [i for i in val_ids if i in y_index]
    te_ids = [i for i in test_ids if i in y_index]

    if not tr_ids:
        raise ValueError("Train split has no labeled examples")
    if not te_ids:
        raise ValueError("Test split has no labeled examples")

    pipeline = FeaturePipeline(config=settings.features)
    pipeline.fit(dataset, train_account_ids=tr_ids, held_out_account_ids=va_ids + te_ids)

    X_train = pipeline.transform(dataset, account_ids=tr_ids).set_index("account_id")
    X_test = pipeline.transform(dataset, account_ids=te_ids).set_index("account_id")
    y_train = y_all.loc[X_train.index]
    y_test = y_all.loc[X_test.index]

    attributed_graph = None
    if settings.model.name in GRAPH_MODEL_NAMES:
        full_features = pipeline.transform(dataset).set_index("account_id")
        full_graph = TemporalGraphBuilder(dataset).build_static_graph()
        attributed_graph = attach_node_features(full_graph, full_features)

    detector = get_detector(
        settings.model,
        training_config=settings.training,
        features_config=settings.features,
        graph=attributed_graph,
    )
    detector.fit(X_train, y_train)

    probs = detector.predict_proba(X_test)
    return detector, pipeline, X_test, y_test, probs


def evaluate_ring_performance(
    y_test_global: pd.Series,
    y_score_global: pd.Series,
    rings: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute recall on positive test accounts grouped by the bank span of their ring."""
    account_to_span = {}
    for r in rings:
        span = r["span"]
        for member in r["members"]:
            account_to_span[member] = span

    pos_test = y_test_global[y_test_global == 1].index

    spans_tp: dict[int, int] = {}
    spans_total: dict[int, int] = {}

    for acc in pos_test:
        span = account_to_span.get(acc, 1)
        # Check if the score is in y_score_global
        if acc not in y_score_global.index:
            continue
        is_tp = int(y_score_global.loc[acc] >= threshold)

        spans_tp[span] = spans_tp.get(span, 0) + is_tp
        spans_total[span] = spans_total.get(span, 0) + 1

    recalls = {}
    # Spans can be 1, 2, 3, 4+ (group 4+ together if N is large)
    # We will report specific spans up to 3, and group >= 4
    for span in sorted(spans_total.keys()):
        recalls[f"ring_recall_span_{span}"] = (
            spans_tp[span] / spans_total[span] if spans_total[span] > 0 else 0.0
        )

    # Summarize general ring recall (span >= 2) vs isolated (span == 1)
    ring_tp = sum(spans_tp[s] for s in spans_total if s >= 2)
    ring_total = sum(spans_total[s] for s in spans_total if s >= 2)
    recalls["ring_recall_multi_bank"] = ring_tp / ring_total if ring_total > 0 else 0.0

    isolated_tp = spans_tp.get(1, 0)
    isolated_total = spans_total.get(1, 0)
    recalls["ring_recall_local_bank"] = (
        isolated_tp / isolated_total if isolated_total > 0 else 0.0
    )

    return recalls


def run_federated_experiment(settings: Settings) -> ExperimentResult:
    """Run the 3-condition isolated / federated / pooled experiment."""
    seed_everything(settings.model.random_seed)

    logger.info("Loading canonical dataset for FL experiment", dataset=settings.data.dataset_name)
    global_dataset = CanonicalDataset.read(
        settings.paths.data_processed_dir, settings.data.dataset_name
    )

    # 1. Partition the dataset
    client_datasets, partition_stats = partition_dataset(
        global_dataset, settings.federated.num_clients, settings.federated.boundary_hash_key_env_var
    )

    # Extract laundering rings from global dataset to evaluate span recall later
    unique_banks = sorted(
        list({str(val) for val in global_dataset.accounts["institution_id"].dropna().unique()})
    )
    bank_to_client = {bank: i % settings.federated.num_clients for i, bank in enumerate(unique_banks)}
    node_to_client = {}
    for row in global_dataset.accounts.itertuples(index=False):
        acc_id = str(row.account_id)
        bank = str(row.institution_id) if pd.notna(row.institution_id) else None
        node_to_client[acc_id] = (
            bank_to_client[bank]
            if bank in bank_to_client
            else int(hashlib.md5(acc_id.encode("utf-8")).hexdigest(), 16)
            % settings.federated.num_clients
        )
    rings = compute_laundering_rings(global_dataset, node_to_client)

    # Project tx labels to account-level labels before splitting to align splits with model node classification
    y_account = build_binary_target(global_dataset)
    account_labels = pd.DataFrame(
        {
            "entity_id": y_account.index,
            "entity_type": "account",
            "label": y_account.map({0: "not_laundering", 1: "laundering"}),
            "label_source": "projected_account_level",
        }
    )
    splitting_dataset = CanonicalDataset(
        name=global_dataset.name,
        accounts=global_dataset.accounts,
        transactions=global_dataset.transactions,
        labels=account_labels,
        rings=global_dataset.rings,
    )

    # 2. Get train/val/test splits on global and client datasets
    # Note: To guarantee identical splits, we split the global dataset first
    # and then subset for each client, OR we split each client dataset independently.
    # The project brief requires comparing Isolated, Federated, and Pooled on "identical splits".
    # We will use the global dataset split, and then each client's split is its intersection with the global split.
    (g_train, g_val, g_test), _ = split_dataset(
        splitting_dataset, settings.data.split, seed=settings.model.random_seed
    )
    g_train_set = set(g_train)
    g_val_set = set(g_val)
    g_test_set = set(g_test)

    client_splits = {}
    for c, c_dataset in client_datasets.items():
        c_accs = set(c_dataset.accounts["account_id"])
        client_splits[c] = {
            "train": [x for x in c_dataset.accounts["account_id"] if x in g_train_set],
            "val": [x for x in c_dataset.accounts["account_id"] if x in g_val_set],
            "test": [x for x in c_dataset.accounts["account_id"] if x in g_test_set],
        }

    GLOBAL_BOUNDARY_REGISTRY.clear()

    # --- Condition A: Isolated ---
    logger.info("Running Isolated Condition")
    isolated_probs = []
    isolated_y_test = []
    for c, c_dataset in client_datasets.items():
        split = client_splits[c]
        if not split["train"] or not split["test"]:
            logger.warning("Client has empty splits, skipping isolated training", client=c)
            continue
        try:
            _, _, _, y_test_c, probs_c = run_single_model_training(
                settings, c_dataset, split["train"], split["val"], split["test"]
            )
            isolated_probs.append(probs_c)
            isolated_y_test.append(y_test_c)
        except Exception as exc:
            logger.error("Isolated training failed for client", client=c, exc=str(exc))

    y_test_isolated = pd.concat(isolated_y_test)
    y_score_isolated = pd.Series(np.concatenate(isolated_probs), index=y_test_isolated.index)

    # --- Condition B: Federated ---
    logger.info("Running Federated Condition")
    GLOBAL_BOUNDARY_REGISTRY.clear()

    # Instantiate local detectors and clients
    clients = {}
    detectors = {}
    pipelines = {}
    client_test_data = {}
    for c, c_dataset in client_datasets.items():
        split = client_splits[c]
        y_all_c = build_binary_target(c_dataset)
        y_index_c = set(y_all_c.index)

        tr_ids = [i for i in split["train"] if i in y_index_c]
        va_ids = [i for i in split["val"] if i in y_index_c]
        te_ids = [i for i in split["test"] if i in y_index_c]

        pipeline = FeaturePipeline(config=settings.features)
        pipeline.fit(c_dataset, train_account_ids=tr_ids, held_out_account_ids=va_ids + te_ids)
        pipelines[c] = pipeline

        X_train_c = pipeline.transform(c_dataset, account_ids=tr_ids).set_index("account_id")
        X_val_c = (
            pipeline.transform(c_dataset, account_ids=va_ids).set_index("account_id")
            if va_ids
            else None
        )
        X_test_c = pipeline.transform(c_dataset, account_ids=te_ids).set_index("account_id")

        y_train_c = y_all_c.loc[X_train_c.index]
        y_val_c = y_all_c.loc[X_val_c.index] if X_val_c is not None else None
        y_test_c = y_all_c.loc[X_test_c.index]

        # Use Boundary-aware version of GNN detectors
        attributed_graph = None
        if settings.model.name in GRAPH_MODEL_NAMES:
            full_features = pipeline.transform(c_dataset).set_index("account_id")
            full_graph = TemporalGraphBuilder(c_dataset).build_static_graph()
            attributed_graph = attach_node_features(full_graph, full_features)

        # Build local detector
        if settings.model.name == "graphsage":
            detector = BoundaryGraphSAGEDetector(
                config=settings.model,
                training_config=settings.training,
                features_config=settings.features,
                graph=attributed_graph,
            )
        elif settings.model.name == "tgat":
            detector = BoundaryTGATDetector(
                config=settings.model,
                training_config=settings.training,
                features_config=settings.features,
                graph=attributed_graph,
            )
        else:
            # Fallback to standard baseline detector
            detector = get_detector(
                settings.model,
                training_config=settings.training,
                features_config=settings.features,
                graph=attributed_graph,
            )

        # Initialize detector structures
        in_dim = X_train_c.shape[1]
        detector._encoder = detector._build_encoder(in_dim)
        from ledger.models.graph_common import MOTIF_COLUMNS

        detector._classifier_head = torch.nn.Linear(settings.model.hidden_dim, 1)
        detector._motif_head = (
            torch.nn.Linear(settings.model.hidden_dim, len(MOTIF_COLUMNS))
            if settings.model.motif_loss_weight > 0
            else None
        )
        detector._feature_names = list(X_train_c.columns)
        detector._is_fitted = True

        detectors[c] = detector

        client = LedgerFlowerClient(
            client_idx=c,
            detector=detector,
            X_train=X_train_c,
            y_train=y_train_c,
            X_val=X_val_c,
            y_val=y_val_c,
            settings=settings,
            registry=GLOBAL_BOUNDARY_REGISTRY,
        )
        clients[c] = client
        client_test_data[c] = (X_test_c, y_test_c)

    # Initialize global weights
    global_weights = get_weights(detectors[0])

    accountant = PrivacyAccountant(target_delta=settings.privacy.target_delta)

    # Round end callback to update boundary embeddings
    def on_round_end(round_idx: int, weights: list[np.ndarray]) -> None:
        if server_round % settings.federated.boundary_exchange_every_n_rounds == 0:
            logger.info("Updating boundary embeddings registry", round=round_idx)
            for c_idx, det in detectors.items():
                set_weights(det, weights)
                publish_local_embeddings(
                    det,
                    round_idx,
                    settings.federated.boundary_hash_key_env_var,
                    settings.privacy,
                    GLOBAL_BOUNDARY_REGISTRY,
                )

    strategy = LedgerFlowerStrategy(
        settings=settings,
        accountant=accountant,
        on_round_end_callback=on_round_end,
    )

    # Simulated FL Loop (strict reproducible execution)
    num_clients = settings.federated.num_clients
    rng = np.random.default_rng(settings.model.random_seed)
    total_bytes_exchanged = 0

    for server_round in range(1, settings.federated.num_rounds + 1):
        logger.info("Starting Federated Round", round=server_round)

        # 1. Sample clients
        sampled_size = int(
            np.ceil(num_clients * settings.federated.clients_per_round_fraction)
        )
        sampled_cids = rng.choice(num_clients, size=sampled_size, replace=False)

        # 2. Fit local clients
        fit_results = []
        for cid in sampled_cids:
            client = clients[cid]
            local_weights, num_examples, _ = client.fit(global_weights, {})
            fit_results.append((cid, local_weights, num_examples))

        # Secure Aggregation implementation
        if settings.federated.secure_aggregation:
            logger.info("Applying secure aggregation pairwise masking")
            shapes = [w.shape for w in global_weights]
            masks = generate_pairwise_masks(
                num_clients, shapes, seed=settings.model.random_seed + server_round
            )
            for i, (cid, local_w, num_ex) in enumerate(fit_results):
                masked_w = [w + m for w, m in zip(local_w, masks[cid])]
                fit_results[i] = (cid, masked_w, num_ex)

        # Map to Flower FitRes
        from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

        flower_results = []
        for cid, local_w, num_ex in fit_results:
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(local_w),
                num_examples=num_ex,
                metrics={},
            )
            # Mock client proxy
            class MockClientProxy:
                def __init__(self, cid: int) -> None:
                    self.cid = cid

            flower_results.append((MockClientProxy(cid), fit_res))  # type: ignore[arg-type]

        # 3. Aggregate weights at Strategy
        aggregated_parameters, _ = strategy.aggregate_fit(server_round, flower_results, [])
        if aggregated_parameters is None:
            raise ValueError(f"Aggregation failed at round {server_round}")

        # Update global weights
        global_weights = parameters_to_ndarrays(aggregated_parameters)

        # Update communication cost
        round_bytes = 2 * sampled_size * sum(w.nbytes for w in global_weights)
        total_bytes_exchanged += round_bytes

    # Evaluate final global weights on client test sets
    federated_probs = []
    federated_y_test = []
    for c, client in clients.items():
        set_weights(client.detector, global_weights)
        X_test_c, y_test_c = client_test_data[c]
        probs_c = client.detector.predict_proba(X_test_c)
        federated_probs.append(probs_c)
        federated_y_test.append(y_test_c)

    y_test_federated = pd.concat(federated_y_test)
    y_score_federated = pd.Series(np.concatenate(federated_probs), index=y_test_federated.index)

    # --- Condition C: Pooled ---
    logger.info("Running Pooled Condition")
    # Centralized training on global dataset using split
    _, _, _, y_test_pooled, probs_pooled = run_single_model_training(
        settings, global_dataset, list(g_train), list(g_val), list(g_test)
    )
    y_score_pooled = pd.Series(probs_pooled, index=y_test_pooled.index)

    # --- Compilation of Results ---
    conditions = [
        ("Isolated", y_test_isolated, y_score_isolated),
        ("Federated", y_test_federated, y_score_federated),
        ("Pooled", y_test_pooled, y_score_pooled),
    ]

    results_table = []
    for cond_name, y_t, y_s in conditions:
        auc = roc_auc_score(y_t.to_numpy(), y_s.to_numpy())
        ap = average_precision_score(y_t.to_numpy(), y_s.to_numpy())

        # Class predictions at alert threshold
        y_pred = (y_s >= settings.training.alert_threshold).astype(int)
        precision = precision_score(y_t.to_numpy(), y_pred, zero_division=0)
        recall = recall_score(y_t.to_numpy(), y_pred, zero_division=0)

        # Ring performance evaluation
        ring_metrics = evaluate_ring_performance(
            y_t, y_s, rings, threshold=settings.training.alert_threshold
        )

        row = {
            "condition": cond_name,
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            **ring_metrics,
        }
        results_table.append(row)

    # DP Epsilon consumed
    final_eps = accountant.get_epsilon() if settings.privacy.dp_enabled else 0.0

    logger.info(
        "FL Experiment Run Complete",
        final_epsilon=final_eps,
        total_communication_mb=total_bytes_exchanged / (1024 * 1024),
    )

    # Create run manifest and write files
    run_id = f"federated_{settings.data.dataset_name}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    run_manifest = RunManifest.create(
        run_id=run_id,
        config_hash=settings.hash(),
        dataset_checksums={},
        seed=settings.model.random_seed,
        repo_root=settings.paths.project_root,
    )
    manifest_path = run_manifest.write(settings.paths.runs_dir)
    run_dir = manifest_path.parent

    table_json_path = run_dir / "federated_comparison_table.json"
    with table_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results_table,
                "partition_stats": {
                    "nodes_per_bank": partition_stats.nodes_per_bank,
                    "cut_edge_count": partition_stats.cut_edge_count,
                    "rings_spanning_k_banks": partition_stats.rings_spanning_k_banks,
                },
                "privacy_consumed_epsilon": final_eps,
                "total_communication_bytes": total_bytes_exchanged,
            },
            f,
            indent=2,
        )

    # Write CSV version
    table_csv_path = run_dir / "federated_comparison_table.csv"
    pd.DataFrame(results_table).to_csv(table_csv_path, index=False)

    return ExperimentResult(
        run_id=run_id,
        table_csv_path=table_csv_path,
        table_json_path=table_json_path,
        manifest_path=manifest_path,
        results=results_table,
    )
