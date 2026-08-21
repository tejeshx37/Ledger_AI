"""Phase 7.2: Adversarial robustness simulation.

Provides tools to simulate transaction-ring evasion efforts (split amounts,
delays, and mule intermediate hops) and measure detector degradation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from ledger.data.canonical import CanonicalDataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.evaluation.labels import build_binary_target
from ledger.models.graph_common import attach_node_features

logger = structlog.get_logger(__name__)


def restructure_dataset_for_evasion(
    dataset: CanonicalDataset, ring_txs: list[str], level: int
) -> CanonicalDataset:
    """Restructure the canonical dataset to simulate evasion at different levels.

    - Level 0: Baseline (original transaction ring).
    - Level 1: Split amounts (split each transaction into two transactions of half the value).
    - Level 2: Delays (increase timestamp spacing of transactions).
    - Level 3: Intermediate hops (insert dummy account/mule between source and destination).
    """
    tx_df = dataset.transactions.copy()
    acc_df = dataset.accounts.copy()

    if len(ring_txs) == 0:
        return dataset

    if level == 1:
        # Split amounts
        ring_mask = tx_df["tx_id"].isin(ring_txs)
        ring_rows = tx_df[ring_mask]

        # Divide amount in half
        tx_df.loc[ring_mask, "amount"] = tx_df.loc[ring_mask, "amount"] / 2.0

        # Duplicate transactions
        dup_rows = ring_rows.copy()
        dup_rows["amount"] = dup_rows["amount"] / 2.0
        dup_rows["tx_id"] = dup_rows["tx_id"].apply(lambda x: f"{x}_split")

        tx_df = pd.concat([tx_df, dup_rows], ignore_index=True)

    elif level == 2:
        # Delays: shift timestamps by adding time offsets
        ring_mask = tx_df["tx_id"].isin(ring_txs)
        shifted_times = pd.to_datetime(
            tx_df.loc[ring_mask, "timestamp"]
        ) + pd.to_timedelta("2 days")
        tx_df.loc[ring_mask, "timestamp"] = shifted_times.dt.strftime("%Y-%m-%dT%H:%M:%S")

    elif level == 3:
        # Add intermediate hops (mules)
        ring_mask = tx_df["tx_id"].isin(ring_txs)
        new_mules = []
        new_txs = []

        for _, row in tx_df[ring_mask].iterrows():
            mule_id = f"mule_{row['tx_id']}"
            new_mules.append(
                {
                    "account_id": mule_id,
                    "bank_id": row.get("bank_id", "bank_unknown"),
                    "attributes": "{}",
                }
            )

            # Tx 1: src -> mule
            tx1 = row.copy()
            tx1["tx_id"] = f"{row['tx_id']}_hop1"
            tx1["dst_account"] = mule_id

            # Tx 2: mule -> dst
            tx2 = row.copy()
            tx2["tx_id"] = f"{row['tx_id']}_hop2"
            tx2["src_account"] = mule_id

            new_txs.append(tx1)
            new_txs.append(tx2)

        # Drop original transactions
        tx_df = tx_df[~ring_mask]

        if new_txs:
            tx_df = pd.concat([tx_df, pd.DataFrame(new_txs)], ignore_index=True)

        if new_mules:
            acc_df = pd.concat([acc_df, pd.DataFrame(new_mules)], ignore_index=True)

    return CanonicalDataset(
        name=dataset.name,
        accounts=acc_df,
        transactions=tx_df,
        labels=dataset.labels.copy(),
        rings=dataset.rings.copy(),
    )


def simulate_adversarial_evasion(
    detector: Any,
    pipeline: Any,
    dataset: CanonicalDataset,
    target_accounts: list[str],
    threshold: float = 0.5,
) -> dict[int, float]:
    """Measure detector's recall degradation on target_accounts under different evasion levels."""
    results = {}

    # Extract all transactions involving target accounts
    ring_txs = list(
        dataset.transactions[
            dataset.transactions["src_account"].isin(target_accounts)
            | dataset.transactions["dst_account"].isin(target_accounts)
        ]["tx_id"]
    )

    # Save original graph/features in detector to restore
    original_graph = detector.graph if hasattr(detector, "graph") else None

    try:
        for level in [0, 1, 2, 3]:
            # 1. Restructure dataset
            evaded_dataset = restructure_dataset_for_evasion(dataset, ring_txs, level)

            # 2. Re-extract features
            # Transform features on the modified dataset
            try:
                features_df = pipeline.transform(evaded_dataset)
                # Drop new mules from target check if any
                valid_targets = [acc for acc in target_accounts if acc in features_df["account_id"].values]
                if not valid_targets:
                    results[level] = 0.0
                    continue

                test_df = features_df[features_df["account_id"].isin(valid_targets)].set_index(
                    "account_id"
                )
                
                # Check GNN vs non-GNN
                if original_graph is not None:
                    # Build and attach new static graph
                    full_graph = TemporalGraphBuilder(evaded_dataset).build_static_graph()
                    attributed_graph = attach_node_features(
                        full_graph, features_df.set_index("account_id")
                    )
                    detector.graph = attributed_graph

                # 3. Predict probabilities
                probs = detector.predict_proba(test_df)
                recall = float(np.mean(probs >= threshold))
                results[level] = recall

            except Exception as exc:
                logger.error(
                    f"Evasion simulation failed at level {level}", error=str(exc)
                )
                results[level] = 0.0

    finally:
        # Restore detector graph
        if original_graph is not None:
            detector.graph = original_graph

    return results
