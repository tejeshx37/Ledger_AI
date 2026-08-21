"""Phase 5.1: Dataset partitioning by account ownership.

This module splits the transaction graph into N simulated institutions. Cross-institution
edges are cut: each bank sees only its own side of a transfer and a
pseudonymous counterparty reference.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import structlog

from ledger.data.canonical import CanonicalDataset

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PartitionStats:
    """Statistics for the dataset partitioning."""

    nodes_per_bank: dict[int, int]
    cut_edge_count: int
    rings_spanning_k_banks: dict[int, int]


def get_pseudonym(account_id: str, key_env_var: str) -> str:
    """Generate a stable, secure pseudonym for an external account using HMAC-SHA256."""
    key_str = os.environ.get(key_env_var, "ledger_default_secret_key_for_testing")
    key_bytes = key_str.encode("utf-8")
    digest = hmac.new(key_bytes, account_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"pseudo_{digest}"


def compute_laundering_rings(
    dataset: CanonicalDataset, node_to_client: dict[str, int]
) -> list[dict[str, Any]]:
    """Identify laundering rings and count how many clients they span.

    Rings are weakly connected components of the laundering transaction subgraph.
    """
    laundering_txs = dataset.labels[
        (dataset.labels["entity_type"] == "tx") & (dataset.labels["label"] == "laundering")
    ]
    la_tx_ids = set(laundering_txs["entity_id"])

    la_df = dataset.transactions[dataset.transactions["tx_id"].isin(la_tx_ids)]

    g = nx.Graph()
    for row in la_df.itertuples(index=False):
        g.add_edge(str(row.src_account), str(row.dst_account))

    components = list(nx.connected_components(g))
    rings = []
    for ring_idx, comp in enumerate(components):
        clients = {node_to_client[node] for node in comp if node in node_to_client}
        rings.append(
            {
                "ring_id": f"ring_{ring_idx}",
                "members": list(comp),
                "clients": list(clients),
                "span": len(clients),
            }
        )
    return rings


def partition_dataset(
    dataset: CanonicalDataset,
    num_clients: int,
    key_env_var: str,
) -> tuple[dict[int, CanonicalDataset], PartitionStats]:
    """Split the dataset into N simulated institutions by account ownership.

    Cross-institution edges are cut — each bank sees only its own side of a transfer
    and a pseudonymous counterparty reference.
    """
    logger.info("Starting dataset partitioning", dataset_name=dataset.name, num_clients=num_clients)

    unique_banks = sorted(
        list({str(val) for val in dataset.accounts["institution_id"].dropna().unique()})
    )

    if unique_banks:
        bank_to_client = {bank: i % num_clients for i, bank in enumerate(unique_banks)}
        logger.info("Mapped bank IDs to client indices", bank_to_client=bank_to_client)
    else:
        bank_to_client = {}
        logger.warning(
            "No institution IDs found in dataset. Falling back to hash-based partitioning."
        )

    node_to_client: dict[str, int] = {}
    for row in dataset.accounts.itertuples(index=False):
        acc_id = str(row.account_id)
        bank = str(row.institution_id) if pd.notna(row.institution_id) else None
        if bank is not None and bank in bank_to_client:
            client_idx = bank_to_client[bank]
        else:
            client_idx = int(hashlib.md5(acc_id.encode("utf-8")).hexdigest(), 16) % num_clients
        node_to_client[acc_id] = client_idx

    txs = dataset.transactions.copy()
    txs["src_client"] = txs["src_account"].map(node_to_client)
    txs["dst_client"] = txs["dst_account"].map(node_to_client)

    cut_mask = (
        (txs["src_client"] != txs["dst_client"])
        & txs["src_client"].notna()
        & txs["dst_client"].notna()
    )
    cut_edge_count = int(cut_mask.sum())

    client_datasets: dict[int, CanonicalDataset] = {}

    for c in range(num_clients):
        c_txs_mask = (txs["src_client"] == c) | (txs["dst_client"] == c)
        c_txs = txs[c_txs_mask].copy()

        is_src_external = c_txs["src_client"] != c
        is_dst_external = c_txs["dst_client"] != c

        if not c_txs.empty:
            c_txs.loc[is_src_external, "src_account"] = c_txs.loc[is_src_external, "src_account"].apply(
                lambda x: get_pseudonym(str(x), key_env_var)
            )
            c_txs.loc[is_dst_external, "dst_account"] = c_txs.loc[is_dst_external, "dst_account"].apply(
                lambda x: get_pseudonym(str(x), key_env_var)
            )

        c_txs = c_txs.drop(columns=["src_client", "dst_client"])

        local_accounts = dataset.accounts[
            dataset.accounts["account_id"].astype(str).map(node_to_client) == c
        ].copy()

        all_src_accounts = set(c_txs["src_account"].astype(str)) if not c_txs.empty else set()
        all_dst_accounts = set(c_txs["dst_account"].astype(str)) if not c_txs.empty else set()
        all_tx_accounts = all_src_accounts.union(all_dst_accounts)

        local_account_ids = set(local_accounts["account_id"].astype(str))
        pseudo_account_ids = all_tx_accounts - local_account_ids

        pseudo_accounts_df = pd.DataFrame(
            {
                "account_id": list(pseudo_account_ids),
                "institution_id": ["external"] * len(pseudo_account_ids),
                "opened_at": [pd.NaT] * len(pseudo_account_ids),
                "attributes": ["{}"] * len(pseudo_account_ids),
            }
        )

        c_accounts = pd.concat([local_accounts, pseudo_accounts_df], ignore_index=True)

        c_labels = dataset.labels[
            dataset.labels["entity_id"].isin(local_accounts["account_id"])
            | dataset.labels["entity_id"].isin(c_txs["tx_id"])
        ].copy()

        c_rings = pd.DataFrame(
            columns=["ring_id", "member_accounts", "pattern_type", "time_window"]
        ).astype(
            {
                "ring_id": str,
                "member_accounts": str,
                "pattern_type": str,
                "time_window": str,
            }
        )

        client_datasets[c] = CanonicalDataset(
            name=f"{dataset.name}_client_{c}",
            accounts=c_accounts,
            transactions=c_txs,
            labels=c_labels,
            rings=c_rings,
        )

    nodes_per_bank = {
        c: sum(1 for node, client in node_to_client.items() if client == c)
        for c in range(num_clients)
    }

    rings = compute_laundering_rings(dataset, node_to_client)
    span_counts: dict[int, int] = {}
    for r in rings:
        span = r["span"]
        span_counts[span] = span_counts.get(span, 0) + 1
    rings_spanning_k_banks = {k: span_counts.get(k, 0) for k in range(1, num_clients + 1)}

    stats = PartitionStats(
        nodes_per_bank=nodes_per_bank,
        cut_edge_count=cut_edge_count,
        rings_spanning_k_banks=rings_spanning_k_banks,
    )

    logger.info(
        "Dataset partitioning complete",
        cut_edge_count=cut_edge_count,
        rings_spanning_k_banks=rings_spanning_k_banks,
    )

    return client_datasets, stats
