"""Maps each registered dataset's string label onto a binary "is this the
laundering/fraud class" target.

Every dataset's canonical ``labels.label`` values are dataset-specific
strings ('illicit', 'laundering', 'fraud', ...) set by its adapter (see
:mod:`ledger.data.adapters`); this module is the single place that decides
which string counts as the positive class, and which labels (e.g.
Elliptic's 'unknown') carry no ground truth and must be excluded from
training and evaluation entirely rather than counted as negatives.
"""

from __future__ import annotations

import pandas as pd

from ledger.data.canonical import CanonicalDataset

POSITIVE_LABELS: dict[str, str] = {
    "elliptic": "illicit",
    "elliptic_pp": "illicit",
    "ibm_aml": "laundering",
    "paysim": "fraud",
    "baf": "fraud",
    "ieee_cis": "fraud",
}

# Labels with no ground-truth signal (not "negative", genuinely unknown):
# excluded from both training and evaluation, never treated as a negative.
EXCLUDED_LABELS: dict[str, frozenset[str]] = {
    "elliptic": frozenset({"unknown"}),
    "elliptic_pp": frozenset({"unknown"}),
}

# Which labels.entity_type this dataset's ground truth lives on. The
# feature pipeline (Phase 2.6) is account-indexed; datasets whose labels
# are transaction-level ('tx') have no compatible feature representation
# yet — see build_binary_target's NotImplementedError below.
LABEL_ENTITY_TYPE: dict[str, str] = {
    "elliptic": "account",
    "elliptic_pp": "account",
    "ibm_aml": "tx",
    "paysim": "tx",
    "baf": "account",
    "ieee_cis": "tx",
}


def build_binary_target(dataset: CanonicalDataset) -> pd.Series:
    """Return a 0/1 Series indexed by account_id for every labeled account
    in ``dataset`` (excluding any label with no ground-truth signal).

    For transaction-labeled datasets (like ibm_aml), projects transaction-level
    labels onto account ownership: an account is positive (1) if it is the
    source or destination of any positive transaction, and negative (0) if
    it is involved only in negative transactions.
    """
    dataset_name = dataset.name
    for reg in POSITIVE_LABELS:
        if dataset_name.startswith(reg):
            dataset_name = reg
            break

    if dataset_name not in POSITIVE_LABELS:
        raise KeyError(f"No positive-label mapping registered for dataset {dataset_name!r}")

    positive_label = POSITIVE_LABELS[dataset_name]
    excluded = EXCLUDED_LABELS.get(dataset_name, frozenset())

    if LABEL_ENTITY_TYPE[dataset_name] == "tx":
        tx_labels = dataset.labels[
            (dataset.labels["entity_type"] == "tx") & (~dataset.labels["label"].isin(excluded))
        ]
        tx_binary = (tx_labels["label"] == positive_label).astype(int)
        tx_binary.index = tx_labels["entity_id"]

        txs = dataset.transactions.merge(
            tx_binary.to_frame("is_positive"),
            left_on="tx_id",
            right_index=True,
            how="inner",
        )

        src_labels = txs.groupby("src_account")["is_positive"].max()
        dst_labels = txs.groupby("dst_account")["is_positive"].max()

        combined = pd.concat([src_labels, dst_labels], axis=1)
        y_series = combined.max(axis=1).astype(int)
        y_series.index.name = "account_id"
        return y_series

    labels = dataset.labels[dataset.labels["entity_type"] == "account"]
    labels = labels[~labels["label"].isin(excluded)]

    y = (labels["label"] == positive_label).astype(int)
    y.index = pd.Index(labels["entity_id"].to_numpy(), name="account_id")
    return y


__all__ = ["POSITIVE_LABELS", "EXCLUDED_LABELS", "LABEL_ENTITY_TYPE", "build_binary_target"]
