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

    Raises :class:`NotImplementedError` for datasets whose ground truth is
    transaction-level ('tx') rather than account-level: LEDGER's feature
    pipeline is account-indexed (Phase 2.6), and there is no
    transaction-level feature extraction yet to make those datasets
    trainable without silently producing a meaningless result.
    """
    if dataset.name not in POSITIVE_LABELS:
        raise KeyError(f"No positive-label mapping registered for dataset {dataset.name!r}")
    if LABEL_ENTITY_TYPE[dataset.name] != "account":
        raise NotImplementedError(
            f"dataset {dataset.name!r} labels transactions ('tx'), not accounts; "
            "transaction-level feature extraction is not implemented yet. Only "
            "account-labeled datasets (elliptic, elliptic_pp, baf) can be trained "
            "with the current baseline pipeline."
        )

    positive_label = POSITIVE_LABELS[dataset.name]
    excluded = EXCLUDED_LABELS.get(dataset.name, frozenset())

    labels = dataset.labels[dataset.labels["entity_type"] == "account"]
    labels = labels[~labels["label"].isin(excluded)]

    y = (labels["label"] == positive_label).astype(int)
    y.index = pd.Index(labels["entity_id"].to_numpy(), name="account_id")
    return y


__all__ = ["POSITIVE_LABELS", "EXCLUDED_LABELS", "LABEL_ENTITY_TYPE", "build_binary_target"]
