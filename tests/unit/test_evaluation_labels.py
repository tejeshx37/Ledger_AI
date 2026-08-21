"""Unit tests for the positive-label mapping and binary target builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledger.data.adapters import get_adapter
from ledger.evaluation.labels import build_binary_target


def test_build_binary_target_elliptic_excludes_unknown(elliptic_raw_dir: Path) -> None:
    dataset = get_adapter("elliptic").run(elliptic_raw_dir)
    y = build_binary_target(dataset)

    known_labels = set(dataset.labels.loc[dataset.labels["label"] != "unknown", "entity_id"])
    assert set(y.index) == known_labels
    assert set(y.unique()).issubset({0, 1})

    illicit_ids = set(dataset.labels.loc[dataset.labels["label"] == "illicit", "entity_id"])
    assert set(y[y == 1].index) == illicit_ids


def test_build_binary_target_baf_uses_fraud_as_positive(baf_raw_dir: Path) -> None:
    dataset = get_adapter("baf").run(baf_raw_dir)
    y = build_binary_target(dataset)
    assert len(y) == len(dataset.accounts)
    fraud_ids = set(dataset.labels.loc[dataset.labels["label"] == "fraud", "entity_id"])
    assert set(y[y == 1].index) == fraud_ids


def test_build_binary_target_tx_entity_dataset_raises(ibm_aml_raw_dir: Path) -> None:
    dataset = get_adapter("ibm_aml").run(ibm_aml_raw_dir)
    with pytest.raises(NotImplementedError, match="transaction-level"):
        build_binary_target(dataset)


def test_build_binary_target_unregistered_dataset_raises() -> None:
    import pandas as pd

    from ledger.data.canonical import CanonicalDataset

    empty = CanonicalDataset(
        name="not_a_real_dataset",
        accounts=pd.DataFrame(columns=["account_id", "institution_id", "opened_at", "attributes"]),
        transactions=pd.DataFrame(
            columns=[
                "tx_id",
                "src_account",
                "dst_account",
                "amount",
                "currency",
                "timestamp",
                "channel",
                "institution_src",
                "institution_dst",
            ]
        ),
        labels=pd.DataFrame(columns=["entity_id", "entity_type", "label", "label_source"]),
        rings=pd.DataFrame(columns=["ring_id", "member_accounts", "pattern_type", "time_window"]),
    )
    with pytest.raises(KeyError):
        build_binary_target(empty)
