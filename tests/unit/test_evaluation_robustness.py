"""Unit tests for Phase 7 (Evaluation, Fairness, and Robustness).

Covers bias audits, adversarial evasion simulation, Population Stability Index
drift tracking, and ablation studies.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ledger.data.canonical import CanonicalDataset
from ledger.evaluation.fairness import audit_bias
from ledger.evaluation.robustness import restructure_dataset_for_evasion
from ledger.evaluation.drift import calculate_psi, calculate_input_drift


def test_audit_bias() -> None:
    # 10 samples, binary labels, group A (0) and B (1)
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    sensitive = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    res = audit_bias(y_true, y_pred, sensitive)

    assert "demographic_parity_difference" in res
    assert "equalized_odds_difference" in res
    assert "group_metrics" in res
    assert "0" in res["group_metrics"]
    assert "1" in res["group_metrics"]


def test_restructure_dataset_for_evasion() -> None:
    accounts = pd.DataFrame([
        {"account_id": "acc_1", "bank_id": "bank_A"},
        {"account_id": "acc_2", "bank_id": "bank_A"},
    ])
    transactions = pd.DataFrame([
        {"tx_id": "tx_1", "src_account": "acc_1", "dst_account": "acc_2", "amount": 100.0, "timestamp": "2022-09-01T00:00:00"},
    ])
    dataset = CanonicalDataset(
        name="dummy",
        accounts=accounts,
        transactions=transactions,
        labels=pd.DataFrame(),
        rings=pd.DataFrame(),
    )

    # Level 1: Split amounts
    ds_split = restructure_dataset_for_evasion(dataset, ["tx_1"], level=1)
    assert len(ds_split.transactions) == 2
    assert ds_split.transactions.loc[0, "amount"] == 50.0
    assert ds_split.transactions.loc[1, "amount"] == 50.0

    # Level 2: Delays
    ds_delay = restructure_dataset_for_evasion(dataset, ["tx_1"], level=2)
    assert ds_delay.transactions.loc[0, "timestamp"] == "2022-09-03T00:00:00"

    # Level 3: Intermediate hops (mules)
    ds_mule = restructure_dataset_for_evasion(dataset, ["tx_1"], level=3)
    # Original transaction dropped, replaced by src -> mule and mule -> dst
    assert len(ds_mule.transactions) == 2
    assert "mule_tx_1" in ds_mule.accounts["account_id"].values
    assert ds_mule.transactions.loc[0, "dst_account"] == "mule_tx_1"
    assert ds_mule.transactions.loc[1, "src_account"] == "mule_tx_1"


def test_calculate_psi_no_drift() -> None:
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)

    psi = calculate_psi(expected, actual, num_buckets=10)
    # Identical distributions should have very small PSI (< 0.1)
    assert psi < 0.1


def test_calculate_psi_significant_drift() -> None:
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(2, 1, 1000)  # shifted mean

    psi = calculate_psi(expected, actual, num_buckets=10)
    # Shifted distributions should have high PSI (> 0.25)
    assert psi > 0.25


def test_calculate_input_drift() -> None:
    train_df = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, 100),
        "feat_2": np.random.normal(0, 1, 100),
    })
    test_df = pd.DataFrame({
        "feat_1": np.random.normal(2, 1, 100),  # drifted
        "feat_2": np.random.normal(0, 1, 100),  # non-drifted
    })

    res = calculate_input_drift(train_df, test_df)

    assert "average_psi" in res
    assert "feature_psis" in res
    assert "feat_1" in res["feature_psis"]
    assert res["feature_psis"]["feat_1"] > res["feature_psis"]["feat_2"]
