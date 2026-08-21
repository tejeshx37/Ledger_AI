"""Shared fixtures for Phase 2 (data layer) tests: tiny synthetic raw files
matching each registered dataset's real column structure, small enough to
run in milliseconds but large enough to exercise every canonical table and
both graph/temporal code paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def elliptic_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "elliptic"
    raw_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    n_nodes = 40
    tx_ids = list(range(1, n_nodes + 1))
    time_steps = [((i - 1) % 10) + 1 for i in tx_ids]

    rows = []
    for tx_id, ts in zip(tx_ids, time_steps, strict=True):
        rows.append([tx_id, float(ts)] + list(rng.normal(size=165)))
    pd.DataFrame(rows).to_csv(raw_dir / "elliptic_txs_features.csv", header=False, index=False)

    classes = []
    for _ in tx_ids:
        r = rng.random()
        classes.append("1" if r < 0.2 else ("2" if r < 0.8 else "unknown"))
    pd.DataFrame({"txId": tx_ids, "class": classes}).to_csv(
        raw_dir / "elliptic_txs_classes.csv", index=False
    )

    edges = []
    for _ in range(80):
        a, b = rng.choice(tx_ids, size=2, replace=False)
        edges.append((int(a), int(b)))
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(
        raw_dir / "elliptic_txs_edgelist.csv", index=False
    )
    return raw_dir


@pytest.fixture
def elliptic_pp_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "elliptic_pp"
    raw_dir.mkdir(parents=True)
    addrs = [f"addr_{i}" for i in range(10)]
    pd.DataFrame(
        {
            "address": addrs,
            "class": ["1", "2", "3", "2", "1", "2", "2", "3", "1", "2"],
            "num_txs": range(10),
            "total_received": [float(i) * 1.5 for i in range(10)],
        }
    ).to_csv(raw_dir / "wallets_features_classes_combined.csv", index=False)
    pd.DataFrame(
        {
            "input_address": ["addr_0", "addr_1", "addr_2", "addr_3"],
            "output_address": ["addr_1", "addr_2", "addr_3", "addr_4"],
        }
    ).to_csv(raw_dir / "AddrAddr_edgelist.csv", index=False)
    return raw_dir


@pytest.fixture
def ibm_aml_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "ibm_aml"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "Timestamp": ["2022/09/01 00:20", "2022/09/01 00:25", "2022/09/01 01:00"],
            "From Bank": [1, 1, 2],
            "Account": ["A1", "A1", "A2"],
            "To Bank": [2, 3, 1],
            "Account.1": ["B1", "B2", "A1"],
            "Amount Received": [100.0, 50.0, 25.0],
            "Receiving Currency": ["USD", "USD", "USD"],
            "Amount Paid": [100.0, 50.0, 25.0],
            "Payment Currency": ["USD", "USD", "USD"],
            "Payment Format": ["ACH", "Wire", "ACH"],
            "Is Laundering": [0, 1, 0],
        }
    ).to_csv(raw_dir / "HI-Small_Trans.csv", index=False)
    return raw_dir


@pytest.fixture
def paysim_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "paysim"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["CASH_OUT", "PAYMENT", "TRANSFER"],
            "amount": [100.0, 50.0, 200.0],
            "nameOrig": ["C1", "C2", "C1"],
            "oldbalanceOrg": [1000.0, 500.0, 900.0],
            "newbalanceOrig": [900.0, 450.0, 700.0],
            "nameDest": ["C2", "M1", "C3"],
            "oldbalanceDest": [0.0, 0.0, 0.0],
            "newbalanceDest": [100.0, 50.0, 200.0],
            "isFraud": [0, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
        }
    ).to_csv(raw_dir / "PS_20174392719_1491204439457_log.csv", index=False)
    return raw_dir


@pytest.fixture
def baf_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "baf"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0],
            "customer_age": [30, 45, 60],
            "month": [0, 3, 7],
            "income": [0.3, 0.6, 0.9],
        }
    ).to_csv(raw_dir / "Base.csv", index=False)
    return raw_dir


@pytest.fixture
def ieee_cis_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "ieee_cis"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "TransactionID": [1000, 1001, 1002],
            "isFraud": [0, 1, 0],
            "TransactionDT": [86400, 86500, 90000],
            "TransactionAmt": [50.0, 120.0, 75.0],
            "ProductCD": ["W", "C", "W"],
            "card1": [1000, 1001, 1000],
            "card2": [100.0, 200.0, 100.0],
            "addr1": [10.0, 20.0, 10.0],
        }
    ).to_csv(raw_dir / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [1000], "DeviceType": ["mobile"]}).to_csv(
        raw_dir / "train_identity.csv", index=False
    )
    return raw_dir
