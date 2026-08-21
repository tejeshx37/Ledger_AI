"""Adapter for the IBM synthetic AML transactions dataset (Kaggle:
ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

This is the only registered dataset with real, per-transaction institution
identifiers on both sides of a transfer, which is exactly what the
federated experiment (Phase 5) needs to partition by bank. Account numbers
are only unique *within* a bank in the source file, so the canonical
``account_id`` is ``"<bank>_<account>"``.

The source amount/currency (``Amount Paid`` / ``Payment Currency``) is used
as the canonical ``amount``/``currency`` rather than the receiving-side
figures, since the sender's side is what the sender's bank actually
observes; the two can differ due to cross-currency conversion, which is
exactly the kind of structural signal Phase 4's motif features are meant to
pick up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset

_RAW_SCHEMA = DataFrameSchema(
    {
        "Timestamp": Column(str, nullable=False),
        "From Bank": Column(str, nullable=False),
        "Account": Column(str, nullable=False),
        "To Bank": Column(str, nullable=False),
        "Account.1": Column(str, nullable=False),
        "Amount Received": Column(float, Check.ge(0.0), nullable=False),
        "Receiving Currency": Column(str, nullable=False),
        "Amount Paid": Column(float, Check.ge(0.0), nullable=False),
        "Payment Currency": Column(str, nullable=False),
        "Payment Format": Column(str, nullable=False),
        "Is Laundering": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
)

_LABEL = {0: "not_laundering", 1: "laundering"}


class IbmAmlAdapter(DatasetAdapter):
    name = "ibm_aml"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(raw_dir / "HI-Small_Trans.csv")
        for col in ("From Bank", "Account", "To Bank", "Account.1"):
            df[col] = df[col].astype(str)
        df = validate_raw(df, _RAW_SCHEMA, "HI-Small_Trans.csv")
        return {"transactions": df}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        df = raw["transactions"]

        src_accounts = df["From Bank"] + "_" + df["Account"]
        dst_accounts = df["To Bank"] + "_" + df["Account.1"]

        account_rows = pd.concat(
            [
                pd.DataFrame({"account_id": src_accounts, "institution_id": df["From Bank"]}),
                pd.DataFrame({"account_id": dst_accounts, "institution_id": df["To Bank"]}),
            ],
            ignore_index=True,
        ).drop_duplicates(subset="account_id")
        accounts = pd.DataFrame(
            {
                "account_id": account_rows["account_id"],
                "institution_id": account_rows["institution_id"],
                "opened_at": pd.NaT,
                "attributes": "{}",
            }
        ).reset_index(drop=True)

        tx_id = [f"ibm_aml_{i}" for i in range(len(df))]
        transactions = pd.DataFrame(
            {
                "tx_id": tx_id,
                "src_account": src_accounts,
                "dst_account": dst_accounts,
                "amount": df["Amount Paid"],
                "currency": df["Payment Currency"],
                "timestamp": pd.to_datetime(df["Timestamp"]),
                "channel": df["Payment Format"],
                "institution_src": df["From Bank"],
                "institution_dst": df["To Bank"],
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": tx_id,
                "entity_type": "tx",
                "label": df["Is Laundering"].map(_LABEL),
                "label_source": "ibm_aml_HI-Small_Trans",
            }
        )

        rings = pd.DataFrame(
            columns=["ring_id", "member_accounts", "pattern_type", "time_window"]
        ).astype(
            {
                "ring_id": str,
                "member_accounts": str,
                "pattern_type": str,
                "time_window": str,
            }
        )

        return CanonicalDataset(
            name=self.name,
            accounts=accounts,
            transactions=transactions,
            labels=labels,
            rings=rings,
        )


__all__ = ["IbmAmlAdapter"]
