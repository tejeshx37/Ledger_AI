"""Adapter for the IEEE-CIS Fraud Detection competition data (Kaggle
competition: ieee-fraud-detection; ``train_transaction.csv`` +
``train_identity.csv``).

IEEE-CIS has no real account or counterparty identifier at all — it is a
pure transaction-classification dataset. LEDGER's canonical schema needs
two endpoints per transaction, so this adapter uses ``card1`` (the most
stable of the six anonymised card-fingerprint columns) as a proxy source
account, and the transaction's ``ProductCD`` category as a proxy
destination "account". Neither is a real identity: ``dst_account`` in
particular is a coarse category, not a merchant, and downstream code must
not treat it as one. ``TransactionDT`` is a relative offset (seconds from
an undisclosed reference point per the competition's own documentation),
so canonical ``timestamp`` here is relative ordering, not a real calendar
time — consistent with this dataset's stated role as a *secondary*
tabular baseline rather than a temporal-graph benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset, json_text

_EPOCH = pd.Timestamp("1970-01-01")

_CARD_ATTRIBUTE_COLUMNS = [
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]

_TRANSACTION_RAW_SCHEMA = DataFrameSchema(
    {
        "TransactionID": Column(int, nullable=False),
        "isFraud": Column(int, Check.isin([0, 1]), nullable=False),
        "TransactionDT": Column(int, Check.ge(0), nullable=False),
        "TransactionAmt": Column(float, Check.ge(0.0), nullable=False),
        "ProductCD": Column(str, nullable=False),
    },
    unique=["TransactionID"],
    strict=False,
    coerce=True,
)

_IDENTITY_RAW_SCHEMA = DataFrameSchema(
    {"TransactionID": Column(int, nullable=False)},
    unique=["TransactionID"],
    strict=False,
    coerce=True,
)

_LABEL = {0: "not_fraud", 1: "fraud"}


class IeeeCisAdapter(DatasetAdapter):
    name = "ieee_cis"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        transaction = pd.read_csv(raw_dir / "train_transaction.csv")
        transaction["ProductCD"] = transaction["ProductCD"].astype(str)
        transaction = validate_raw(transaction, _TRANSACTION_RAW_SCHEMA, "train_transaction.csv")

        identity = pd.read_csv(raw_dir / "train_identity.csv")
        identity = validate_raw(identity, _IDENTITY_RAW_SCHEMA, "train_identity.csv")

        return {"transaction": transaction, "identity": identity}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        df = raw["transaction"].merge(raw["identity"], on="TransactionID", how="left")

        card_key = df["card1"].fillna(-1).astype(int).astype(str)
        src_account = "card_" + card_key
        dst_account = "merchant_" + df["ProductCD"]

        present_attr_cols = [c for c in _CARD_ATTRIBUTE_COLUMNS if c in df.columns]
        present_attr_cols += [
            c
            for c in ("DeviceType", "DeviceInfo")
            if c in df.columns and c not in present_attr_cols
        ]
        card_attrs = df[present_attr_cols].astype(object).where(df[present_attr_cols].notna(), None)
        card_attrs.insert(0, "src_account", src_account)
        card_records = card_attrs.groupby("src_account", as_index=True).first()

        card_accounts = pd.DataFrame(
            {
                "account_id": card_records.index,
                "institution_id": pd.array([None] * len(card_records), dtype="str"),
                "opened_at": pd.NaT,
                "attributes": [json_text(rec) for rec in card_records.to_dict("records")],
            }
        )
        merchant_ids = sorted(dst_account.unique())
        merchant_accounts = pd.DataFrame(
            {
                "account_id": merchant_ids,
                "institution_id": pd.array([None] * len(merchant_ids), dtype="str"),
                "opened_at": pd.NaT,
                "attributes": "{}",
            }
        )
        accounts = pd.concat([card_accounts, merchant_accounts], ignore_index=True).drop_duplicates(
            subset="account_id"
        )

        tx_id = df["TransactionID"].astype(str)
        transactions = pd.DataFrame(
            {
                "tx_id": tx_id,
                "src_account": src_account,
                "dst_account": dst_account,
                "amount": df["TransactionAmt"],
                "currency": "USD",
                "timestamp": [_EPOCH + pd.Timedelta(seconds=int(dt)) for dt in df["TransactionDT"]],
                "channel": df["ProductCD"],
                "institution_src": pd.array([None] * len(df), dtype="str"),
                "institution_dst": pd.array([None] * len(df), dtype="str"),
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": tx_id,
                "entity_type": "tx",
                "label": df["isFraud"].map(_LABEL),
                "label_source": "ieee_cis_train_transaction",
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


__all__ = ["IeeeCisAdapter"]
