"""Adapter for the Elliptic Bitcoin dataset (Kaggle: ellipticco/elliptic-data-set).

Elliptic's graph nodes are transactions, not accounts: the classic
benchmark is a node-classification task over a transaction graph, with no
notion of an "account" at all. LEDGER's canonical schema is account-centric
(``accounts`` are graph nodes; ``transactions`` are edges between them), so
this adapter treats each Elliptic transaction node as a canonical account —
consistent with how the dataset is used in published work — and each entry
in ``elliptic_txs_edgelist.csv`` as a canonical transaction between those
pseudo-accounts. ``institution_id`` is unknown for every account, per the
project brief.

Elliptic ships no real-world timestamps, only a relative ``time_step`` in
``[1, 49]`` per node (this is the dataset's first disclosed feature column).
Both ``accounts.opened_at`` and ``transactions.timestamp`` here are a
deterministic function of that time step (epoch + time_step days) purely so
temporal graph construction and temporal splitting (Phase 2.4/2.5) have
something to order on; they are not real calendar times and downstream code
must not treat them as such.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset, json_text

N_FEATURES = 166
_FEATURE_COLUMNS = [f"feat_{i}" for i in range(1, N_FEATURES + 1)]
_EPOCH = pd.Timestamp("1970-01-01")

_FEATURES_RAW_SCHEMA = DataFrameSchema(
    {
        "txId": Column(int, nullable=False),
        **{c: Column(float, nullable=False) for c in _FEATURE_COLUMNS},
    },
    unique=["txId"],
    strict=True,
    coerce=True,
)

_CLASSES_RAW_SCHEMA = DataFrameSchema(
    {
        "txId": Column(int, nullable=False),
        "class": Column(str, Check.isin(["1", "2", "unknown"]), nullable=False),
    },
    unique=["txId"],
    strict=True,
    coerce=True,
)

_EDGELIST_RAW_SCHEMA = DataFrameSchema(
    {
        "txId1": Column(int, nullable=False),
        "txId2": Column(int, nullable=False),
    },
    strict=True,
    coerce=True,
)

_CLASS_TO_LABEL = {"1": "illicit", "2": "licit", "unknown": "unknown"}


class EllipticAdapter(DatasetAdapter):
    name = "elliptic"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        features = pd.read_csv(
            raw_dir / "elliptic_txs_features.csv",
            header=None,
            names=["txId", *_FEATURE_COLUMNS],
        )
        features = validate_raw(features, _FEATURES_RAW_SCHEMA, "elliptic_txs_features.csv")

        classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
        classes = classes.rename(columns={"class": "class"})
        classes["class"] = classes["class"].astype(str)
        classes = validate_raw(classes, _CLASSES_RAW_SCHEMA, "elliptic_txs_classes.csv")

        edgelist = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
        edgelist = validate_raw(edgelist, _EDGELIST_RAW_SCHEMA, "elliptic_txs_edgelist.csv")

        return {"features": features, "classes": classes, "edgelist": edgelist}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        features = raw["features"]
        classes = raw["classes"]
        edgelist = raw["edgelist"]

        time_step = features["feat_1"].astype(int)
        feature_records = features[_FEATURE_COLUMNS].astype(float).to_dict("records")
        attributes = [
            json_text({"time_step": int(ts), **rec})
            for ts, rec in zip(time_step, feature_records, strict=True)
        ]

        accounts = pd.DataFrame(
            {
                "account_id": features["txId"].astype(str),
                "institution_id": pd.array([None] * len(features), dtype="str"),
                "opened_at": [_EPOCH + pd.Timedelta(days=int(ts)) for ts in time_step],
                "attributes": attributes,
            }
        )

        time_step_by_tx = dict(zip(features["txId"], time_step, strict=True))
        src_time_step = edgelist["txId1"].map(time_step_by_tx)
        transactions = pd.DataFrame(
            {
                "tx_id": [f"elliptic_edge_{i}" for i in range(len(edgelist))],
                "src_account": edgelist["txId1"].astype(str),
                "dst_account": edgelist["txId2"].astype(str),
                "amount": pd.array([None] * len(edgelist), dtype="float64"),
                "currency": "unknown",
                "timestamp": [
                    _EPOCH + pd.Timedelta(days=int(ts)) if pd.notna(ts) else pd.NaT
                    for ts in src_time_step
                ],
                "channel": "unknown",
                "institution_src": pd.array([None] * len(edgelist), dtype="str"),
                "institution_dst": pd.array([None] * len(edgelist), dtype="str"),
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": classes["txId"].astype(str),
                "entity_type": "account",
                "label": classes["class"].map(_CLASS_TO_LABEL),
                "label_source": "elliptic_txs_classes",
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


__all__ = ["EllipticAdapter"]
