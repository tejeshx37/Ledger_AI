"""Adapter for the Elliptic++ actor-level release (git-disl/EllipticPlusPlus).

Elliptic++ extends Elliptic to actor (wallet address) level, which is what
enables ring-level evaluation: a "ring" is a set of actors, not a set of
individual transactions. This adapter is deliberately permissive about the
exact feature-column set in ``wallets_features_classes_combined.csv`` — the
release has evolved between versions and LEDGER does not assume a fixed
column count — validating only the two columns every version is documented
to carry (``address``, ``class``) and passing every remaining column
through unchanged into each account's ``attributes`` JSON.

Wallet classes are documented as ``1`` (illicit), ``2`` (licit), ``3``
(unknown). No per-edge timestamp is available at actor level in the public
release, so ``transactions.timestamp`` is null here; this dataset is not
used for the temporal-split headline numbers (Elliptic and IBM AML are).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from ledger.data.adapters.base import DatasetAdapter, validate_raw
from ledger.data.canonical import CanonicalDataset, json_text

_WALLETS_RAW_SCHEMA = DataFrameSchema(
    {
        "address": Column(str, nullable=False),
        "class": Column(str, Check.isin(["1", "2", "3"]), nullable=False),
    },
    unique=["address"],
    strict=False,
    coerce=True,
)

_ADDR_EDGELIST_RAW_SCHEMA = DataFrameSchema(
    {
        "input_address": Column(str, nullable=False),
        "output_address": Column(str, nullable=False),
    },
    strict=True,
    coerce=True,
)

_CLASS_TO_LABEL = {"1": "illicit", "2": "licit", "3": "unknown"}


class EllipticPlusPlusAdapter(DatasetAdapter):
    name = "elliptic_pp"

    def load_raw(self, raw_dir: Path) -> dict[str, pd.DataFrame]:
        wallets = pd.read_csv(raw_dir / "wallets_features_classes_combined.csv")
        wallets["address"] = wallets["address"].astype(str)
        wallets["class"] = wallets["class"].astype(str)
        wallets = validate_raw(
            wallets, _WALLETS_RAW_SCHEMA, "wallets_features_classes_combined.csv"
        )

        edgelist = pd.read_csv(raw_dir / "AddrAddr_edgelist.csv")
        edgelist["input_address"] = edgelist["input_address"].astype(str)
        edgelist["output_address"] = edgelist["output_address"].astype(str)
        edgelist = validate_raw(edgelist, _ADDR_EDGELIST_RAW_SCHEMA, "AddrAddr_edgelist.csv")

        return {"wallets": wallets, "edgelist": edgelist}

    def to_canonical(self, raw: dict[str, pd.DataFrame]) -> CanonicalDataset:
        wallets = raw["wallets"]
        edgelist = raw["edgelist"]

        feature_cols = [c for c in wallets.columns if c not in ("address", "class")]
        feature_records = wallets[feature_cols].to_dict("records")
        attributes = [json_text(rec) for rec in feature_records]

        accounts = pd.DataFrame(
            {
                "account_id": wallets["address"],
                "institution_id": pd.array([None] * len(wallets), dtype="str"),
                "opened_at": pd.NaT,
                "attributes": attributes,
            }
        )

        transactions = pd.DataFrame(
            {
                "tx_id": [f"elliptic_pp_edge_{i}" for i in range(len(edgelist))],
                "src_account": edgelist["input_address"],
                "dst_account": edgelist["output_address"],
                "amount": pd.array([None] * len(edgelist), dtype="float64"),
                "currency": "unknown",
                "timestamp": pd.NaT,
                "channel": "unknown",
                "institution_src": pd.array([None] * len(edgelist), dtype="str"),
                "institution_dst": pd.array([None] * len(edgelist), dtype="str"),
            }
        )

        labels = pd.DataFrame(
            {
                "entity_id": wallets["address"],
                "entity_type": "account",
                "label": wallets["class"].map(_CLASS_TO_LABEL),
                "label_source": "wallets_features_classes_combined",
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


__all__ = ["EllipticPlusPlusAdapter"]
