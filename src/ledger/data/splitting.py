"""Train/validation/test splitting over a canonical dataset's labelled
entities (transactions or accounts, whichever the dataset's ``labels``
table uses).

Three strategies, selected by :class:`ledger.config.models.SplitConfig`:

``temporal``
    Sort by timestamp, cut at the configured fractions. The only strategy
    used for headline numbers — see the module docstring on
    :class:`~ledger.config.models.SplitConfig` for why.
``random``
    A uniform random shuffle, seeded for reproducibility. Exists only to
    measure the temporal-leakage gap, never for headline claims.
``institution``
    Hold out one institution's entities entirely for test; split the rest
    into train/val. Used by the federated experiment (Phase 5).

Every split function returns three arrays of ``entity_id`` (not raw
positional indices — those are fragile across a table reload) plus a
:class:`SplitManifest` recording the strategy, its boundaries, and the
label balance of each fold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from ledger.config.models import SplitConfig
from ledger.data.canonical import CanonicalDataset

SplitArrays = tuple[npt.NDArray[np.str_], npt.NDArray[np.str_], npt.NDArray[np.str_]]


@dataclass(frozen=True)
class SplitManifest:
    """Records exactly how a dataset was split, for the run manifest and for audit."""

    dataset_name: str
    strategy: str
    n_train: int
    n_val: int
    n_test: int
    boundaries: dict[str, str | None]
    class_balance: dict[str, dict[str, float]]
    seed: int | None

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    @classmethod
    def read(cls, path: Path | str) -> SplitManifest:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)


def build_entity_frame(dataset: CanonicalDataset) -> pd.DataFrame:
    """Join ``labels`` (excluding ring-level labels) onto a timestamp and
    institution_id, sourced from ``transactions`` for tx-level labels and
    from ``accounts`` for account-level labels.
    """
    labels = dataset.labels[dataset.labels["entity_type"] != "ring"]

    tx_labels = labels[labels["entity_type"] == "tx"]
    tx_part = tx_labels.merge(
        dataset.transactions[["tx_id", "timestamp", "institution_src"]],
        left_on="entity_id",
        right_on="tx_id",
        how="left",
    ).rename(columns={"institution_src": "institution_id"})

    account_labels = labels[labels["entity_type"] == "account"]
    account_part = account_labels.merge(
        dataset.accounts[["account_id", "opened_at", "institution_id"]],
        left_on="entity_id",
        right_on="account_id",
        how="left",
    ).rename(columns={"opened_at": "timestamp"})

    combined = pd.concat([tx_part, account_part], ignore_index=True)
    return combined[["entity_id", "entity_type", "label", "timestamp", "institution_id"]]


def _iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return str(pd.Timestamp(ts).isoformat())


def _class_balance(folds: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    balance: dict[str, dict[str, float]] = {}
    for fold_name, df in folds.items():
        if len(df) == 0:
            balance[fold_name] = {}
            continue
        counts = df["label"].value_counts(normalize=True)
        balance[fold_name] = {str(label): float(frac) for label, frac in counts.items()}
    return balance


def _ids(df: pd.DataFrame) -> npt.NDArray[np.str_]:
    return np.asarray(df["entity_id"].to_numpy(dtype=str), dtype=np.str_)


def split_temporal(
    entities: pd.DataFrame, split_config: SplitConfig, dataset_name: str
) -> tuple[SplitArrays, SplitManifest]:
    """Sort by timestamp and cut at the configured train/val/test fractions.

    Raises :class:`ValueError` if any entity has an unknown timestamp —
    some datasets (e.g. elliptic_pp) carry no reliable per-entity temporal
    signal in canonical form, and a temporal split there would be
    meaningless rather than merely imprecise.
    """
    if entities["timestamp"].isna().any():
        n_missing = int(entities["timestamp"].isna().sum())
        raise ValueError(
            f"split_temporal requires every entity to have a known timestamp; "
            f"{n_missing}/{len(entities)} entities in dataset {dataset_name!r} do not. "
            "This dataset's canonical timestamps may be unknown for its label entity type."
        )

    ordered = entities.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_train = min(int(round(split_config.train_fraction * n)), n)
    n_val = min(int(round(split_config.val_fraction * n)), n - n_train)

    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]

    manifest = SplitManifest(
        dataset_name=dataset_name,
        strategy="temporal",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        boundaries={
            "train_end": _iso(train["timestamp"].max() if len(train) else None),
            "val_end": _iso(val["timestamp"].max() if len(val) else None),
        },
        class_balance=_class_balance({"train": train, "val": val, "test": test}),
        seed=None,
    )
    return (_ids(train), _ids(val), _ids(test)), manifest


def split_random(
    entities: pd.DataFrame, split_config: SplitConfig, dataset_name: str, seed: int
) -> tuple[SplitArrays, SplitManifest]:
    """Uniformly shuffle and cut at the configured fractions.

    Exists to measure the gap a temporal split closes, never for headline
    numbers — see the project brief's Phase 2.5.
    """
    rng = np.random.default_rng(seed)
    shuffled = entities.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    n = len(shuffled)
    n_train = min(int(round(split_config.train_fraction * n)), n)
    n_val = min(int(round(split_config.val_fraction * n)), n - n_train)

    train = shuffled.iloc[:n_train]
    val = shuffled.iloc[n_train : n_train + n_val]
    test = shuffled.iloc[n_train + n_val :]

    manifest = SplitManifest(
        dataset_name=dataset_name,
        strategy="random",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        boundaries={},
        class_balance=_class_balance({"train": train, "val": val, "test": test}),
        seed=seed,
    )
    return (_ids(train), _ids(val), _ids(test)), manifest


def split_institution(
    entities: pd.DataFrame, split_config: SplitConfig, dataset_name: str
) -> tuple[SplitArrays, SplitManifest]:
    """Hold one institution out entirely for test; split the rest into train/val.

    Raises :class:`ValueError` if ``institution_id`` is unknown for every
    entity (most datasets besides ibm_aml) or if the configured
    ``held_out_institution_id`` matches nothing.
    """
    held_out = split_config.held_out_institution_id
    if held_out is None:
        raise ValueError("split_institution requires split_config.held_out_institution_id")
    if entities["institution_id"].isna().all():
        raise ValueError(
            f"split_institution requires known institution_id values; dataset "
            f"{dataset_name!r} has none for its label entity type."
        )

    test = entities[entities["institution_id"] == held_out]
    if test.empty:
        raise ValueError(
            f"held_out_institution_id {held_out!r} matches no entities in dataset {dataset_name!r}"
        )
    remaining = entities[entities["institution_id"] != held_out].sort_values(
        "entity_id", kind="mergesort"
    )

    denom = split_config.train_fraction + split_config.val_fraction
    train_fraction_within_remaining = split_config.train_fraction / denom
    n_train = int(round(train_fraction_within_remaining * len(remaining)))
    train = remaining.iloc[:n_train]
    val = remaining.iloc[n_train:]

    manifest = SplitManifest(
        dataset_name=dataset_name,
        strategy="institution",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        boundaries={"held_out_institution_id": held_out},
        class_balance=_class_balance({"train": train, "val": val, "test": test}),
        seed=None,
    )
    return (_ids(train), _ids(val), _ids(test)), manifest


def split_dataset(
    dataset: CanonicalDataset, split_config: SplitConfig, seed: int = 0
) -> tuple[SplitArrays, SplitManifest]:
    """Dispatch to the split function named by ``split_config.strategy``."""
    entities = build_entity_frame(dataset)
    if entities.empty:
        raise ValueError(f"dataset {dataset.name!r} has no tx/account labels to split on")

    if split_config.strategy == "temporal":
        return split_temporal(entities, split_config, dataset.name)
    if split_config.strategy == "random":
        return split_random(entities, split_config, dataset.name, seed)
    if split_config.strategy == "institution":
        return split_institution(entities, split_config, dataset.name)
    raise ValueError(f"unknown split strategy {split_config.strategy!r}")  # unreachable


__all__ = [
    "SplitArrays",
    "SplitManifest",
    "build_entity_frame",
    "split_temporal",
    "split_random",
    "split_institution",
    "split_dataset",
]
