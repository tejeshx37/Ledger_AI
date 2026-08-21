"""Feature engineering pipeline: fit on train, apply unchanged everywhere else.

Combines raw dataset attributes (numeric fields parsed out of
``accounts.attributes`` JSON), graph-topological, temporal, and motif
features into one matrix per account. The only part of this that is
statistically "fit" is the scaler (``config.scaler``); everything else is
a deterministic function of observed graph structure. Fitting on any
account outside the training split is leakage — :meth:`FeaturePipeline.fit`
takes the held-out ids explicitly and raises :class:`FeatureLeakageError`
if they overlap the training ids, and :meth:`FeaturePipeline.transform`
raises :class:`FeaturePipelineNotFittedError` if called before ``fit``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ledger.config.models import FeaturesConfig
from ledger.data.canonical import CanonicalDataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.features.graph_features import compute_graph_topological_features
from ledger.features.motif_features import compute_motif_features
from ledger.features.temporal_features import compute_temporal_features


class FeaturePipelineNotFittedError(RuntimeError):
    """Raised when :meth:`FeaturePipeline.transform` is called before ``fit``."""


class FeatureLeakageError(ValueError):
    """Raised when ``fit`` is given train ids that overlap held-out ids."""


def _raw_attribute_features(dataset: CanonicalDataset) -> pd.DataFrame:
    """Numeric-only features parsed out of ``accounts.attributes`` JSON.

    Non-numeric raw fields (device strings, email domains, and the like)
    are intentionally excluded here: they are wildly heterogeneous across
    the six registered datasets, and categorical encoding is a modelling
    concern for Phase 3+, not a general-purpose feature-engineering one.
    """
    records = [json.loads(a) for a in dataset.accounts["attributes"]]
    raw = pd.json_normalize(records)
    numeric_cols = raw.select_dtypes(include="number").columns
    raw = raw[numeric_cols].add_prefix("raw_")
    raw.insert(0, "account_id", dataset.accounts["account_id"].to_numpy())
    return raw


def _build_feature_matrix(
    dataset: CanonicalDataset, graph_builder: TemporalGraphBuilder, config: FeaturesConfig
) -> pd.DataFrame:
    matrix = pd.DataFrame({"account_id": dataset.accounts["account_id"]})
    matrix = matrix.merge(_raw_attribute_features(dataset), on="account_id", how="left")

    if config.use_graph_topological:
        graph_feats = compute_graph_topological_features(graph_builder.build_static_graph())
        matrix = matrix.merge(graph_feats, on="account_id", how="left")

    if config.use_temporal:
        temporal_feats = compute_temporal_features(dataset, config)
        matrix = matrix.merge(temporal_feats, on="account_id", how="left")

    if config.use_motif_counts:
        motif_feats = compute_motif_features(graph_builder.build_static_graph(), config)
        matrix = matrix.merge(motif_feats, on="account_id", how="left")

    feature_cols = [c for c in matrix.columns if c != "account_id"]
    matrix[feature_cols] = matrix[feature_cols].fillna(0.0)
    return matrix


def _make_scaler(name: str) -> Any:
    if name == "none":
        return None
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    factories = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    return factories[name]()


@dataclass
class FeaturePipeline:
    """Fits a scaler on train-only feature statistics, then applies the
    identical transformation (feature computation + fitted scaler) to any
    account set — test, serving requests, or otherwise.
    """

    config: FeaturesConfig
    _scaler: Any = field(default=None, repr=False)
    _feature_columns: list[str] | None = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        dataset: CanonicalDataset,
        train_account_ids: list[str] | pd.Index | pd.Series,
        held_out_account_ids: list[str] | pd.Index | pd.Series | None = None,
    ) -> FeaturePipeline:
        """Fit the scaler on ``train_account_ids`` only.

        If ``held_out_account_ids`` is given (typically the val/test split),
        any overlap with ``train_account_ids`` raises :class:`FeatureLeakageError`
        instead of silently fitting on held-out data.
        """
        train_ids = set(train_account_ids)
        if held_out_account_ids is not None:
            overlap = train_ids & set(held_out_account_ids)
            if overlap:
                raise FeatureLeakageError(
                    f"fit() received {len(overlap)} account id(s) present in both "
                    "train_account_ids and held_out_account_ids — fitting on "
                    f"held-out data is leakage. First few: {sorted(overlap)[:5]}"
                )

        graph_builder = TemporalGraphBuilder(dataset)
        full_matrix = _build_feature_matrix(dataset, graph_builder, self.config)
        train_matrix = full_matrix[full_matrix["account_id"].isin(train_ids)]
        if train_matrix.empty:
            raise ValueError("fit() found no rows for the given train_account_ids")

        self._feature_columns = [c for c in full_matrix.columns if c != "account_id"]
        self._scaler = _make_scaler(self.config.scaler)
        if self._scaler is not None:
            self._scaler.fit(train_matrix[self._feature_columns].to_numpy())
        self._is_fitted = True
        return self

    def transform(
        self,
        dataset: CanonicalDataset,
        account_ids: list[str] | pd.Index | pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply the fitted feature computation (and scaler, if any) to
        ``account_ids`` (or every account in ``dataset`` if omitted).
        """
        if not self._is_fitted or self._feature_columns is None:
            raise FeaturePipelineNotFittedError(
                "FeaturePipeline.transform() called before fit(). Fit on the "
                "training split before transforming anything, including test "
                "and serving requests."
            )

        graph_builder = TemporalGraphBuilder(dataset)
        matrix = _build_feature_matrix(dataset, graph_builder, self.config)

        # A fitted pipeline always emits exactly the columns it was fitted
        # on, in the same order, even if this dataset lacks some of them
        # (e.g. a raw attribute absent at serving time).
        for col in self._feature_columns:
            if col not in matrix.columns:
                matrix[col] = 0.0
        matrix = matrix[["account_id", *self._feature_columns]]

        if account_ids is not None:
            wanted = set(account_ids)
            matrix = matrix[matrix["account_id"].isin(wanted)].reset_index(drop=True)

        if self._scaler is not None:
            scaled = self._scaler.transform(matrix[self._feature_columns].to_numpy())
            matrix = matrix.copy()
            matrix[self._feature_columns] = scaled

        return matrix

    def save(self, path: Path | str) -> Path:
        """Persist the fitted pipeline (including scaler state) via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> FeaturePipeline:
        """Load a pipeline previously written by :meth:`save`."""
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return loaded


__all__ = ["FeaturePipeline", "FeaturePipelineNotFittedError", "FeatureLeakageError"]
