"""Shared fixtures for Phase 3/4 (model) tests: a small labelled split plus
an attributed graph, built once per test from the root conftest's
``elliptic_raw_dir`` fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import networkx as nx
import pandas as pd
import pytest

from ledger.config.models import FeaturesConfig, SplitConfig
from ledger.data.adapters import get_adapter
from ledger.data.canonical import CanonicalDataset
from ledger.data.graph import TemporalGraphBuilder
from ledger.data.splitting import split_dataset
from ledger.evaluation.labels import build_binary_target
from ledger.features.pipeline import FeaturePipeline
from ledger.models.graph_common import attach_node_features


class GraphFixture(NamedTuple):
    canonical: CanonicalDataset
    features_config: FeaturesConfig
    pipeline: FeaturePipeline
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    attributed_graph: nx.MultiDiGraph[str]


@pytest.fixture
def elliptic_graph_fixture(elliptic_raw_dir: Path) -> GraphFixture:
    canonical = get_adapter("elliptic").run(elliptic_raw_dir)
    (train_ids, val_ids, test_ids), _ = split_dataset(canonical, SplitConfig(strategy="temporal"))

    y_all = build_binary_target(canonical)
    y_index = set(y_all.index)
    train_ids = [i for i in train_ids if i in y_index]
    val_ids = [i for i in val_ids if i in y_index]
    test_ids = [i for i in test_ids if i in y_index]

    features_config = FeaturesConfig(
        use_graph_topological=True, use_temporal=True, use_motif_counts=True
    )
    pipeline = FeaturePipeline(config=features_config)
    pipeline.fit(canonical, train_account_ids=train_ids, held_out_account_ids=val_ids + test_ids)

    X_train = pipeline.transform(canonical, account_ids=train_ids).set_index("account_id")
    X_test = pipeline.transform(canonical, account_ids=test_ids).set_index("account_id")
    y_train = y_all.loc[X_train.index]
    y_test = y_all.loc[X_test.index]

    full_features = pipeline.transform(canonical).set_index("account_id")
    full_graph = TemporalGraphBuilder(canonical).build_static_graph()
    attributed_graph = attach_node_features(full_graph, full_features)

    return GraphFixture(
        canonical=canonical,
        features_config=features_config,
        pipeline=pipeline,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        attributed_graph=attributed_graph,
    )
