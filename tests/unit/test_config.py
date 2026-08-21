"""Unit tests for the config domain models and layered loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ledger.config.models import DataConfig, PrivacyConfig, SplitConfig
from ledger.config.settings import Settings, load_settings


def test_settings_loads_with_defaults() -> None:
    settings = Settings()
    assert settings.model.name == "xgboost_baseline"
    assert settings.data.dataset_name == "elliptic"


def test_settings_is_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.model.name = "graphsage"  # type: ignore[misc]


def test_settings_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        Settings(unexpected_top_level_field=1)  # type: ignore[call-arg]


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(train_fraction=0.5, val_fraction=0.3, test_fraction=0.3)


def test_split_strategy_must_be_known() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy="bogus")


def test_institution_split_requires_held_out_id() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(strategy="institution")
    # valid when the id is supplied
    cfg = SplitConfig(strategy="institution", held_out_institution_id="bank_7")
    assert cfg.held_out_institution_id == "bank_7"


def test_dataset_name_must_be_registered() -> None:
    with pytest.raises(ValidationError):
        DataConfig(dataset_name="not_a_real_dataset")


def test_privacy_epsilon_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PrivacyConfig(target_epsilon=0.0)


def test_hash_is_stable_for_identical_config() -> None:
    a = Settings()
    b = Settings()
    assert a.hash() == b.hash()


def test_hash_changes_when_config_changes() -> None:
    a = Settings()
    b = Settings(model={"name": "graphsage"})
    assert a.hash() != b.hash()


def test_load_settings_reads_base_yaml(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "model:\n  name: graphsage\ndata:\n  dataset_name: ibm_aml\n"
    )
    settings = load_settings(config_dir=tmp_path)
    assert settings.model.name == "graphsage"
    assert settings.data.dataset_name == "ibm_aml"


def test_load_settings_environment_overlay_wins(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  name: xgboost_baseline\n")
    (tmp_path / "prod.yaml").write_text("model:\n  name: tgat\n")
    settings = load_settings(config_dir=tmp_path, environment="prod")
    assert settings.model.name == "tgat"


def test_load_settings_missing_environment_overlay_is_noop(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  name: graphsage\n")
    settings = load_settings(config_dir=tmp_path, environment="does_not_exist")
    assert settings.model.name == "graphsage"


def test_load_settings_cli_overrides_win_over_everything(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  name: xgboost_baseline\n")
    settings = load_settings(config_dir=tmp_path, cli_overrides={"model": {"name": "tgat"}})
    assert settings.model.name == "tgat"


def test_load_settings_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_settings(config_dir=tmp_path)


def test_repo_base_yaml_loads_cleanly() -> None:
    """The actual configs/base.yaml shipped in the repo must load without error."""
    settings = load_settings(config_dir=Path("configs"))
    assert isinstance(settings, Settings)
