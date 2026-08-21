"""Top-level LEDGER settings aggregate and layered loader.

Load order (each layer overrides the previous one field-by-field):

    configs/base.yaml -> configs/<environment>.yaml -> environment
    variables (prefix ``LEDGER__``, ``__`` as nesting delimiter) -> explicit
    CLI overrides passed as a dict.

The resulting :class:`Settings` object is frozen: nothing downstream can
mutate it after load, so a run manifest's ``config_hash`` is guaranteed to
describe the configuration that actually ran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from ledger.config.models import (
    DatabaseConfig,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    FederatedConfig,
    LLMConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    PrivacyConfig,
    ServingConfig,
    TrainingConfig,
)


class Settings(BaseSettings):
    """Root configuration object, composed of one model per domain.

    Instances are frozen after construction (see ``model_config``). Use
    :func:`load_settings` to build one via the full layered load order
    rather than instantiating this directly, so that YAML and environment
    layers are applied consistently.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="LEDGER__",
        env_nested_delimiter="__",
    )

    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    federated: FederatedConfig = FederatedConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    serving: ServingConfig = ServingConfig()
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()

    def hash(self) -> str:
        """Return a stable SHA-256 hex digest of the resolved configuration.

        Used by :class:`ledger.utils.manifest.RunManifest` so every
        experiment run can be traced back to the exact configuration that
        produced it. Stability across process runs requires
        ``sort_keys=True`` and a fixed JSON serialisation of ``Path``
        objects (via ``default=str``).
        """
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_dir: Path | str = Path("configs"),
    environment: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a :class:`Settings` instance via the full layered load order.

    Args:
        config_dir: directory containing ``base.yaml`` and optional
            environment-specific YAML files.
        environment: name of an environment-specific YAML overlay
            (``<config_dir>/<environment>.yaml``); skipped if ``None`` or
            the file does not exist.
        cli_overrides: nested dict of explicit overrides, applied last and
            therefore taking precedence over everything else including
            environment variables.

    Returns:
        A fully validated, frozen ``Settings`` instance.
    """
    config_dir = Path(config_dir)
    merged: dict[str, Any] = _read_yaml(config_dir / "base.yaml")

    if environment is not None:
        merged = _deep_merge(merged, _read_yaml(config_dir / f"{environment}.yaml"))

    # Environment variables are layered on top via pydantic-settings' own
    # resolution, which we trigger by constructing Settings with the merged
    # YAML as init-provided defaults: pydantic-settings gives env vars
    # priority over values passed to __init__.
    settings = Settings(**merged)

    if cli_overrides:
        merged_dump = _deep_merge(settings.model_dump(mode="json"), cli_overrides)
        settings = Settings(**merged_dump)

    return settings


def load_settings_from_overlay_path(
    overlay_path: Path | str, cli_overrides: dict[str, Any] | None = None
) -> Settings:
    """Load settings using a specific overlay YAML file's directory as
    ``config_dir`` and its stem as the ``environment`` name.

    This is what every ``--config <path>`` CLI flag (``ledger train``,
    ``ledger federated run``, ...) resolves to: ``--config
    configs/model_baseline.yaml`` loads ``configs/base.yaml`` with
    ``configs/model_baseline.yaml`` layered on top, exactly like passing
    ``config_dir="configs", environment="model_baseline"`` to
    :func:`load_settings` directly.
    """
    overlay_path = Path(overlay_path)
    return load_settings(
        config_dir=overlay_path.parent,
        environment=overlay_path.stem,
        cli_overrides=cli_overrides,
    )


__all__ = ["Settings", "load_settings"]
