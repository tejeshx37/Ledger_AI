"""Typed configuration domains for LEDGER.

Every tunable used anywhere in the system — alert thresholds, DP noise
multipliers, learning rates, split boundaries, ring-size minimums — is a
field on one of these frozen Pydantic models. Nothing here has a "just for
now" magic number in application code: if a value governs behaviour, it is
named and documented here, with a validator that rejects nonsense.

Models are frozen (immutable) once constructed so that a run's configuration
cannot drift after ``Settings.hash()`` has been computed for its manifest.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    """Base class for all config domains: immutable and forbids unknown keys.

    ``extra="forbid"`` exists specifically to catch typos in YAML — a
    misspelled key silently ignored is a defect in a compliance system.
    """

    model_config = {"frozen": True, "extra": "forbid"}


class PathsConfig(FrozenModel):
    """Filesystem locations. No path is ever written literally in ``src/``."""

    project_root: Path = Field(default=Path("."))
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    runs_dir: Path = Field(default=Path("runs"))
    models_dir: Path = Field(default=Path("models"))


class SplitConfig(FrozenModel):
    """Train/validation/test split policy.

    ``temporal`` is the only strategy used for headline metrics: deployment
    only ever has the past, and a random split leaks future graph structure
    into training. ``random`` exists purely to measure and report that
    leakage gap. ``institution`` holds out an entire bank for the federated
    experiment.
    """

    strategy: str = Field(default="temporal")
    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    held_out_institution_id: str | None = Field(default=None)

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {"temporal", "random", "institution"}
        if v not in allowed:
            raise ValueError(f"split.strategy must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")
        return self

    @model_validator(mode="after")
    def _institution_requires_id(self) -> SplitConfig:
        if self.strategy == "institution" and self.held_out_institution_id is None:
            raise ValueError(
                "split.held_out_institution_id is required when strategy='institution'"
            )
        return self


class DataConfig(FrozenModel):
    """Dataset selection, canonical-schema policy, and split configuration."""

    dataset_name: str = Field(default="elliptic")
    min_ring_size: int = Field(default=3, ge=2)
    unknown_label_value: str = Field(default="unknown")
    checksum_algorithm: str = Field(default="sha256")
    split: SplitConfig = Field(default_factory=SplitConfig)

    @field_validator("dataset_name")
    @classmethod
    def _validate_dataset_name(cls, v: str) -> str:
        allowed = {"elliptic", "elliptic_pp", "ibm_aml", "paysim", "baf", "ieee_cis"}
        if v not in allowed:
            raise ValueError(f"data.dataset_name must be one of {allowed}, got {v!r}")
        return v


class FeaturesConfig(FrozenModel):
    """Feature engineering knobs: which feature families to compute."""

    use_graph_topological: bool = Field(default=True)
    use_temporal: bool = Field(default=True)
    use_motif_counts: bool = Field(default=True)
    rolling_window_steps: int = Field(default=5, ge=1)
    fan_in_threshold: int = Field(default=6, ge=1)
    fan_out_threshold: int = Field(default=6, ge=1)
    scaler: str = Field(default="standard")

    @field_validator("scaler")
    @classmethod
    def _validate_scaler(cls, v: str) -> str:
        allowed = {"standard", "robust", "minmax", "none"}
        if v not in allowed:
            raise ValueError(f"features.scaler must be one of {allowed}, got {v!r}")
        return v


class ModelConfig(FrozenModel):
    """Model architecture selection and hyperparameters.

    Holds hyperparameters for whichever architecture ``name`` selects.
    Fields irrelevant to a given architecture are simply unused, never
    silently defaulted deep inside model code.
    """

    name: str = Field(default="xgboost_baseline")
    random_seed: int = Field(default=42)
    hidden_dim: int = Field(default=64, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    fanout_per_layer: list[int] = Field(default_factory=lambda: [15, 10])
    time_encoding_dim: int = Field(default=16, ge=1)
    motif_loss_weight: float = Field(default=0.0, ge=0.0)
    xgboost_max_depth: int = Field(default=6, ge=1)
    xgboost_n_estimators: int = Field(default=300, ge=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        allowed = {"xgboost_baseline", "graphsage", "tgat"}
        if v not in allowed:
            raise ValueError(f"model.name must be one of {allowed}, got {v!r}")
        return v


class TrainingConfig(FrozenModel):
    """Optimisation loop configuration."""

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    batch_size: int = Field(default=512, ge=1)
    max_epochs: int = Field(default=50, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    alert_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    alert_budget_per_period: int | None = Field(default=None, ge=1)


class FederatedConfig(FrozenModel):
    """Federated learning protocol configuration."""

    num_clients: int = Field(default=4, ge=2)
    clients_per_round_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    num_rounds: int = Field(default=20, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    strategy: str = Field(default="fedavg")
    fedprox_mu: float = Field(default=0.01, ge=0.0)
    boundary_exchange_every_n_rounds: int = Field(default=1, ge=1)
    boundary_hash_key_env_var: str = Field(default="LEDGER_BOUNDARY_HASH_KEY")
    secure_aggregation: bool = Field(default=False)

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        allowed = {"fedavg", "fedprox"}
        if v not in allowed:
            raise ValueError(f"federated.strategy must be one of {allowed}, got {v!r}")
        return v


class PrivacyConfig(FrozenModel):
    """Differential privacy budget and mechanism parameters."""

    dp_enabled: bool = Field(default=True)
    target_epsilon: float = Field(default=8.0, gt=0.0)
    target_delta: float = Field(default=1e-5, gt=0.0, lt=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    noise_multiplier: float = Field(default=1.1, gt=0.0)

    @model_validator(mode="after")
    def _delta_below_inverse_dataset_scale(self) -> PrivacyConfig:
        if self.target_delta >= 1.0:
            raise ValueError("privacy.target_delta must be < 1.0")
        return self


class EvaluationConfig(FrozenModel):
    """Evaluation harness and fairness-audit configuration."""

    precision_at_k_values: list[int] = Field(default_factory=lambda: [50, 100, 500])
    protected_attribute_columns: list[str] = Field(default_factory=list)
    fairness_disparity_tolerance: float = Field(default=0.1, gt=0.0, lt=1.0)
    evasion_effort_levels: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    threshold_sweep_steps: int = Field(default=19, ge=2)


class ServingConfig(FrozenModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    rate_limit_per_minute: int = Field(default=120, ge=1)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)


class DatabaseConfig(FrozenModel):
    """PostgreSQL connection configuration.

    ``dsn`` is intentionally left unset by default: it must come from the
    ``LEDGER_DATABASE__DSN`` environment variable or a deployment-specific
    YAML overlay, never a literal in source.
    """

    dsn: str | None = Field(default=None)
    pool_size: int = Field(default=5, ge=1)
    pool_max_overflow: int = Field(default=10, ge=0)


class LLMConfig(FrozenModel):
    """Optional LLM narrative-refinement layer.

    The system must run correctly with ``provider="null"`` (the default):
    no LLM configured, deterministic narratives only.
    """

    provider: str = Field(default="null")
    model_name: str = Field(default="")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1)
    prompt_template: str = Field(
        default="Refine this raw structured alert evidence into a professional, report-ready prose narrative. Strict constraint: Do not invent any names, dates, amounts, accounts, or facts not explicitly listed in the evidence.\n\nEvidence:\n{evidence_json}"
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"null", "anthropic", "openai", "gemini"}
        if v not in allowed:
            raise ValueError(f"llm.provider must be one of {allowed}, got {v!r}")
        return v


class LoggingConfig(FrozenModel):
    """Structured logging configuration."""

    level: str = Field(default="INFO")
    json_format: bool = Field(default=True)

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got {v!r}")
        return upper
