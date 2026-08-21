"""Phase 8.1: SQLAlchemy 2.0 database ORM models.

Defines schemas for accounts, transactions, models, runs, alerts, evidence,
narratives, reviews, and audit logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from ledger.db.connection import Base


class Account(Base):
    """Represents a bank account node."""

    __tablename__ = "accounts"

    account_id = Column(String, primary_key=True, index=True)
    bank_id = Column(String, nullable=False)
    features = Column(JSON, nullable=True)  # Store feature representation
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    alerts = relationship("Alert", back_populates="account")


class Transaction(Base):
    """Represents a transaction edge between accounts."""

    __tablename__ = "transactions"

    tx_id = Column(String, primary_key=True, index=True)
    src_account = Column(String, ForeignKey("accounts.account_id"), nullable=False)
    dst_account = Column(String, ForeignKey("accounts.account_id"), nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Run(Base):
    """Represents an experiment or training run."""

    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, index=True)
    dataset_name = Column(String, nullable=False)
    config_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("Alert", back_populates="run")
    models = relationship("Model", back_populates="run")


class Model(Base):
    """Represents a registered model version."""

    __tablename__ = "models"

    model_version = Column(String, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    architecture = Column(String, nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="models")
    alerts = relationship("Alert", back_populates="model")


class Alert(Base):
    """Represents a suspicious activity alert."""

    __tablename__ = "alerts"

    alert_id = Column(String, primary_key=True, index=True)
    account_id = Column(String, ForeignKey("accounts.account_id"), nullable=False)
    confidence_score = Column(Float, nullable=False)
    status = Column(String, default="open")  # "open", "confirmed", "dismissed"
    model_version = Column(String, ForeignKey("models.model_version"), nullable=False)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="alerts")
    model = relationship("Model", back_populates="alerts")
    run = relationship("Run", back_populates="alerts")
    evidence = relationship("Evidence", back_populates="alert", uselist=False)
    narrative = relationship("Narrative", back_populates="alert", uselist=False)
    reviews = relationship("Review", back_populates="alert")


class Evidence(Base):
    """Stores the extracted subgraph evidence for an alert."""

    __tablename__ = "evidence"

    alert_id = Column(String, ForeignKey("alerts.alert_id"), primary_key=True)
    subgraph = Column(JSON, nullable=False)  # JSON representation of evidence
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("Alert", back_populates="evidence")


class Narrative(Base):
    """Stores narrative explanations for an alert."""

    __tablename__ = "narratives"

    alert_id = Column(String, ForeignKey("alerts.alert_id"), primary_key=True)
    narrative_text = Column(Text, nullable=False)
    narrative_type = Column(String, default="deterministic")  # "deterministic" or "llm"
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("Alert", back_populates="narrative")


class Review(Base):
    """Analyst decisions (confirm/dismiss) with justifications."""

    __tablename__ = "reviews"

    review_id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String, ForeignKey("alerts.alert_id"), nullable=False)
    decision = Column(String, nullable=False)  # "confirm" or "dismiss"
    reason = Column(Text, nullable=False)
    reviewer = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("Alert", back_populates="reviews")


class AuditLog(Base):
    """Append-only audit log tracing all database state changes."""

    __tablename__ = "audit_log"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String, nullable=False)  # e.g., "alert", "review"
    entity_id = Column(String, nullable=False)
    action = Column(String, nullable=False)  # e.g., "create", "update_status", "review"
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    actor = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
