"""Phase 8.1: Database connection and session lifecycle.

Provides connection managers and FastAPI dependency handlers for SQLAlchemy.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Base ORM model class
Base = declarative_base()

# DSN Resolution: prioritize environment, fallback to SQLite
DATABASE_DSN = os.environ.get("LEDGER_DATABASE__DSN") or "sqlite:///ledger.db"

# Create engine (SQLite requires check_same_thread=False for FastAPI concurrency)
connect_args = {}
if DATABASE_DSN.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_DSN, connect_args=connect_args, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI route dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Helper function to create all tables for SQLite/dev setups."""
    import ledger.db.models
    Base.metadata.create_all(bind=engine)
