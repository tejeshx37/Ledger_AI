"""Phase 8 integration tests for Database ORM, FastAPI REST endpoints, and Async Worker.

Covers RBAC validation, batch transaction ingestion, idempotency caching,
alert queue filtering, review submissions, and worker scoring loops.
"""

from __future__ import annotations

import json
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ledger.api.auth import create_access_token
from ledger.api.main import app
from ledger.api.middleware import IDEMPOTENCY_CACHE
from ledger.api.worker import process_scoring_job
from ledger.db.connection import Base, get_db
from ledger.db.models import Account, Alert, AuditLog, Model, Narrative, Review, Run, Transaction

# Create SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///test_ledger.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Fixture yielding a clean database session per test."""
    import ledger.db.models
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        import os
        from pathlib import Path
        db_file = Path("test_ledger.db")
        if db_file.exists():
            try:
                db_file.unlink()
            except Exception:
                pass


@pytest.fixture(scope="function")
def client(db_session):
    """Fixture yielding a FastAPI TestClient bound to the clean database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    # Clear idempotency cache between tests
    IDEMPOTENCY_CACHE.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_ready_metrics(client) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}

    res = client.get("/metrics")
    assert res.status_code == 200
    assert "ledger_api_requests_total" in res.text


def test_auth_token_roles(client) -> None:
    # Login as analyst
    res = client.post("/v1/token", data={"username": "analyst", "password": "password"})
    assert res.status_code == 200
    token_data = res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

    # Login with invalid role username
    res = client.post("/v1/token", data={"username": "bad_role", "password": "password"})
    assert res.status_code == 401


def test_rbac_and_ingestion(client) -> None:
    # 1. Accessing transactions without token should fail
    payload = {"transactions": [
        {"tx_id": "tx_api_1", "src_account": "A", "dst_account": "B", "amount": 1000.0, "timestamp": "2022-09-01 12:00:00"}
    ]}
    res = client.post("/v1/transactions", json=payload)
    assert res.status_code == 401

    # 2. Accessing transactions with analyst token should succeed
    analyst_token = create_access_token(data={"sub": "test_analyst", "role": "analyst"})
    headers = {"Authorization": f"Bearer {analyst_token}", "X-Idempotency-Key": "key_1"}
    
    res = client.post("/v1/transactions", json=payload, headers=headers)
    assert res.status_code == 201
    assert res.json()["status"] == "success"
    assert res.json()["count"] == 1

    # 3. Test Idempotency key replay returns cached 201 response directly
    res_replay = client.post("/v1/transactions", json=payload, headers=headers)
    assert res_replay.status_code == 201
    assert res_replay.json()["status"] == "success"


def test_alerts_reviews_audit_trail(client, db_session) -> None:
    # Insert test data into SQLite
    acc = Account(account_id="A1", bank_id="bank_test")
    run = Run(run_id="run_test", dataset_name="ibm_aml", config_hash="h1")
    model = Model(model_version="v1.0", run_id="run_test", architecture="graphsage")
    
    db_session.add(acc)
    db_session.add(run)
    db_session.add(model)
    db_session.commit()

    alert = Alert(
        alert_id="alert_1",
        account_id="A1",
        confidence_score=0.85,
        status="open",
        model_version="v1.0",
        run_id="run_test",
    )
    db_session.add(alert)
    db_session.commit()

    # Get tokens
    analyst_token = create_access_token(data={"sub": "john_analyst", "role": "analyst"})
    supervisor_token = create_access_token(data={"sub": "alice_supervisor", "role": "supervisor"})

    # 1. Analyst tries to confirm alert -> Forbidden (403)
    res = client.post(
        "/v1/alerts/alert_1/review",
        json={"decision": "confirm", "reason": "Looks very suspicious"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert res.status_code == 403

    # 2. Supervisor confirms alert -> Success (200)
    res = client.post(
        "/v1/alerts/alert_1/review",
        json={"decision": "confirm", "reason": "Verified money laundering cycle"},
        headers={"Authorization": f"Bearer {supervisor_token}"},
    )
    assert res.status_code == 200
    assert res.json()["new_status"] == "confirmed"

    # 3. Check review and audit log records exist in DB
    reviews = db_session.query(Review).all()
    assert len(reviews) == 1
    assert reviews[0].decision == "confirm"
    assert reviews[0].reviewer == "alice_supervisor"

    audit_logs = db_session.query(AuditLog).all()
    assert len(audit_logs) == 1
    assert audit_logs[0].entity_type == "alert"
    assert audit_logs[0].entity_id == "alert_1"
    assert audit_logs[0].action == "update_status"
    assert audit_logs[0].actor == "alice_supervisor"


def test_worker_scoring_job(db_session) -> None:
    # Setup db records
    run = Run(run_id="test_explain_run", dataset_name="ibm_aml", config_hash="h1")
    model = Model(model_version="v1.0", run_id="test_explain_run", architecture="graphsage")
    acc = Account(account_id="1_A2", bank_id="bank_test")
    db_session.add(run)
    db_session.add(model)
    db_session.add(acc)
    db_session.commit()

    # Override SessionLocal in worker with TestingSessionLocal
    import ledger.api.worker
    original_session_local = ledger.api.worker.SessionLocal
    ledger.api.worker.SessionLocal = TestingSessionLocal

    try:
        # Run scoring job worker task
        res = process_scoring_job(
            job_id="job_worker_test",
            account_ids=["1_A2"],
            model_version="v1.0",
            alert_threshold=0.3,
        )
        assert res["status"] == "success"

        # Check alert generated
        alerts = db_session.query(Alert).filter_by(account_id="1_A2").all()
        assert len(alerts) == 1
        assert alerts[0].status == "open"

        # Check narrative exists
        narrative = db_session.query(Narrative).filter_by(alert_id=alerts[0].alert_id).first()
        assert narrative is not None
        assert "Alert" in narrative.narrative_text
    finally:
        ledger.api.worker.SessionLocal = original_session_local
