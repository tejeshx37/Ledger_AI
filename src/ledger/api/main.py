"""Phase 8.2: FastAPI Web Application.

Defines all routes, request/response validation schemas, and application startup.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import structlog

from ledger.api.auth import (
    RoleChecker,
    create_access_token,
    get_current_user,
)
from ledger.api.middleware import (
    IDEMPOTENCY_CACHE,
    cache_idempotency_response,
    check_idempotency,
    rate_limiter,
)
from ledger.db.connection import get_db, init_db
from ledger.db.models import Account, Alert, AuditLog, Evidence, Model, Narrative, Review, Run, Transaction

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="LEDGER AML Compliance Platform API",
    version="1.0.0",
    dependencies=[Depends(rate_limiter)],
)


# === Pydantic Validation Schemas ===

class Token(BaseModel):
    access_token: str
    token_type: str


class TransactionIngest(BaseModel):
    tx_id: str = Field(..., examples=["tx_999"])
    src_account: str = Field(..., examples=["1_A1"])
    dst_account: str = Field(..., examples=["1_B1"])
    amount: float = Field(..., gt=0.0, examples=[25000.0])
    timestamp: str = Field(..., examples=["2022-09-01T12:00:00"])


class TransactionBatch(BaseModel):
    transactions: List[TransactionIngest]


class ScoreRequest(BaseModel):
    account_ids: List[str]
    model_version: str


class ScoreResponse(BaseModel):
    job_id: str
    status: str


class AlertSummary(BaseModel):
    alert_id: str
    account_id: str
    confidence_score: float
    status: str
    model_version: str
    created_at: datetime


class AlertDetail(BaseModel):
    alert_id: str
    account_id: str
    confidence_score: float
    status: str
    model_version: str
    run_id: str
    created_at: datetime
    evidence: Optional[Any] = None
    narrative: Optional[str] = None


class ReviewSubmit(BaseModel):
    decision: str = Field(..., description="Must be 'confirm' or 'dismiss'")
    reason: str = Field(..., min_length=5)


# === REST Endpoints ===

@app.on_event("startup")
def startup_event():
    """Ensure database schema is created on startup."""
    init_db()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    # Simulated Prometheus format metrics
    lines = [
        "# HELP ledger_api_requests_total Total number of requests",
        "# TYPE ledger_api_requests_total counter",
        "ledger_api_requests_total 42",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")


@app.post("/v1/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Simulated login mapping username to matching role (e.g. username 'admin' gets role 'admin')."""
    role = form_data.username
    if role not in ["analyst", "supervisor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid role. Must login with username 'analyst', 'supervisor', or 'admin'.",
        )
    access_token = create_access_token(data={"sub": form_data.username, "role": role})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/v1/transactions", status_code=status.HTTP_201_CREATED)
def ingest_transactions(
    batch: TransactionBatch,
    x_idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "admin"])),
):
    """Batch ingest transactions with validation, persistence, and idempotency cache check."""
    if x_idempotency_key and x_idempotency_key in IDEMPOTENCY_CACHE:
        # Cache hit: Return stored response
        status_code, body = IDEMPOTENCY_CACHE[x_idempotency_key]
        return Response(content=body, media_type="application/json", status_code=status_code)

    try:
        # Ingest account placeholders if not present
        accounts_to_add = set()
        for tx in batch.transactions:
            accounts_to_add.add(tx.src_account)
            accounts_to_add.add(tx.dst_account)

        for acc_id in accounts_to_add:
            exists = db.query(Account).filter_by(account_id=acc_id).first()
            if not exists:
                db.add(Account(account_id=acc_id, bank_id="bank_imported"))

        # Ingest transactions
        for tx in batch.transactions:
            # Check if tx already exists
            exists = db.query(Transaction).filter_by(tx_id=tx.tx_id).first()
            if not exists:
                db.add(
                    Transaction(
                        tx_id=tx.tx_id,
                        src_account=tx.src_account,
                        dst_account=tx.dst_account,
                        amount=tx.amount,
                        timestamp=pd.to_datetime(tx.timestamp).to_pydatetime()
                        if "pd" in globals()
                        else datetime.strptime(tx.timestamp.replace("T", " "), "%Y-%m-%d %H:%M:%S")
                        if "T" in tx.timestamp
                        else datetime.strptime(tx.timestamp, "%Y-%m-%d %H:%M:%S"),
                    )
                )

        db.commit()
        response_payload = {"status": "success", "count": len(batch.transactions)}
        response_body = json.dumps(response_payload).encode("utf-8")

        if x_idempotency_key:
            cache_idempotency_response(x_idempotency_key, 201, response_body)

        return response_payload

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ingestion failed: {exc}"
        )


@app.post("/v1/score", response_model=ScoreResponse)
def score_subgraph(
    req: ScoreRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "supervisor", "admin"])),
):
    """Enqueues async scoring task and returns Uuid job token."""
    # Verify model exists
    model = db.query(Model).filter_by(model_version=req.model_version).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{req.model_version}' not found in database.",
        )

    # In a real system, we'd enqueue to Celery/arq.
    # Here, we simulate by returning a deterministic job ID.
    import hashlib
    job_id = f"job_{hashlib.md5(f'{req.model_version}_{len(req.account_ids)}'.encode()).hexdigest()[:8]}"
    return {"job_id": job_id, "status": "pending"}


@app.get("/v1/alerts", response_model=List[AlertSummary])
def list_alerts(
    min_score: float = 0.0,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "supervisor", "admin"])),
):
    """Paginated, filterable alerts queue."""
    query = db.query(Alert).filter(Alert.confidence_score >= min_score)
    if status_filter:
        query = query.filter_by(status=status_filter)
    alerts = query.order_by(Alert.confidence_score.desc()).offset(offset).limit(limit).all()
    return alerts


@app.get("/v1/alerts/{id}", response_model=AlertDetail)
def get_alert_detail(
    id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "supervisor", "admin"])),
):
    """Alert detail with evidence subgraphs and narratives."""
    alert = db.query(Alert).filter_by(alert_id=id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{id}' not found."
        )

    evidence_data = alert.evidence.subgraph if alert.evidence else None
    narrative_text = alert.narrative.narrative_text if alert.narrative else None

    return {
        "alert_id": alert.alert_id,
        "account_id": alert.account_id,
        "confidence_score": alert.confidence_score,
        "status": alert.status,
        "model_version": alert.model_version,
        "run_id": alert.run_id,
        "created_at": alert.created_at,
        "evidence": evidence_data,
        "narrative": narrative_text,
    }


@app.post("/v1/alerts/{id}/review")
def review_alert(
    id: str,
    review: ReviewSubmit,
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "supervisor", "admin"])),
):
    """Record analyst review decisions with audit logging."""
    alert = db.query(Alert).filter_by(alert_id=id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{id}' not found."
        )

    old_status = alert.status
    new_status = "confirmed" if review.decision == "confirm" else "dismissed"

    # Enforce supervisor role for supervisor operations if needed
    if review.decision == "confirm" and user["role"] not in ["supervisor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only supervisors can confirm alerts as true positives.",
        )

    # 1. Update Alert Status
    alert.status = new_status

    # 2. Add Review Row
    rev = Review(
        alert_id=id,
        decision=review.decision,
        reason=review.reason,
        reviewer=user["username"],
    )
    db.add(rev)

    # 3. Add Audit Log (Append-only)
    audit = AuditLog(
        entity_type="alert",
        entity_id=id,
        action="update_status",
        old_value={"status": old_status},
        new_value={"status": new_status, "reviewer": user["username"], "reason": review.reason},
        actor=user["username"],
    )
    db.add(audit)

    db.commit()
    return {"status": "success", "alert_id": id, "new_status": new_status}


@app.get("/v1/models")
def list_models(
    db: Session = Depends(get_db),
    user: dict = Depends(RoleChecker(["analyst", "supervisor", "admin"])),
):
    """Retrieve model performance metrics list."""
    models = db.query(Model).all()
    return [
        {
            "model_version": m.model_version,
            "architecture": m.architecture,
            "metrics": m.metrics,
            "created_at": m.created_at,
        }
        for m in models
    ]
