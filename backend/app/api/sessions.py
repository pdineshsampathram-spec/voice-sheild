"""
Session history endpoints.

GET /api/sessions — List recent analysis sessions (from in-memory store for prototype).
GET /api/sessions/{session_id} — Get details of a specific session.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import SecurityEvent, AlertPayload
from app.services.alerts.alert_service import alert_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session store for prototype
# Will be replaced with PostgreSQL in Phase 9
_session_store: dict[str, dict] = {}
_event_store: list[SecurityEvent] = []


def log_security_event(event: SecurityEvent):
    """Log a security event (in-memory for prototype)."""
    _event_store.append(event)
    if len(_event_store) > 1000:
        _event_store.pop(0)


@router.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """List recent analysis sessions."""
    events = list(reversed(_event_store[-limit:]))
    return {"sessions": [e.model_dump(mode="json") for e in events]}


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific session."""
    matching = [e for e in _event_store if e.session_id == session_id]
    if not matching:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"events": [e.model_dump(mode="json") for e in matching]}


@router.get("/api/alerts/recent")
async def get_recent_alerts(limit: int = 20):
    """Get recent security alerts."""
    alerts = alert_service.get_recent_alerts(limit)
    return {"alerts": [a.model_dump(mode="json") for a in alerts]}
