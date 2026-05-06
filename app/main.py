"""CrewAI service skeleton.

Minimal FastAPI app exposing:
  GET  /health           — liveness probe
  GET  /                 — service info
  POST /run              — run a crew (stub, returns echo until agents are wired)

Auth: every protected route requires `Authorization: Bearer <SHARED_TOKEN>`.

The actual CrewAI agents/tools will be wired in by the consumer app
(CIO KB v2). This file deliberately keeps the crew definition empty so the
consumer can drop in agents, tasks, and tools without fighting framework
boilerplate.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .settings import settings

logging.basicConfig(level=settings.LOG_LEVEL.upper())
log = logging.getLogger("crew-ai-service")

app = FastAPI(title="Crew AI Service", version="0.1.0")


# ---------- auth ----------
def require_bearer(authorization: str | None = Header(default=None)) -> None:
    if not settings.SHARED_TOKEN:
        # Fail closed if the operator forgot to set the token.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHARED_TOKEN not configured on server",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.SHARED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# ---------- schema ----------
class RunRequest(BaseModel):
    crew: str = Field(..., description="Name of the crew to run (e.g. 'research')")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Inputs passed to the crew")
    company: str | None = Field(default=None, description="Optional company scope")
    model: str | None = Field(default=None, description="LLM model override")


class RunResponse(BaseModel):
    run_id: str
    crew: str
    status: str
    output: Any | None = None
    error: str | None = None


# ---------- routes ----------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "crew-ai-service",
        "version": app.version,
        "endpoints": ["/health", "/run"],
        "auth": "Bearer SHARED_TOKEN",
    }


@app.post("/run", response_model=RunResponse, dependencies=[Depends(require_bearer)])
def run(req: RunRequest) -> RunResponse:
    """Run a named crew.

    NOTE: This is a skeleton. Crews are not yet defined — the consumer
    application (CIO KB v2) will register them. Until then this endpoint
    returns a deterministic echo so callers can validate connectivity,
    auth, and request/response shape.
    """
    run_id = str(uuid.uuid4())
    log.info("run requested crew=%s run_id=%s company=%s", req.crew, run_id, req.company)

    # Placeholder: replace with crew registry lookup + Crew().kickoff()
    return RunResponse(
        run_id=run_id,
        crew=req.crew,
        status="not_implemented",
        output={
            "message": "Crew registry not yet populated. Wire crews into app/crews/.",
            "echo": req.model_dump(),
        },
    )