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
import os
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .settings import settings
from .tools import build_tools_for_agent

logging.basicConfig(level=settings.LOG_LEVEL.upper())
log = logging.getLogger("crew-ai-service")

# Make LLM keys visible to litellm/CrewAI under the canonical env names.
# Lovable AI Gateway is OpenAI-compatible, so we expose it as OPENAI_*.
if settings.LOVABLE_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.LOVABLE_API_KEY
    os.environ.setdefault("OPENAI_BASE_URL", settings.LOVABLE_AI_BASE_URL)
elif settings.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.OPENAI_BASE_URL:
        os.environ.setdefault("OPENAI_BASE_URL", settings.OPENAI_BASE_URL)

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
class AgentSpec(BaseModel):
    name: str | None = None
    role: str
    goal: str = ""
    backstory: str = ""
    context: str = ""
    tools: list[str] = Field(default_factory=list)
    http_endpoints: list[dict[str, Any]] = Field(default_factory=list)
    allow_delegation: bool = False
    is_master: bool = False
    model: str | None = None


class RunRequest(BaseModel):
    crew: str = Field(..., description="Name of the crew to run (e.g. 'research')")
    process: str | None = Field(default=None, description="'hierarchical' or 'sequential'")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Inputs passed to the crew")
    company: str | None = Field(default=None, description="Optional company scope")
    model: str | None = Field(default=None, description="LLM model override")
    agents: list[AgentSpec] = Field(default_factory=list, description="Active agent roster")


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
    """Run a dynamic crew built from the request body.

    The CIO KB v2 app sends the full active agent roster on every call.
    We build a CrewAI Crew at request time and dispatch the question.
    """
    run_id = str(uuid.uuid4())
    crew_name = req.crew or "dynamic"
    question = (req.inputs or {}).get("question") or (req.inputs or {}).get("prompt") or ""
    log.info("run crew=%s run_id=%s company=%s agents=%d process=%s",
             crew_name, run_id, req.company, len(req.agents), req.process)

    if not req.agents:
        return RunResponse(run_id=run_id, crew=crew_name, status="error",
                           error="No agents provided in request body")
    if not question.strip():
        return RunResponse(run_id=run_id, crew=crew_name, status="error",
                           error="inputs.question is required")

    # Lazy import — CrewAI pulls in heavy deps; keep startup fast and
    # keep import errors visible per-request.
    try:
        from crewai import Agent, Crew, Process, Task
    except Exception as e:
        return RunResponse(run_id=run_id, crew=crew_name, status="error",
                           error=f"CrewAI import failed: {e}")

    default_model = req.model or settings.DEFAULT_MODEL

    def build_agent(spec: AgentSpec) -> Any:
        backstory = spec.backstory or ""
        if spec.context:
            backstory = (backstory + "\n\nAdditional instructions:\n" + spec.context).strip()
        tools = build_tools_for_agent(
            spec.tools, company=req.company, http_endpoints=spec.http_endpoints,
        )
        kwargs: dict[str, Any] = {
            "role": spec.role or spec.name or "Specialist",
            "goal": spec.goal or "Answer the user's question accurately.",
            "backstory": backstory or "An expert collaborator on the CIO KB team.",
            "allow_delegation": bool(spec.allow_delegation),
            "verbose": False,
            "tools": tools,
        }
        model = spec.model or default_model
        if model:
            kwargs["llm"] = model
        return Agent(**kwargs)

    try:
        masters = [a for a in req.agents if a.is_master]
        non_masters = [a for a in req.agents if not a.is_master]
        wants_hier = (req.process or "").lower() == "hierarchical" or len(masters) == 1

        if wants_hier and len(masters) == 1 and non_masters:
            manager = build_agent(masters[0])
            workers = [build_agent(a) for a in non_masters]
            task = Task(
                description=question,
                expected_output=(
                    "A clear, well-structured Markdown answer. Include numbered "
                    "citations like [1] when you use facts from kb_search, and a "
                    "Sources list at the end."
                ),
                agent=manager,
            )
            crew = Crew(
                agents=workers,
                tasks=[task],
                process=Process.hierarchical,
                manager_agent=manager,
                verbose=False,
            )
        else:
            # Sequential: chain one task per agent in the order received.
            agents = [build_agent(a) for a in req.agents]
            tasks: list[Any] = []
            for i, ag in enumerate(agents):
                tasks.append(Task(
                    description=(question if i == 0
                                 else f"Refine and extend the previous step's output to better answer: {question}"),
                    expected_output=(
                        "Markdown contribution; cite kb_search passages by [n] markers."
                    ),
                    agent=ag,
                    context=tasks[:i] or None,
                ))
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
            )

        result = crew.kickoff(inputs={"question": question, "company": req.company or ""})
        # CrewAI returns a CrewOutput object in recent versions; coerce to str.
        output_text = getattr(result, "raw", None) or str(result)

        return RunResponse(
            run_id=run_id,
            crew=crew_name,
            status="ok",
            output=output_text,
        )
    except Exception as e:
        log.exception("crew kickoff failed")
        return RunResponse(
            run_id=run_id, crew=crew_name, status="error",
            error=f"{type(e).__name__}: {e}",
        )