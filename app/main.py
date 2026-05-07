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
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)


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

    # Per-run collector that kb_search appends to. Returned as `sources`.
    sources_sink: list[dict[str, Any]] = []
    # Per-run tool invocation trace (kb_search etc append to this).
    tool_trace: list[dict[str, Any]] = []

    MASTER_DELEGATION_RULES = (
        "\n\nDelegation rules (MANDATORY):\n"
        "- You MUST delegate any factual, how-to, configuration, or "
        "troubleshooting question to the Knowledge Base Specialist.\n"
        "- Never answer from your own training data.\n"
        "- If no specialist can handle the request, say so plainly and "
        "stop — do not invent an answer."
    )
    KB_SPECIALIST_RULES = (
        "\n\nKnowledge Base rules (MANDATORY):\n"
        "- You MUST call kb_search at least TWICE with DIFFERENT phrasings "
        "of the question before answering (e.g. 'install new user', "
        "'create user account', 'user provisioning steps').\n"
        "- Only after exhausting reasonable rephrasings may you say "
        "'no documentation found'.\n"
        "- When you answer, you MUST: (1) quote or paraphrase the snippet "
        "you used; (2) cite the doc_id and title for every claim, like "
        "[abc-123 — \"How to add a new user\"]; (3) ignore any result "
        "that looks off-topic for the current customer."
    )

    def build_agent(spec: AgentSpec) -> Any:
        backstory = spec.backstory or ""
        if spec.context:
            backstory = (backstory + "\n\nAdditional instructions:\n" + spec.context).strip()
        # CrewAI requires the manager to have no tools when running
        # hierarchically. We also strip tools from masters in our forced
        # sequential KB-first pipeline so the master only synthesizes.
        agent_tools = [] if spec.is_master else spec.tools
        tools = build_tools_for_agent(
            agent_tools,
            company=req.company,
            http_endpoints=spec.http_endpoints,
            sources_sink=sources_sink,
            tool_trace=tool_trace,
        )
        role_lc = (spec.role or "").lower()
        if spec.is_master:
            backstory = (backstory + MASTER_DELEGATION_RULES).strip()
        if "knowledge base" in role_lc or "kb_search" in (spec.tools or []):
            backstory = (backstory + KB_SPECIALIST_RULES).strip()
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
        kb_specialists = [a for a in non_masters
                          if "kb_search" in [t.lower() for t in (a.tools or [])]]

        if len(masters) == 1 and kb_specialists:
            # FORCED KB-FIRST SEQUENTIAL PIPELINE.
            # Hierarchical delegation in CrewAI is fragile — managers
            # frequently narrate "I will delegate" without actually firing
            # the delegation tool. Instead: run KB Specialist(s) first so
            # kb_search definitely executes, then have the master
            # synthesize a final grounded answer from those results.
            kb_agents = [build_agent(a) for a in kb_specialists]
            other_workers = [build_agent(a) for a in non_masters
                             if a not in kb_specialists]
            manager = build_agent(masters[0])

            tasks: list[Any] = []
            for ag in kb_agents:
                tasks.append(Task(
                    description=(
                        f"Research the user's question using kb_search. "
                        f"You MUST call kb_search at least twice with "
                        f"different phrasings before concluding. "
                        f"Question: {question}"
                    ),
                    expected_output=(
                        "Markdown research notes. Quote the relevant passages "
                        "and cite each with [doc_id — \"title\"]. If nothing "
                        "is found after multiple searches, say so explicitly."
                    ),
                    agent=ag,
                ))
            for ag in other_workers:
                tasks.append(Task(
                    description=(
                        f"Using the prior research, contribute your "
                        f"specialty's perspective on: {question}"
                    ),
                    expected_output="Markdown contribution with citations.",
                    agent=ag,
                    context=tasks[:],
                ))
            tasks.append(Task(
                description=(
                    f"Synthesize the prior research into a final, "
                    f"well-structured Markdown answer to the user's "
                    f"question. Use ONLY facts from the research above. "
                    f"If the research found nothing, say so plainly and "
                    f"do not invent an answer.\n\nQuestion: {question}"
                ),
                expected_output=(
                    "Final Markdown answer with [doc_id — \"title\"] "
                    "citations and a Sources list at the end."
                ),
                agent=manager,
                context=tasks[:],
            ))
            crew = Crew(
                agents=kb_agents + other_workers + [manager],
                tasks=tasks,
                process=Process.sequential,
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
            sources=sources_sink,
            tool_trace=tool_trace,
        )
    except Exception as e:
        log.exception("crew kickoff failed")
        return RunResponse(
            run_id=run_id, crew=crew_name, status="error",
            error=f"{type(e).__name__}: {e}",
            tool_trace=tool_trace,
        )