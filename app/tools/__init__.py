"""Tool factory for the crew-ai-service.

Each tool ID sent by the CIO KB v2 app maps to a CrewAI BaseTool. The
factory is per-request because some tools (notably http_call) are
scoped to the agent's whitelisted endpoints.
"""
from __future__ import annotations

from typing import Any

from .kb_search import build_kb_search
from .http_call import build_http_call


def build_tools_for_agent(
    tool_ids: list[str],
    *,
    company: str | None,
    http_endpoints: list[dict[str, Any]] | None,
    sources_sink: list[dict[str, Any]] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> list[Any]:
    out: list[Any] = []
    for tid in tool_ids or []:
        tid = (tid or "").strip().lower()
        if tid == "kb_search":
            out.append(build_kb_search(
                company=company,
                sources_sink=sources_sink,
                tool_trace=tool_trace,
            ))
        elif tid == "web_search":
            # web_search has been removed from the platform; treat as no-op
            # so older agent configs don't crash the run.
            continue
        elif tid == "http_call":
            out.append(build_http_call(endpoints=http_endpoints or []))
        # silently ignore unknown ids — the consumer app may roll out
        # new tool IDs before the service knows them
    return out