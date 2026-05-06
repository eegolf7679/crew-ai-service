"""http_call — whitelisted REST caller.

Each agent gets its own instance bound to its `http_endpoints` list.
The LLM may only invoke endpoints by `endpoint_name`; arbitrary URLs
are rejected to keep the surface area tight.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx
from crewai.tools import BaseTool
from pydantic import Field

from ..settings import settings


class HttpCallTool(BaseTool):
    name: str = "http_call"
    description: str = (
        "Call a pre-approved external HTTP endpoint by NAME. Args: "
        "endpoint_name (str, required — must be one of the names listed below), "
        "path (str, optional — appended to the endpoint URL), "
        "query (dict, optional), body (dict, optional). "
        "Returns 'HTTP <status>\\n<body>' truncated to ~8KB."
    )
    endpoints_by_name: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def _run(
        self,
        endpoint_name: str,
        path: str = "",
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> str:
        ep = self.endpoints_by_name.get((endpoint_name or "").strip())
        if not ep:
            allowed = ", ".join(sorted(self.endpoints_by_name.keys())) or "(none)"
            return f"[http_call rejected: unknown endpoint '{endpoint_name}'. Allowed: {allowed}]"

        method = (ep.get("method") or "GET").upper()
        base_url = ep.get("url") or ""
        url = urljoin(base_url if base_url.endswith("/") else base_url + "/", path.lstrip("/")) if path else base_url
        headers = dict(ep.get("headers") or {})

        try:
            r = httpx.request(
                method, url,
                params=query or None,
                json=body if (method in ("POST", "PUT", "PATCH") and body is not None) else None,
                headers=headers,
                timeout=settings.HTTP_CALL_TIMEOUT_S,
            )
        except Exception as e:
            return f"[http_call error: {e}]"

        text = r.text or ""
        if len(text) > settings.HTTP_CALL_MAX_BYTES:
            text = text[: settings.HTTP_CALL_MAX_BYTES] + "\n…[truncated]"
        return f"HTTP {r.status_code}\n{text}"


def build_http_call(endpoints: list[dict[str, Any]]) -> HttpCallTool:
    by_name: dict[str, dict[str, Any]] = {}
    for e in endpoints or []:
        n = (e.get("name") or "").strip()
        if n and e.get("url"):
            by_name[n] = e
    tool = HttpCallTool(endpoints_by_name=by_name)
    if by_name:
        listing = "; ".join(f"{n} ({by_name[n].get('method','GET')} {by_name[n]['url']})" for n in by_name)
        tool.description = tool.description + f"\nAvailable endpoints: {listing}"
    else:
        tool.description = tool.description + "\nAvailable endpoints: (none — do not call this tool)"
    return tool