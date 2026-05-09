# app/tools/http_endpoint_tool.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class _HttpEndpointInput(BaseModel):
    """Default input schema — accepts a free-form JSON string of params/body."""
    payload: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON object as a string. For GET requests, keys become "
            "query string parameters. For POST requests, this is sent as the "
            "JSON body. Example: '{\"size\": 50, \"_filters\": \"...\"}'."
        ),
    )
    path_params: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON object of path parameters to substitute into the "
            "URL template (e.g. {device_id}). Example: '{\"device_id\": \"abc\"}'."
        ),
    )


class HttpEndpointTool(BaseTool):
    """Wrap a single declarative HTTP endpoint as a CrewAI tool."""

    name: str
    description: str
    args_schema: Type[BaseModel] = _HttpEndpointInput

    # Endpoint config (excluded from the schema the LLM sees)
    method: str = "GET"
    url: str = ""
    headers: Dict[str, str] = {}
    destructive: bool = False
    timeout_s: int = 60

    def _run(self, payload: Optional[str] = None, path_params: Optional[str] = None) -> str:  # type: ignore[override]
        try:
            params_obj: Dict[str, Any] = {}
            if payload:
                try:
                    params_obj = json.loads(payload)
                    if not isinstance(params_obj, dict):
                        return f"[{self.name}] payload must be a JSON object"
                except json.JSONDecodeError as e:
                    return f"[{self.name}] invalid JSON in payload: {e}"

            url = self.url
            if path_params:
                try:
                    pp = json.loads(path_params)
                    if isinstance(pp, dict):
                        for k, v in pp.items():
                            url = url.replace("{" + k + "}", str(v))
                except json.JSONDecodeError:
                    pass

            method = (self.method or "GET").upper()
            req_kwargs: Dict[str, Any] = {
                "headers": self.headers or {},
                "timeout": self.timeout_s,
            }
            if method == "GET":
                req_kwargs["params"] = params_obj
            else:
                req_kwargs["json"] = params_obj

            r = requests.request(method, url, **req_kwargs)
            ct = r.headers.get("content-type", "")
            body: Any
            if "application/json" in ct:
                try:
                    body = r.json()
                except Exception:
                    body = r.text
            else:
                body = r.text

            # Truncate huge responses so we don't blow the LLM context
            text = body if isinstance(body, str) else json.dumps(body, default=str)
            if len(text) > 12000:
                text = text[:12000] + f"\n…[truncated, total {len(text)} chars]"

            if not r.ok:
                return f"[{self.name}] HTTP {r.status_code}: {text}"
            return text
        except requests.RequestException as e:
            return f"[{self.name}] request error: {e}"


def build_http_tools(http_endpoints: Optional[list]) -> list[HttpEndpointTool]:
    """Convert a list of forwarded endpoint definitions into CrewAI tools."""
    tools: list[HttpEndpointTool] = []
    if not http_endpoints:
        return tools
    for ep in http_endpoints:
        if not isinstance(ep, dict):
            continue
        name = str(ep.get("name") or "").strip()
        url = str(ep.get("url") or "").strip()
        if not name or not url:
            continue
        tools.append(
            HttpEndpointTool(
                name=name,
                description=str(ep.get("description") or name),
                method=str(ep.get("method") or "GET"),
                url=url,
                headers=dict(ep.get("headers") or {}),
                destructive=bool(ep.get("destructive") or False),
            )
        )
    return tools