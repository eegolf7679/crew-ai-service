"""web_search — provider-agnostic web search.

Picks the first configured provider in this order: Tavily, Serper,
Brave. If none are configured, returns a clear notice so the agent
stops calling it.
"""
from __future__ import annotations

from typing import Any

import httpx
from crewai.tools import BaseTool

from ..settings import settings


def _tavily(query: str) -> list[dict[str, Any]]:
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": settings.TAVILY_API_KEY, "query": query, "max_results": 5},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    return [
        {"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content")}
        for x in (data.get("results") or [])[:5]
    ]


def _serper(query: str) -> list[dict[str, Any]]:
    r = httpx.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": 5},
        headers={"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for x in (data.get("organic") or [])[:5]:
        out.append({"title": x.get("title"), "url": x.get("link"), "snippet": x.get("snippet")})
    return out


def _brave(query: str) -> list[dict[str, Any]]:
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": 5},
        headers={"X-Subscription-Token": settings.BRAVE_API_KEY, "Accept": "application/json"},
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for x in ((data.get("web") or {}).get("results") or [])[:5]:
        out.append({"title": x.get("title"), "url": x.get("url"), "snippet": x.get("description")})
    return out


def _format(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No web results."
    lines = []
    for r in results:
        lines.append(f"- [{r.get('title') or r.get('url')}]({r.get('url')})\n  {r.get('snippet') or ''}")
    return "\n".join(lines)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the public web. Args: query (str). Returns top 5 results "
        "as a markdown bullet list with title, URL, and snippet."
    )

    def _run(self, query: str) -> str:
        try:
            if settings.TAVILY_API_KEY:
                return _format(_tavily(query))
            if settings.SERPER_API_KEY:
                return _format(_serper(query))
            if settings.BRAVE_API_KEY:
                return _format(_brave(query))
        except Exception as e:
            return f"[web_search error: {e}]"
        return "[web_search disabled — set TAVILY_API_KEY, SERPER_API_KEY, or BRAVE_API_KEY]"


def build_web_search() -> WebSearchTool:
    return WebSearchTool()