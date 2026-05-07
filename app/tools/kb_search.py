"""kb_search — Vectara v2 query tool.

Calls Vectara v2 multi-corpora /v2/query endpoint, scopes results to the
caller's customer via metadata_filter, and appends every hit to a shared
per-run source collector (deduped by doc_id) so /run can return a
top-level `sources` array to the UI.
"""
from __future__ import annotations

from typing import Any

import httpx
from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr

from ..settings import settings


def _esc(s: str) -> str:
    return s.replace("'", "''")


def build_filter(company: str | None, extra: str | None = None) -> str:
    parts: list[str] = []
    if company and company.strip():
        # Document-level metadata fields on this corpus.
        # Match the company OR shared/global docs.
        c = _esc(company.strip())
        parts.append(
            f"(doc.source_company = '{c}' OR doc.company_slug = '_shared')"
        )
    if extra and extra.strip():
        parts.append(f"({extra.strip()})")
    return " AND ".join(parts)


def _vectara_query(query: str, customer: str | None, top_k: int) -> dict[str, Any]:
    if not (settings.VECTARA_API_KEY and settings.VECTARA_CORPUS_KEY):
        return {"configured": False, "results": [], "total": 0,
                "error": "Vectara not configured"}

    metadata_filter = build_filter(customer)

    payload: dict[str, Any] = {
        "query": query,
        "search": {
            "corpora": [
                {
                    "corpus_key": settings.VECTARA_CORPUS_KEY,
                    "metadata_filter": metadata_filter,
                    "lexical_interpolation": 0.05,
                }
            ],
            "limit": max(1, min(int(top_k or 10), 25)),
            "context_configuration": {
                "sentences_before": 2,
                "sentences_after": 2,
            },
            "reranker": {"type": "mmr", "diversity_bias": 0.2},
        },
        "generation": None,
    }

    url = f"{settings.VECTARA_BASE_URL}/v2/query"
    try:
        r = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": settings.VECTARA_API_KEY,
            },
            timeout=25.0,
        )
    except Exception as e:
        return {"configured": True, "results": [], "total": 0,
                "error": f"vectara fetch failed: {e}"}

    if r.status_code >= 400:
        return {"configured": True, "results": [], "total": 0,
                "error": f"vectara [{r.status_code}]: {r.text[:300]}"}

    data = r.json() if r.content else {}
    raw = data.get("search_results") or []
    results: list[dict[str, Any]] = []
    for row in raw:
        meta = row.get("document_metadata") or row.get("metadata") or {}
        if isinstance(meta, list):
            meta = {m.get("name"): m.get("value") for m in meta if isinstance(m, dict)}
        doc_id = str(row.get("document_id") or row.get("doc_id") or "")
        if not doc_id:
            continue
        results.append({
            "doc_id": doc_id,
            "title": str(meta.get("title") or meta.get("source_type") or doc_id),
            "score": float(row.get("score") or 0),
            "snippet": str(row.get("text") or row.get("snippet") or "").strip(),
            "url": meta.get("url") or meta.get("source_url"),
            "source_type": meta.get("source_type"),
            "metadata": meta,
        })

    return {"configured": True, "results": results, "total": len(results)}


def _format_for_agent(query: str, customer: str | None, result: dict[str, Any]) -> str:
    if not result.get("configured"):
        return f"[kb_search disabled — {result.get('error', 'no Vectara credentials')}]"
    if result.get("error"):
        return f"[kb_search error: {result['error']}]"
    results = result.get("results") or []
    scope = f" (customer={customer})" if customer else ""
    if not results:
        return (f"No documentation found for: {query}{scope}\n"
                f"Try rephrasing the search before answering.")
    lines = [f"Query: {query}{scope}", "", "Passages:"]
    sources = []
    for i, h in enumerate(results, start=1):
        lines.append(f"[{i}] {h['snippet']}")
        title = h.get("title") or h["doc_id"]
        sources.append(f"[{i}] {title} (doc_id={h['doc_id']})")
    lines.append("")
    lines.append("Cite each claim using the doc_id and title, e.g. "
                 "[abc-123 — \"How to add a new user\"].")
    lines.append("")
    lines.append("Sources:")
    lines.extend(sources)
    return "\n".join(lines)


class KbSearchTool(BaseTool):
    name: str = "kb_search"
    description: str = (
        "Search the customer-scoped knowledge base (Vectara). "
        "Args: query (str). Optional: top_k (int, default 10). "
        "The customer scope is auto-injected from the run; do NOT override. "
        "You MUST call this tool with at least 2 different phrasings before "
        "concluding 'no documentation found'. Cite every claim with "
        "[doc_id — \"title\"]."
    )
    default_customer: str | None = Field(default=None)
    require_customer: bool = Field(default=True)
    _sources_sink: list[dict[str, Any]] | None = PrivateAttr(default=None)
    _tool_trace: list[dict[str, Any]] | None = PrivateAttr(default=None)

    def attach_sink(self, sink: list[dict[str, Any]]) -> None:
        self._sources_sink = sink

    def attach_trace(self, trace: list[dict[str, Any]]) -> None:
        self._tool_trace = trace

    def _run(self, query: str, top_k: int = 10, **_: Any) -> str:
        cust = self.default_customer
        if self.require_customer and not (cust and cust.strip()):
            if self._tool_trace is not None:
                self._tool_trace.append({
                    "tool": "kb_search", "query": query, "customer": None,
                    "status": "refused_no_customer", "result_count": 0,
                })
            return ("[kb_search refused — no customer scope on this run. "
                    "Tell the user you cannot search without a selected customer.]")

        result = _vectara_query(query, cust, top_k)

        if self._tool_trace is not None:
            self._tool_trace.append({
                "tool": "kb_search",
                "query": query,
                "customer": cust,
                "status": "error" if result.get("error") else "ok",
                "error": result.get("error"),
                "result_count": len(result.get("results") or []),
            })

        # dedupe-append into the per-run sources sink
        if self._sources_sink is not None and result.get("results"):
            seen = {s["doc_id"] for s in self._sources_sink}
            for h in result["results"]:
                if h["doc_id"] in seen:
                    continue
                seen.add(h["doc_id"])
                self._sources_sink.append({
                    "doc_id": h["doc_id"],
                    "title": h.get("title"),
                    "url": h.get("url"),
                    "score": h.get("score"),
                    "source_type": h.get("source_type"),
                })

        return _format_for_agent(query, cust, result)


def build_kb_search(company: str | None,
                    sources_sink: list[dict[str, Any]] | None = None,
                    tool_trace: list[dict[str, Any]] | None = None) -> KbSearchTool:
    tool = KbSearchTool(default_customer=company)
    if sources_sink is not None:
        tool.attach_sink(sources_sink)
    if tool_trace is not None:
        tool.attach_trace(tool_trace)
    return tool
