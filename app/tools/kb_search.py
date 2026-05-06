"""kb_search — Vectara v2 query tool.

Mirrors the request shape used by `supabase/functions/_shared/vectara-search.ts`
in the Lovable repo so the agent sees results consistent with the rest
of CIO KB.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from crewai.tools import BaseTool
from pydantic import Field

from ..settings import settings


def _esc(s: str) -> str:
    return s.replace("'", "''")


def _vectara_query(query: str, customer: str | None, top_k: int) -> dict[str, Any]:
    if not (settings.VECTARA_API_KEY and settings.VECTARA_CUSTOMER_ID and settings.VECTARA_CORPUS_KEY):
        return {"configured": False, "hits": [], "error": "Vectara not configured"}

    clauses: list[str] = []
    if customer and customer.strip():
        c = _esc(customer.strip())
        clauses.append(f"(doc.source_company = '{c}' OR doc.company_slug = '{c}' OR doc.customer = '{c}' OR doc.customer = '_shared')")

    payload: dict[str, Any] = {
        "query": query,
        "search": {
            "limit": max(1, min(int(top_k or 8), 25)),
            "reranker": {"type": "mmr"},
        },
    }
    if clauses:
        payload["search"]["metadata_filter"] = " AND ".join(clauses)

    url = f"{settings.VECTARA_BASE_URL}/v2/corpora/{settings.VECTARA_CORPUS_KEY}/query"
    try:
        r = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": settings.VECTARA_API_KEY,
                "customer-id": settings.VECTARA_CUSTOMER_ID,
            },
            timeout=20.0,
        )
    except Exception as e:
        return {"configured": True, "hits": [], "error": f"vectara fetch failed: {e}"}

    if r.status_code >= 400:
        return {"configured": True, "hits": [], "error": f"vectara [{r.status_code}]: {r.text[:300]}"}

    data = r.json() if r.content else {}
    raw = data.get("search_results") or []
    hits = []
    for i, row in enumerate(raw):
        hits.append({
            "rank": i + 1,
            "document_id": str(row.get("document_id") or row.get("doc_id") or ""),
            "score": float(row.get("score") or 0),
            "snippet": str(row.get("text") or row.get("snippet") or "").strip(),
            "metadata": row.get("document_metadata") or row.get("metadata") or {},
        })
    return {"configured": True, "hits": hits}


def _format_hits(query: str, result: dict[str, Any]) -> str:
    if not result.get("configured"):
        return f"[kb_search disabled — {result.get('error', 'no Vectara credentials')}]"
    if result.get("error"):
        return f"[kb_search error: {result['error']}]"
    hits = result.get("hits") or []
    if not hits:
        return f"No knowledge base results for: {query}"
    lines = [f"Query: {query}", "", "Passages:"]
    sources = []
    for h in hits:
        meta = h.get("metadata") or {}
        title = meta.get("title") or meta.get("source_type") or h["document_id"]
        lines.append(f"[{h['rank']}] {h['snippet']}")
        sources.append(f"[{h['rank']}] {title} (id={h['document_id']})")
    lines.append("")
    lines.append("Sources:")
    lines.extend(sources)
    return "\n".join(lines)


class KbSearchTool(BaseTool):
    name: str = "kb_search"
    description: str = (
        "Search the CIO knowledge base (Vectara). Args: query (str), "
        "customer (str, optional — scopes to a customer + _shared), "
        "top_k (int, default 8). Returns numbered passages with [n] markers "
        "and a Sources block. Cite passages by their [n] marker."
    )
    default_customer: str | None = Field(default=None)

    def _run(self, query: str, customer: str | None = None, top_k: int = 8) -> str:
        cust = customer if customer is not None else self.default_customer
        return _format_hits(query, _vectara_query(query, cust, top_k))


def build_kb_search(company: str | None) -> KbSearchTool:
    return KbSearchTool(default_customer=company)