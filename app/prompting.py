# app/prompting.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_rag_prompt(q: str, hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns:
      - prompt (str)
      - citations (list) aligned to [1], [2], ...
    Each citation entry includes doc_title, doc_source, chunk_id, chunk_index.
    """
    citations = []
    context_blocks = []

    for i, h in enumerate(hits, start=1):
        citations.append(
            {
                "n": i,
                "doc_title": h.get("doc_title"),
                "doc_source": h.get("doc_source"),
                "chunk_id": h.get("chunk_id"),
                "chunk_index": h.get("chunk_index"),
            }
        )
        content = (h.get("content") or "").strip()
        context_blocks.append(
            f"[{i}] doc_title={h.get('doc_title')} | source={h.get('doc_source')} | chunk_id={h.get('chunk_id')}\n"
            f"{content}"
        )

    system = (
        "You are a study assistant for AWS certifications.\n"
        "Rules:\n"
        "- Use ONLY the provided Context to answer.\n"
        "- If the Context does not contain enough information, say that clearly and ask a follow-up question.\n"
        "- When you use a fact from a context item, cite it inline like [1], [2].\n"
        "- Do NOT invent sources or details.\n"
    )

    user = (
        f"Question:\n{q}\n\n"
        f"Context:\n" + ("\n\n".join(context_blocks) if context_blocks else "(no context retrieved)")
    )

    prompt = f"{system}\n\n{user}"
    return prompt, citations
