from typing import List, Dict, Any


SYSTEM_PROMPT = """You are an AWS AI Practitioner study assistant.

Rules:
- Use ONLY the provided context.
- If the answer is not in the context, say: "Not found in knowledge base."
- Be concise, factual, and structured.
- Prefer bullet points when appropriate.
"""


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Assemble an MCP-style prompt with strict system rules
    and retrieved context.
    """
    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"[source: {c['doc_title']} | chunk:{c['chunk_index']} | score:{c['score']:.3f}]\n"
            f"{c['content']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""
