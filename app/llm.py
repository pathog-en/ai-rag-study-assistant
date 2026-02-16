# app/llm.py
from __future__ import annotations

import os
from typing import Optional


class LLMNotConfiguredError(RuntimeError):
    pass


def generate_answer(prompt: str) -> str:
    """
    Minimal adapter. Start as a stub so /chat works end-to-end.
    Later swap this to OpenAI or Bedrock without changing the endpoint.
    """
    mode = os.getenv("LLM_MODE", "stub").lower()

    if mode == "stub":
        # Simple placeholder: echoes that the pipeline works.
        return (
            "STUB ANSWER (LLM_MODE=stub)\n\n"
            "I can retrieve relevant context, but an LLM provider is not configured yet.\n"
            "Once configured, I will answer using ONLY the provided context and include citations like [1], [2]."
        )

    raise LLMNotConfiguredError(
        f"Unsupported LLM_MODE={mode}. Set LLM_MODE=stub for now."
    )
