# app/rag/ingest.py
import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .db import get_db_mode, sqlite_conn
from .embeddings import embed_texts


@dataclass
class Chunk:
    chunk_index: int
    content: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk_index=idx, content=chunk))
            idx += 1

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def _emb_to_blob(vec) -> tuple[bytes, int]:
    a = np.asarray(vec, dtype=np.float32)
    return a.tobytes(), int(a.shape[0])


def ingest_markdown(
    *,
    user_id: str,
    notebook: str,
    title: str,
    source: str,
    markdown: str
) -> Tuple[str, int]:
    """
    Multi-tenant ingest:
      - scoped to (user_id, notebook)
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(markdown, chunk_size, overlap)

    if not chunks:
        return doc_id, 0

    embeddings = embed_texts([c.content for c in chunks])

    if get_db_mode() == "sqlite":
        with sqlite_conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, user_id, notebook, title, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, user_id, notebook, title, source),
            )

            for c, emb in zip(chunks, embeddings):
                chunk_id = str(uuid.uuid4())
                blob, dim = _emb_to_blob(emb)
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, user_id, doc_id, notebook, chunk_index, content, token_count, embedding, embedding_dim
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, user_id, doc_id, notebook, c.chunk_index, c.content, None, blob, dim),
                )

        return doc_id, len(chunks)

    raise RuntimeError("Postgres mode is not supported for multi-tenant ingest in this MVP")
