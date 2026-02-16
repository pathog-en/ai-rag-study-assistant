# app/rag/ingest.py
import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pgvector.psycopg import Vector

from .db import DB_MODE, db_conn, sqlite_conn
from .embeddings import embed_texts


@dataclass
class Chunk:
    chunk_index: int
    content: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Simple character-based chunking (MVP).
    Keeps overlap to preserve context across chunk boundaries.
    """
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


def ingest_markdown(title: str, source: str, markdown: str) -> Tuple[str, int]:
    """
    Ingest markdown text into:
      - documents table (metadata)
      - chunks table (chunk text + embedding)

    SQLite mode:
      - embedding stored as float32 BLOB + embedding_dim
    Postgres mode:
      - embedding stored as pgvector

    Returns: (doc_id, chunks_added)
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(markdown, chunk_size, overlap)

    if not chunks:
        return doc_id, 0

    embeddings = embed_texts([c.content for c in chunks])

    if DB_MODE == "sqlite":
        with sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO documents (id, title, source) VALUES (?, ?, ?)",
                (doc_id, title, source),
            )

            for c, emb in zip(chunks, embeddings):
                chunk_id = str(uuid.uuid4())
                blob, dim = _emb_to_blob(emb)
                conn.execute(
                    """
                    INSERT INTO chunks (id, doc_id, chunk_index, content, token_count, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, c.chunk_index, c.content, None, blob, dim),
                )

        return doc_id, len(chunks)

    # Postgres / pgvector path (your original logic)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, title, source) VALUES (%s, %s, %s)",
            (doc_id, title, source),
        )

        for c, emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chunks (id, doc_id, chunk_index, content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (chunk_id, doc_id, c.chunk_index, c.content, None, Vector(emb)),
            )

    return doc_id, len(chunks)

