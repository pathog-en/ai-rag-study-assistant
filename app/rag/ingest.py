import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from .db import db_conn
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
        start = end - overlap if end - overlap > start else end

    return chunks


def ingest_markdown(title: str, source: str, markdown: str) -> Tuple[str, int]:
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(markdown, chunk_size, overlap)

    if not chunks:
        return doc_id, 0

    embeddings = embed_texts([c.content for c in chunks])

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
                (chunk_id, doc_id, c.chunk_index, c.content, None, emb),
            )

        conn.commit()

    return doc_id, len(chunks)
