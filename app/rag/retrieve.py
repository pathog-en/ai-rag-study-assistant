import os
from typing import List, Dict, Any

from .db import db_conn
from .embeddings import embed_texts


def _to_pgvector_literal(vec: list[float]) -> str:
    # pgvector accepts text like: '[0.1,0.2,0.3]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    k = top_k or int(os.getenv("TOP_K", "5"))
    query_embedding = embed_texts([query])[0]
    qv = _to_pgvector_literal(query_embedding)

    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              c.id AS chunk_id,
              d.title AS doc_title,
              d.source AS doc_source,
              c.chunk_index,
              c.content,
              1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
            """,
            (qv, qv, k),
        )
        rows = cur.fetchall()

    return [dict(r) for r in rows]
