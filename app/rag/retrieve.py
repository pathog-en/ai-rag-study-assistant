import os
from typing import List, Dict, Any

from .db import db_conn
from .embeddings import embed_texts


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Embed the query, perform vector similarity search,
    and return the top-k matching chunks.
    """
    k = top_k or int(os.getenv("TOP_K", "5"))
    query_embedding = embed_texts([query])[0]

    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              c.id AS chunk_id,
              d.title AS doc_title,
              d.source AS doc_source,
              c.chunk_index,
              c.content,
              1 - (c.embedding <=> %s) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            ORDER BY c.embedding <=> %s
            LIMIT %s;
            """,
            (query_embedding, query_embedding, k),
        )
        rows = cur.fetchall()

    return rows
