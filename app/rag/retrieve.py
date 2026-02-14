import os
from typing import List, Dict, Any

from pgvector import Vector

from .db import db_conn
from .embeddings import embed_texts


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks using pgvector cosine distance (<=>).

    Key detail: wrap the embedding as pgvector.Vector so psycopg sends it
    as a pgvector 'vector' type (not float8[]).
    """
    k = top_k or int(os.getenv("TOP_K", "5"))
    query_embedding = embed_texts([query])[0]
    qv = Vector(query_embedding)

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
            (qv, qv, k),
        )
        rows = cur.fetchall()

    return [dict(r) for r in rows]
