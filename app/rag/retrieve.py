# app/rag/retrieve.py
import os
from typing import List, Dict, Any

import numpy as np
from pgvector.psycopg import Vector

from .db import get_db_mode, db_conn, sqlite_conn
from .embeddings import embed_texts


def _cosine_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q: (d,), M: (n, d)
    qn = np.linalg.norm(q) + 1e-12
    Mn = np.linalg.norm(M, axis=1) + 1e-12
    return (M @ q) / (Mn * qn)


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    k = top_k or int(os.getenv("TOP_K", "5"))

    if get_db_mode() == "sqlite":
        # 1) embed query
        q_vec = embed_texts([query])[0]
        q = np.asarray(q_vec, dtype=np.float32)

        # 2) load all chunks + embeddings from sqlite
        with sqlite_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                  c.id AS chunk_id,
                  d.title AS doc_title,
                  d.source AS doc_source,
                  c.chunk_index,
                  c.content,
                  c.embedding,
                  c.embedding_dim
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                """
            ).fetchall()

        if not rows:
            return []

        embs = []
        metas = []
        for r in rows:
            dim = int(r["embedding_dim"])
            v = np.frombuffer(r["embedding"], dtype=np.float32, count=dim)
            embs.append(v)
            metas.append(r)

        M = np.vstack(embs)  # (n, d)
        sims = _cosine_sim_matrix(q, M)

        idx = np.argsort(-sims)[:k]

        hits: List[Dict[str, Any]] = []
        for i in idx:
            r = metas[int(i)]
            hits.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_title": r["doc_title"],
                    "doc_source": r["doc_source"],
                    "chunk_index": int(r["chunk_index"]),
                    "content": r["content"],
                    "score": float(sims[int(i)]),  # higher is better
                }
            )
        return hits

    # Postgres / pgvector path (your original logic)
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


