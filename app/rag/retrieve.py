# app/rag/retrieve.py
import os
from typing import List, Dict, Any

import numpy as np

from .db import get_db_mode, sqlite_conn
from .embeddings import embed_texts


def _cosine_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(q) + 1e-12
    Mn = np.linalg.norm(M, axis=1) + 1e-12
    return (M @ q) / (Mn * qn)


def retrieve(
    *,
    user_id: str,
    notebook: str,
    query: str,
    top_k: int | None = None
) -> List[Dict[str, Any]]:
    k = top_k or int(os.getenv("TOP_K", "5"))

    if get_db_mode() != "sqlite":
        raise RuntimeError("Postgres mode is not supported for multi-tenant retrieve in this MVP")

    q_vec = embed_texts([query])[0]
    q = np.asarray(q_vec, dtype=np.float32)

    with sqlite_conn() as conn:
        rows = conn.execute(
            """
            SELECT
              c.id AS chunk_id,
              d.title AS doc_title,
              d.source AS doc_source,
              c.notebook AS notebook,
              c.chunk_index,
              c.content,
              c.embedding,
              c.embedding_dim
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.user_id = ?
              AND c.notebook = ?
            """,
            (user_id, notebook),
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
                "notebook": r["notebook"],
                "chunk_index": int(r["chunk_index"]),
                "content": r["content"],
                "score": float(sims[int(i)]),
            }
        )

    return hits


