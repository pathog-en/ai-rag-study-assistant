import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


def _env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def db_conn() -> psycopg.Connection:
    """
    Create a psycopg (v3) connection and register pgvector type adapters.

    Requires env vars:
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    conn = psycopg.connect(
        host=_env("DB_HOST", "localhost"),
        port=int(_env("DB_PORT", "5432")),
        dbname=_env("DB_NAME", "ragdb"),
        user=_env("DB_USER", "rag"),
        password=_env("DB_PASSWORD", "rag"),
        row_factory=dict_row,
    )
    # This makes Python lists/pgvector types work cleanly with psycopg parameters
    register_vector(conn)
    return conn


@contextmanager
def get_cursor() -> Iterator[psycopg.Cursor]:
    """
    Convenience context manager if you want it elsewhere.
    """
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize schema (documents + chunks) and ensure pgvector extension exists.
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              source TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )

        # NOTE: embedding dimension defaults to 1024 (Titan embed v2).
        # If you change embedding model, update this env var (EMBED_DIM).
        embed_dim = int(os.getenv("EMBED_DIM", "1024"))

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              chunk_index INT NOT NULL,
              content TEXT NOT NULL,
              token_count INT NULL,
              embedding vector({embed_dim}) NOT NULL
            );
            """
        )

        # Optional index (fine to keep for now; pgvector warns when table is tiny)
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            """
        )

        # If you already have an ivfflat index in your compose/init, keep it.
        # If not, you can add later once you have more chunks.
        conn.commit()
