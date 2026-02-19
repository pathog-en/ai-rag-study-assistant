# app/rag/db.py
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator


def get_db_mode() -> str:
    return os.getenv("DB_MODE", "sqlite").lower()


SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/app.db")


def _ensure_sqlite_dir():
    d = os.path.dirname(SQLITE_PATH)
    if d:
        os.makedirs(d, exist_ok=True)


@contextmanager
def sqlite_conn() -> Iterator[sqlite3.Connection]:
    _ensure_sqlite_dir()
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_conn():
    """
    Backwards-compatible connection factory.

    - If DB_MODE=sqlite -> yields sqlite3.Connection
    - Else -> yields psycopg.Connection (Postgres)
    """
    if get_db_mode() == "sqlite":
        with sqlite_conn() as conn:
            yield conn
        return

    import psycopg

    url = os.getenv("DATABASE_URL")
    if url:
        conn = psycopg.connect(url)
    else:
        conn = psycopg.connect(
            host=os.getenv("PGHOST", "127.0.0.1"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "rag"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _sqlite_column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r["name"] == col for r in rows)


def init_db() -> None:
    """
    Initializes schema depending on DB_MODE.

    For SQLite, we also do a lightweight migration:
    - ensure users table exists
    - ensure documents/chunks have user_id + notebook columns
    """
    if get_db_mode() != "sqlite":
        # Postgres init not implemented in this MVP.
        return

    with sqlite_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                api_key_hash TEXT NOT NULL UNIQUE,
                label TEXT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                notebook TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                notebook TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NULL,
                embedding BLOB NOT NULL,      -- float32 bytes
                embedding_dim INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(doc_id) REFERENCES documents(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_user_notebook ON documents(user_id, notebook);
            CREATE INDEX IF NOT EXISTS idx_chunks_user_notebook ON chunks(user_id, notebook);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            """
        )

        # If you have older SQLite DB files created before multi-tenant, migrate.
        # We only try ALTERs if the old tables exist and columns are missing.
        for table, col, ddl in [
            ("documents", "user_id", "ALTER TABLE documents ADD COLUMN user_id TEXT;"),
            ("documents", "notebook", "ALTER TABLE documents ADD COLUMN notebook TEXT;"),
            ("chunks", "user_id", "ALTER TABLE chunks ADD COLUMN user_id TEXT;"),
            ("chunks", "notebook", "ALTER TABLE chunks ADD COLUMN notebook TEXT;"),
        ]:
            try:
                if conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                    (table,),
                ).fetchone():
                    if not _sqlite_column_exists(conn, table, col):
                        conn.execute(ddl)
            except Exception:
                # Safe ignore: migrations are best-effort for dev DBs.
                pass
