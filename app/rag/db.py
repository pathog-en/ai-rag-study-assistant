# app/rag/db.py
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DB_MODE = os.getenv("DB_MODE", "postgres").lower()
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
    if DB_MODE == "sqlite":
        with sqlite_conn() as conn:
            yield conn
        return

    # Postgres mode (local/dev or if you later add Render Postgres)
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


def init_db() -> None:
    """
    Initializes the database schema depending on DB_MODE.
    """
    if DB_MODE == "sqlite":
        with sqlite_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NULL,
                    embedding BLOB NOT NULL,      -- float32 bytes
                    embedding_dim INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                """
            )
        return

    # Postgres schema init (minimal; assumes you already created tables locally)
    # If you want, we can add full CREATE EXTENSION vector + CREATE TABLE here later.
    return

