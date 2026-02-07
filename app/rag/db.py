import os
from psycopg import connect
from psycopg.rows import dict_row

def db_conn():
    return connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "rag"),
        password=os.getenv("DB_PASSWORD", "rag"),
        row_factory=dict_row,
    )

def init_db():
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          title TEXT,
          source TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
          chunk_index INT NOT NULL,
          content TEXT NOT NULL,
          token_count INT,
          embedding vector(1024),
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

        conn.commit()
