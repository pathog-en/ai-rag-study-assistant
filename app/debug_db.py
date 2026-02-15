from rag.db import db_conn
from rag.embeddings import embed_texts
from pgvector.psycopg import Vector

conn = db_conn()
cur = conn.cursor()

print("\n--- A) Plain rows (no vector math) ---")
cur.execute("""
SELECT c.chunk_index,
       LEFT(c.content, 40) AS preview,
       d.title AS doc_title
FROM chunks c
JOIN documents d ON d.id = c.doc_id
ORDER BY c.chunk_index
LIMIT 10;
""")
print(cur.fetchall())

print("\n--- B) Self-distance (embedding <=> embedding) ---")
cur.execute("""
SELECT chunk_index, (embedding <=> embedding) AS dist
FROM chunks
ORDER BY chunk_index;
""")
print(cur.fetchall())

print("\n--- C) Distance using embedding from table (no Python param) ---")
cur.execute("""
SELECT c.chunk_index, (c.embedding <=> (SELECT embedding FROM chunks LIMIT 1)) AS dist
FROM chunks c
ORDER BY dist
LIMIT 10;
""")
print(cur.fetchall())

print("\n--- D) Distance with Python parameter (Vector) ---")
raw = embed_texts(["RAG vs fine-tuning"])[0]
qv = Vector([float(x) for x in raw])

cur.execute(
    """
    SELECT chunk_index, (embedding <=> %s) AS dist
    FROM chunks
    ORDER BY dist
    LIMIT 10;
    """,
    (qv,),
)
print(cur.fetchall())

cur.close()
conn.close()
