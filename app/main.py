import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.db import init_db
from rag.ingest import ingest_markdown
from rag.retrieve import retrieve
from rag.prompt import build_prompt
from rag.chat import generate_answer
from rag.bedrock_status import bedrock_status

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "ai-rag-study-assistant"))


class IngestRequest(BaseModel):
    title: str
    source: str = "local"
    markdown: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class ChatRequest(BaseModel):
    q: str
    top_k: int = 4


@app.on_event("startup")
def startup():
    init_db()


@app.get("/debug/which-app")
def which_app():
    return {"app_name": app.title, "file": __file__}


@app.get("/debug/env")
def debug_env():
    import os
    keys = [
        "DB_MODE",
        "SQLITE_PATH",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "BEDROCK_MODEL_ID",
        "USE_BEDROCK",
    ]
    out = {}
    for k in keys:
        v = os.getenv(k)
        if k == "AWS_ACCESS_KEY_ID" and v:
            out[k] = v[:4] + "..." + v[-4:]  # mask
        else:
            out[k] = v
    return out


@app.get("/bedrock/status")
def bedrock_status_endpoint():
    """
    Single source of truth: uses rag.bedrock_status.bedrock_status()
    (You previously had TWO /bedrock/status endpoints; FastAPI overrides duplicates.)
    """
    return bedrock_status()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/retrieve")
def debug_retrieve(q: str = "RAG vs fine-tuning"):
    hits = retrieve(q, top_k=10)
    return {"q": q, "hits_count": len(hits), "hits_preview": hits[:2]}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    doc_id, chunk_count = ingest_markdown(req.title, req.source, req.markdown)
    return {"doc_id": doc_id, "chunks_added": chunk_count}


@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    hits = retrieve(q, top_k=req.top_k)

    # Guardrail: if retrieval returns nothing, d


class ChatRequest(BaseModel):
    q: str
    top_k: int = 4


@app.post("/chat")
def chat(req: ChatRequest):
    q = req.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is empty")

    hits = retrieve(q, top_k=req.top_k)

    if not hits:
        return {
            "q": q,
            "answer": "Not found in knowledge base.",
            "grounded": False,
            "citations": [],
            "hits_count": 0,
        }

    citations = []
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        citations.append(
            {
                "n": i,
                "doc_title": h.get("doc_title"),
                "doc_source": h.get("doc_source"),
                "chunk_index": h.get("chunk_index"),
                "chunk_id": h.get("chunk_id"),
                "score": float(h.get("score", 0.0)),
            }
        )
        content = (h.get("content") or "").strip()
        context_blocks.append(
            f"[{i}] doc_title={h.get('doc_title')} | source={h.get('doc_source')} | chunk_id={h.get('chunk_id')}\n"
            f"{content}"
        )

    system = (
        "You are a study assistant for AWS certifications.\n"
        "Rules:\n"
        "- Use ONLY the provided Context to answer.\n"
        "- If the Context does not contain enough information, say so and ask a follow-up.\n"
        "- Cite sources inline like [1], [2] based on the Context items.\n"
        "- Do NOT invent facts or sources.\n"
    )

    prompt = f"{system}\n\nQuestion:\n{q}\n\nContext:\n" + "\n\n".join(context_blocks)
    answer = generate_answer(prompt)

    return {
        "q": q,
        "answer": answer,
        "grounded": False,
        "citations": citations,
        "hits_count": len(hits),
    }

