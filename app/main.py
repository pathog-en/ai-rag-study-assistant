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


@app.on_event("startup")
def startup():
    init_db()


@app.get("/bedrock/status")
def bedrock_status_endpoint():
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

    # Guardrail: if retrieval returns nothing, do not generate
    if not hits:
        return {
            "answer": "Not found in knowledge base.",
            "grounded": False,
            "top_score": None,
            "sources": [],
        }

    top_score = float(hits[0].get("score", 0.0))

    prompt = build_prompt(q, hits)
    answer = generate_answer(prompt)

    sources = [
        {
            "doc_title": h["doc_title"],
            "doc_source": h["doc_source"],
            "chunk_index": h["chunk_index"],
            "score": float(h["score"]),
            "chunk_id": h["chunk_id"],
        }
        for h in hits
    ]

    return {
        "answer": answer,
        "grounded": True,
        "top_score": top_score,
        "sources": sources,
    }


    return {"answer": answer, "sources": sources}
