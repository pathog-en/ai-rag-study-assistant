import os
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.db import init_db
from rag.ingest import ingest_markdown
from rag.retrieve import retrieve
from rag.chat import generate_answer
from rag.auth import require_user, create_user_api_key, UserContext

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "ai-rag-study-assistant"))


# ----------- Models -----------

class V1IngestRequest(BaseModel):
    title: str
    markdown: str
    source: str = "user"
    notebook: str | None = "default"


class V1ChatRequest(BaseModel):
    q: str
    top_k: int = 4
    notebook: str | None = "default"


class AdminCreateUserRequest(BaseModel):
    label: str | None = None


# ----------- Startup -----------

@app.on_event("startup")
def startup():
    init_db()


# ----------- Health / Debug -----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/which-app")
def which_app():
    return {
        "app_name": app.title,
        "file": _file_,
        ""render_git_commit": os.getenv("RENDER_GIT_COMMIT", None),
        }


@app.get("/debug/env")
def debug_env():
    keys = [
        "DB_MODE",
        "SQLITE_PATH",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "BEDROCK_MODEL_ID",
        "USE_BEDROCK",
        "ADMIN_API_KEY",
    ]
    out = {}
    for k in keys:
        v = os.getenv(k)
        if k in ("AWS_ACCESS_KEY_ID", "ADMIN_API_KEY") and v:
            out[k] = v[:4] + "..." + v[-4:]
        else:
            out[k] = v
    return out


# ----------- Admin -----------

@app.post("/admin/create_user_key")
def admin_create_user_key(
    req: AdminCreateUserRequest,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    admin_expected = os.getenv("ADMIN_API_KEY")
    if not admin_expected:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")

    if not x_admin_key or x_admin_key != admin_expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id, api_key = create_user_api_key(label=req.label)

    return {
        "user_id": user_id,
        "api_key": api_key,  # shown once
        "note": "Send this api_key to the user. They must pass it as X-API-Key on /v1/* endpoints.",
    }


# ----------- v1 Public API -----------

@app.get("/v1/me")
def v1_me(user: UserContext = Depends(require_user)):
    return {"user_id": user.user_id, "label": user.label}


@app.post("/v1/ingest")
def v1_ingest(req: V1IngestRequest, user: UserContext = Depends(require_user)):
    if not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    nb = (req.notebook or "default").strip() or "default"

    doc_id, chunk_count = ingest_markdown(
        user_id=user.user_id,
        notebook=nb,
        title=req.title,
        source=req.source,
        markdown=req.markdown,
    )
    return {"doc_id": doc_id, "chunks_added": chunk_count, "notebook": nb}


@app.post("/v1/chat")
def v1_chat(req: V1ChatRequest, user: UserContext = Depends(require_user)):
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is empty")

    nb = (req.notebook or "default").strip() or "default"

    hits = retrieve(user_id=user.user_id, notebook=nb, query=q, top_k=req.top_k)

    if not hits:
        return {"q": q, "answer": "Not found in knowledge base.", "grounded": False, "citations": [], "hits_count": 0}

    citations = []
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        citations.append(
            {
                "n": i,
                "doc_title": h.get("doc_title"),
                "doc_source": h.get("doc_source"),
                "notebook": h.get("notebook"),
                "chunk_index": h.get("chunk_index"),
                "chunk_id": h.get("chunk_id"),
                "score": float(h.get("score", 0.0)),
            }
        )
        content = (h.get("content") or "").strip()
        context_blocks.append(
            f"[{i}] doc_title={h.get('doc_title')} | source={h.get('doc_source')} | notebook={h.get('notebook')} | chunk_id={h.get('chunk_id')}\n"
            f"{content}"
        )

    system = (
        "You are a study assistant.\n"
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
        "grounded": True,
        "citations": citations,
        "hits_count": len(hits),
        "notebook": nb,
    }


