import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.db import init_db
from rag.ingest import ingest_markdown
from rag.retrieve import retrieve
from rag.chat import generate_answer
from rag.auth import require_user, create_user_api_key, UserContext

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "ai-rag-study-assistant"))


# ----------- Request models (v1 public API) -----------

class V1IngestRequest(BaseModel):
    title: str
    markdown: str
    source: str = "user"
    notebook: str | None = "default"


class V1ChatRequest(BaseModel):
    q: str
    top_k: int = 4
    notebook: str | None = "default"


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
    return {"app_name": app.title, "file": __file__}


@app.get("/debug/env")
def debug_env():
    """
    Safe-ish environment visibility to debug Render config.
    Masks AWS access key (if present). Does NOT expose secrets.
    """
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


# ----------- Admin: create user keys -----------

class AdminCreateUserRequest(BaseModel):
    notebook: str | None = "default"
    label: str | None = None


@app.post("/admin/create_user_key")
def admin_create_user_key(req: AdminCreateUserRequest):
    """
    Creates a new user and returns an API key ONCE.

    Call with header:
      X-Admin-Key: <ADMIN_API_KEY>

    (We keep this intentionally simple: you mint keys and hand them to testers.)
    """
    admin_expected = os.getenv("ADMIN_API_KEY")
    if not admin_expected:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")

    # FastAPI doesn't inject headers here by default; read from env-protected expectation.
    # Use a dependency-style pattern if you prefer, but this is fine for MVP.
    from fastapi import Request
    from fastapi import Body

    # NOTE: We can't type-inject Request without adding it to signature; do a minimal workaround:
    # We'll use a tiny hack: FastAPI provides request via context, but easiest is just add Request param.
    # So we re-define properly below (see admin_create_user_key2).
    raise HTTPException(status_code=500, detail="Use /admin/create_user_key2 endpoint (fixed header handling).")


from fastapi import Header, Request

@app.post("/admin/create_user_key2")
def admin_create_user_key2(
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
        "api_key": api_key,          # show once, store it securely
        "default_notebook": req.notebook or "default",
        "note": "Send this api_key to the user. They must pass it as X-API-Key on /v1/* endpoints.",
    }


# ----------- v1 Public API (multi-tenant) -----------

@app.get("/v1/me")
def v1_me(user: UserContext = Depends(require_user)):
    return {"user_id": user.user_id, "label": user.label}


@app.post("/v1/ingest")
def v1_ingest(req: V1IngestRequest, user: UserContext = Depends(require_user)):
    if not req.markdown.strip():
        raise HTTPException(status_code=400, detail="markdown is empty")

    nb = (req.notebook or "default").strip() or "default"

    try:
        doc_id, chunk_count = ingest_markdown(
            user_id=user.user_id,
            notebook=nb,
            title=req.title,
            source=req.source,
            markdown=req.markdown,
        )
        return {"doc_id": doc_id, "chunks_added": chunk_count, "notebook": nb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {type(e).__name__}: {e}")


@app.post("/v1/chat")
def v1_chat(req: V1ChatRequest, user: UserContext = Depends(require_user)):
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is empty")

    nb = (req.notebook or "default").strip() or "default"

    try:
        hits = retrieve(user_id=user.user_id, notebook=nb, query=q, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieve failed: {type(e).__name__}: {e}")

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

