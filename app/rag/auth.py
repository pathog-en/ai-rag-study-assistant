# app/rag/auth.py
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass

from fastapi import Header, HTTPException

from .db import sqlite_conn, get_db_mode


@dataclass(frozen=True)
class UserContext:
    user_id: str
    label: str | None = None


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_user_api_key(label: str | None = None) -> tuple[str, str]:
    """
    Creates a new user row and returns (user_id, api_key).
    API key is returned ONCE; store it securely.
    """
    if get_db_mode() != "sqlite":
        raise RuntimeError("create_user_api_key currently supports sqlite mode only")

    user_id = str(uuid.uuid4())
    api_key = secrets.token_urlsafe(32)
    api_key_hash = _hash_api_key(api_key)

    with sqlite_conn() as conn:
        conn.execute(
            "INSERT INTO users (id, api_key_hash, label) VALUES (?, ?, ?)",
            (user_id, api_key_hash, label),
        )

    return user_id, api_key


def require_user(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> UserContext:
    """
    Dependency that authenticates a user via API key and returns a UserContext.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")

    if get_db_mode() != "sqlite":
        raise HTTPException(status_code=500, detail="Auth currently supports sqlite mode only")

    api_key_hash = _hash_api_key(x_api_key)

    with sqlite_conn() as conn:
        row = conn.execute(
            "SELECT id, label FROM users WHERE api_key_hash = ?",
            (api_key_hash,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return UserContext(user_id=row["id"], label=row["label"])
