"""
Authentication Service — password hashing, JWT encode/decode, user repo.

Provides building blocks consumed by ``routes/auth.py`` and
``dependencies/auth.py``. All functions assume MongoDB is available; callers
that may run before DB connect should guard with ``get_db() is not None``.

Key design notes
----------------
* First user registered becomes ``admin``; all subsequent ones default to
  ``operator``. Role can be promoted manually in MongoDB later.
* Access tokens carry ``sub`` (email) + ``role``; refresh tokens carry only
  ``sub``. Both have a ``type`` claim so they cannot be used interchangeably.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt

from constant_var import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from services.database import get_db

_BCRYPT_MAX_BYTES = 72


def _prep_password_bytes(password: str) -> bytes:
    """Encode and truncate to bcrypt's 72-byte password limit."""
    encoded = password.encode("utf-8")
    return encoded[:_BCRYPT_MAX_BYTES]


# ─── Password helpers ────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prep_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prep_password_bytes(plain), hashed.encode("utf-8"))
    except (TypeError, ValueError):
        return False


# ─── JWT helpers ─────────────────────────────────────────────────────────────

def create_access_token(sub: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": sub, "role": role, "type": "access", "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": sub, "type": "refresh", "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Raises jose.JWTError on invalid/expired token."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ─── User repository ─────────────────────────────────────────────────────────

async def get_user_by_email(email: str) -> Optional[dict]:
    db = get_db()
    if db is None:
        return None
    return await db.users.find_one({"email": email.lower()})


async def create_user(email: str, name: str, password: str) -> dict:
    """Insert a new user. First user gets ``admin``, rest get ``operator``."""
    db = get_db()
    if db is None:
        raise RuntimeError("Database not available")

    user_count = await db.users.count_documents({})
    role = "admin" if user_count == 0 else "operator"

    doc = {
        "email": email.lower(),
        "name": name,
        "password_hash": hash_password(password),
        "role": role,
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def user_to_response(user: dict) -> dict:
    """Strip sensitive fields before returning to the client."""
    return {
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


# ─── Last-seen tracking ──────────────────────────────────────────────────────

_LAST_SEEN_THROTTLE_SECONDS = 30


async def maybe_update_last_seen(user: dict) -> None:
    """Throttled update of ``user.last_seen_at``.

    Called from ``get_current_user`` after status check. Updates the field at
    most once per ``_LAST_SEEN_THROTTLE_SECONDS`` to avoid write spam from
    polling endpoints (e.g. Dashboard refresh, video-job status). Fire-and-
    forget — failures are silent. Mutates the in-memory ``user`` dict so
    callers see the fresh timestamp without re-querying.
    """
    db = get_db()
    if db is None:
        return
    now = datetime.now(timezone.utc)
    last_seen_iso = user.get("last_seen_at")
    if last_seen_iso:
        try:
            last_seen = datetime.fromisoformat(last_seen_iso)
            if (now - last_seen).total_seconds() < _LAST_SEEN_THROTTLE_SECONDS:
                return
        except (ValueError, TypeError):
            pass  # malformed → treat as stale, update below
    iso = now.isoformat()
    try:
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_seen_at": iso}},
        )
        user["last_seen_at"] = iso
    except Exception:
        pass  # non-critical; don't break the request
