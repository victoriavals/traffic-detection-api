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

from jose import jwt
from passlib.context import CryptContext

from constant_var import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from services.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─── Password helpers ────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


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
