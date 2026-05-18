"""
Org / Team service — invite code generation, lookup, dan rotation.

Konsep
------
* Setiap user punya ``org_id`` (ObjectId) — multi-tenant scope key.
* ``orgs`` collection menyimpan metadata org: ``{_id, invite_code, created_at, created_by}``.
* **Owner** org adalah user dengan ``_id == org.created_by`` — permanent, tidak
  berubah meskipun user-user lain join/leave. Hanya owner (atau admin) yang
  boleh rotate invite_code.
* Invite code: format ``ORG-XXXXXX`` (6 random alphanumeric uppercase) untuk
  mudah diketik manual. Unique global (sparse index).

Lifecycle
---------
1. **Register tanpa invite_code** → :func:`create_org_for_user` dipanggil di
   ``services.auth_service.create_user``: bikin ``orgs`` doc baru, kembalikan
   ``org_id`` yang sama dipakai ke ``user.org_id``. ``created_by`` = user._id.
2. **Register dengan invite_code** → :func:`find_org_by_invite_code` lookup,
   user dapat ``org_id`` org existing, ``orgs`` doc tidak di-insert ulang.
3. **Rotate code** → :func:`rotate_invite_code` cek caller owner (atau admin),
   generate code baru, update ``orgs.invite_code``.

Migration
---------
Untuk org legacy (admin org pertama, hasil migration single→multi-tenant),
``scripts/migrate_multitenant.py`` upsert ``orgs`` doc dengan ``_id`` = admin
org_id, ``created_by`` = admin._id, dan invite_code di-generate baru.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from services.database import get_db


_INVITE_CODE_PREFIX = "ORG-"
_INVITE_CODE_LEN = 6
_INVITE_CODE_ALPHABET = string.ascii_uppercase + string.digits
_INVITE_CODE_MAX_RETRY = 5


def generate_invite_code() -> str:
    """Generate kode undangan baru — format ``ORG-XXXXXX``.

    6 random alphanumeric uppercase. Memakai ``secrets`` untuk randomness
    yang cryptographically strong (hindari predictable code).
    """
    suffix = "".join(secrets.choice(_INVITE_CODE_ALPHABET) for _ in range(_INVITE_CODE_LEN))
    return f"{_INVITE_CODE_PREFIX}{suffix}"


async def _generate_unique_code(db) -> str:
    """Generate code yang dipastikan unik (retry beberapa kali kalau collision)."""
    for _ in range(_INVITE_CODE_MAX_RETRY):
        code = generate_invite_code()
        existing = await db.orgs.find_one({"invite_code": code}, projection={"_id": 1})
        if not existing:
            return code
    # Highly unlikely (~1 in 36^6 collision per attempt)
    raise RuntimeError("Gagal generate invite_code unik setelah beberapa percobaan")


async def create_org_for_user(user_id: ObjectId, default_name: Optional[str] = None) -> ObjectId:
    """Bikin ``orgs`` doc baru dengan owner = user_id. Return org_id (= orgs._id).

    Dipanggil saat register tanpa invite_code — user dapat org sendiri.
    ``default_name`` diisi caller (biasanya ``"Tim {nama_user}"``) supaya
    org punya display name yang masuk akal sejak awal.
    """
    db = get_db()
    if db is None:
        raise RuntimeError("Database not available")

    org_id = ObjectId()
    code = await _generate_unique_code(db)
    doc = {
        "_id": org_id,
        "invite_code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": user_id,
    }
    if default_name:
        doc["name"] = default_name.strip()[:60]
    await db.orgs.insert_one(doc)
    return org_id


async def find_org_by_invite_code(code: str) -> Optional[dict]:
    """Cari org berdasarkan invite_code. Return doc atau None."""
    db = get_db()
    if db is None:
        return None
    return await db.orgs.find_one({"invite_code": code.strip()})


async def get_org(org_id: ObjectId) -> Optional[dict]:
    """Ambil dokumen org berdasarkan ``_id`` (= org_id)."""
    db = get_db()
    if db is None:
        return None
    return await db.orgs.find_one({"_id": org_id})


async def ensure_org_exists(org_id: ObjectId, fallback_owner_id: ObjectId) -> dict:
    """Pastikan ``orgs`` doc ada untuk ``org_id``. Bikin lazily kalau belum.

    Dipakai untuk org legacy yang user_doc-nya punya ``org_id`` tapi
    ``orgs`` collection belum punya entry-nya (mis. data pre-migration).
    """
    db = get_db()
    if db is None:
        raise RuntimeError("Database not available")
    doc = await db.orgs.find_one({"_id": org_id})
    if doc:
        return doc
    code = await _generate_unique_code(db)
    new_doc = {
        "_id": org_id,
        "invite_code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": fallback_owner_id,
    }
    try:
        await db.orgs.insert_one(new_doc)
    except DuplicateKeyError:
        # Race: org baru saja di-insert oleh request paralel
        existing = await db.orgs.find_one({"_id": org_id})
        if existing:
            return existing
        raise
    return new_doc


async def rotate_invite_code(org_id: ObjectId) -> str:
    """Generate code baru untuk org dan update doc-nya. Return code baru.

    Authorization (owner / admin) dicek di route handler, bukan di sini.
    """
    db = get_db()
    if db is None:
        raise RuntimeError("Database not available")
    code = await _generate_unique_code(db)
    await db.orgs.update_one(
        {"_id": org_id},
        {"$set": {"invite_code": code, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    return code


async def update_org_name(org_id: ObjectId, name: str) -> str:
    """Update ``orgs.name``. Trim + clamp ke 60 char. Return nama final.

    Authorization (owner / admin) dicek di route handler, bukan di sini.
    """
    db = get_db()
    if db is None:
        raise RuntimeError("Database not available")
    cleaned = name.strip()[:60]
    if not cleaned:
        raise ValueError("Nama tim tidak boleh kosong")
    await db.orgs.update_one(
        {"_id": org_id},
        {"$set": {"name": cleaned, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    return cleaned


def is_org_owner(user: dict, org: dict) -> bool:
    """Cek apakah user adalah owner org (= org.created_by == user._id)."""
    owner_id = org.get("created_by")
    if not owner_id:
        return False
    return str(owner_id) == str(user.get("_id"))
