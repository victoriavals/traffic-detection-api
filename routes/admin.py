"""
Admin routes — user management. All endpoints require admin role.

Endpoints
---------
- GET    /admin/users                          List all users
- PATCH  /admin/users/{user_id}/role           Promote/demote
- PATCH  /admin/users/{user_id}/status         Enable/disable
- POST   /admin/users/{user_id}/reset-password Set new password
- DELETE /admin/users/{user_id}                Delete user

Guards
------
* **Self-action block** — admin tidak bisa ubah role/status/delete diri
  sendiri (anti-lockout). Reset password DIPERBOLEHKAN ke diri sendiri.
* **Last-admin guard** — sebelum demote/disable/delete admin aktif, hitung
  admin aktif lain. Kalau jadi 0, reject 400.

Status semantics
----------------
* ``active`` — user dapat login dan mengakses endpoint
* ``disabled`` — login ditolak (401, pesan generik); access token aktif
  ditolak di ``get_current_user`` (403). Token akan expire alami (≈15 menit).
"""

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from constant_var import debug_info, debug_warning
from dependencies.auth import require_admin
from models.schemas import (
    ResetPasswordRequest,
    RoleUpdate,
    StatusUpdate,
    UserAdminResponse,
)
from services.auth_service import hash_password
from services.database import get_db

router = APIRouter(prefix="/admin", tags=["🛡️ Admin"])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _doc_to_response(doc: dict) -> UserAdminResponse:
    """Map MongoDB user document to ``UserAdminResponse`` (no ``password_hash``)."""
    return UserAdminResponse(
        id=str(doc["_id"]),
        email=doc["email"],
        name=doc["name"],
        role=doc.get("role", "operator"),
        status=doc.get("status", "active"),
        created_at=doc.get("created_at", ""),
        last_seen_at=doc.get("last_seen_at"),
    )


def _parse_object_id(user_id: str) -> ObjectId:
    """Parse string to ObjectId or raise 400."""
    try:
        return ObjectId(user_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID user tidak valid")


async def _ensure_target_exists_and_not_self(
    db,
    user_id: str,
    current_user: dict,
) -> tuple[ObjectId, dict]:
    """Validate ID, fetch target, block self-action.

    Returns ``(target_oid, target_doc)`` if all checks pass.
    """
    oid = _parse_object_id(user_id)
    if oid == current_user["_id"]:
        raise HTTPException(status_code=400, detail="Tidak bisa mengubah diri sendiri")
    target = await db.users.find_one({"_id": oid})
    if not target:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    return oid, target


async def _ensure_not_last_active_admin(db, target_oid: ObjectId, target_doc: dict) -> None:
    """Reject if removing/demoting/disabling target would leave 0 active admins.

    Only applies when target is currently an *active admin*. Counts OTHER active
    admins (excluding target).
    """
    if target_doc.get("role") != "admin":
        return
    if target_doc.get("status", "active") == "disabled":
        return  # already disabled — won't reduce active count
    other_active_admins = await db.users.count_documents({
        "role": "admin",
        "status": {"$ne": "disabled"},
        "_id": {"$ne": target_oid},
    })
    if other_active_admins == 0:
        raise HTTPException(status_code=400, detail="Minimal harus ada satu admin aktif")


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=list[UserAdminResponse],
    summary="List semua user (admin only)",
)
async def list_users(_admin: dict = Depends(require_admin)) -> list[UserAdminResponse]:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")
    cursor = db.users.find({}).sort("created_at", -1)
    rows: list[UserAdminResponse] = []
    async for doc in cursor:
        rows.append(_doc_to_response(doc))
    return rows


@router.patch(
    "/users/{user_id}/role",
    response_model=UserAdminResponse,
    summary="Ubah role user (admin only)",
)
async def update_role(
    user_id: str,
    body: RoleUpdate,
    admin: dict = Depends(require_admin),
) -> UserAdminResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    oid, target = await _ensure_target_exists_and_not_self(db, user_id, admin)

    # Demoting active admin → check last-admin
    if body.role == "operator":
        await _ensure_not_last_active_admin(db, oid, target)

    await db.users.update_one({"_id": oid}, {"$set": {"role": body.role}})
    updated = await db.users.find_one({"_id": oid})
    debug_info(f"[Admin] {admin['email']} changed role of {target['email']} → {body.role}")
    return _doc_to_response(updated)


@router.patch(
    "/users/{user_id}/status",
    response_model=UserAdminResponse,
    summary="Aktifkan / nonaktifkan user (admin only)",
)
async def update_status(
    user_id: str,
    body: StatusUpdate,
    admin: dict = Depends(require_admin),
) -> UserAdminResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    oid, target = await _ensure_target_exists_and_not_self(db, user_id, admin)

    # Disabling active admin → check last-admin
    if body.status == "disabled":
        await _ensure_not_last_active_admin(db, oid, target)

    await db.users.update_one({"_id": oid}, {"$set": {"status": body.status}})
    updated = await db.users.find_one({"_id": oid})
    debug_info(f"[Admin] {admin['email']} set status of {target['email']} → {body.status}")
    return _doc_to_response(updated)


@router.post(
    "/users/{user_id}/reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reset password user (admin only)",
)
async def reset_password(
    user_id: str,
    body: ResetPasswordRequest,
    admin: dict = Depends(require_admin),
) -> None:
    """Set password baru untuk user lain (atau diri sendiri).

    Tidak ada self-action block — admin boleh reset password diri.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    oid = _parse_object_id(user_id)
    target = await db.users.find_one({"_id": oid})
    if not target:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    await db.users.update_one(
        {"_id": oid},
        {"$set": {"password_hash": hash_password(body.new_password)}},
    )
    debug_warning(f"[Admin] {admin['email']} reset password for {target['email']}")
    return None


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Hapus user (admin only)",
)
async def delete_user(
    user_id: str,
    admin: dict = Depends(require_admin),
) -> None:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    oid, target = await _ensure_target_exists_and_not_self(db, user_id, admin)
    await _ensure_not_last_active_admin(db, oid, target)

    await db.users.delete_one({"_id": oid})
    debug_warning(f"[Admin] {admin['email']} deleted user {target['email']}")
    return None
