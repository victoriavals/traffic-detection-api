"""
Org / Team routes — info org user + rotate invite code.

Endpoints
---------
- ``GET  /org/me``           — info org user yang sedang login (id, anggota,
  invite_code kalau punya akses, is_owner flag).
- ``POST /org/rotate-code``  — generate kode undangan baru untuk org user.
  Hanya owner org (= ``orgs.created_by``) atau admin yang boleh.
- ``GET  /org/list``         — admin-only: list semua org dengan owner + member
  count (dipakai dropdown ``Pindah Org`` di halaman admin Pengguna).

Authorization
-------------
* ``GET /org/me``: setiap user authenticated (lihat org-nya sendiri).
* ``POST /org/rotate-code``: owner org ATAU admin (untuk org-nya sendiri).
  Admin pindah org code milik org lain → bukan via endpoint ini (admin
  tidak punya target_org_id query — selalu rotate untuk org sendiri).
* ``GET /org/list``: admin only.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from constant_var import debug_info
from dependencies.auth import get_current_user, require_admin
from models.schemas import (
    OrgInfoResponse,
    OrgJoinRequest,
    OrgMemberSummary,
    OrgNameUpdate,
    OrgRotateCodeResponse,
    OrgSummary,
)
from services.database import get_db
from services.org_service import (
    ensure_org_exists,
    find_org_by_invite_code,
    get_org,
    is_org_owner,
    rotate_invite_code,
    update_org_name,
)


router = APIRouter(prefix="/org", tags=["👥 Org / Team"])


@router.get(
    "/me",
    response_model=OrgInfoResponse,
    summary="Info org user yang sedang login",
    description="Mengembalikan info org (id, daftar anggota, invite_code, is_owner).",
)
async def get_my_org(user: dict = Depends(get_current_user)) -> OrgInfoResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    user_org_id = user.get("org_id")
    if not user_org_id:
        raise HTTPException(status_code=400, detail="User belum punya org_id (perlu migrasi)")

    # Ensure org doc ada — lazy create untuk org legacy (pre-orgs-collection).
    org = await ensure_org_exists(user_org_id, fallback_owner_id=user["_id"])

    # List members
    members: list[OrgMemberSummary] = []
    async for doc in db.users.find({"org_id": user_org_id}).sort("created_at", 1):
        members.append(OrgMemberSummary(
            id=str(doc["_id"]),
            email=doc["email"],
            name=doc["name"],
            role=doc.get("role", "operator"),
            is_owner=str(doc["_id"]) == str(org.get("created_by")),
        ))

    owner_doc = next((m for m in members if m.is_owner), None)

    return OrgInfoResponse(
        org_id=str(org["_id"]),
        name=org.get("name"),
        owner_email=owner_doc.email if owner_doc else None,
        owner_id=owner_doc.id if owner_doc else None,
        invite_code=org.get("invite_code"),
        member_count=len(members),
        members=members,
        is_owner=is_org_owner(user, org),
        created_at=org.get("created_at"),
    )


@router.patch(
    "",
    response_model=OrgInfoResponse,
    summary="Update nama org (owner / admin only)",
    description=(
        "Owner atau admin bisa rename org-nya. Trim whitespace, max 60 char. "
        "Tidak ada uniqueness constraint — dua org boleh punya nama sama."
    ),
)
async def update_my_org(
    body: OrgNameUpdate,
    user: dict = Depends(get_current_user),
) -> OrgInfoResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    user_org_id = user.get("org_id")
    if not user_org_id:
        raise HTTPException(status_code=400, detail="User belum punya org_id")

    org = await ensure_org_exists(user_org_id, fallback_owner_id=user["_id"])

    if not (is_org_owner(user, org) or user.get("role") == "admin"):
        raise HTTPException(
            status_code=403,
            detail="Hanya pemilik org atau admin yang boleh mengubah nama tim",
        )

    try:
        new_name = await update_org_name(user_org_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    debug_info(f"[Org] {user['email']} renamed org {user_org_id} -> {new_name!r}")

    # Re-fetch org untuk return full info
    org = await get_org(user_org_id)
    members: list[OrgMemberSummary] = []
    async for doc in db.users.find({"org_id": user_org_id}).sort("created_at", 1):
        members.append(OrgMemberSummary(
            id=str(doc["_id"]),
            email=doc["email"],
            name=doc["name"],
            role=doc.get("role", "operator"),
            is_owner=str(doc["_id"]) == str(org.get("created_by")),
        ))
    owner_doc = next((m for m in members if m.is_owner), None)
    return OrgInfoResponse(
        org_id=str(org["_id"]),
        name=org.get("name"),
        owner_email=owner_doc.email if owner_doc else None,
        owner_id=owner_doc.id if owner_doc else None,
        invite_code=org.get("invite_code"),
        member_count=len(members),
        members=members,
        is_owner=is_org_owner(user, org),
        created_at=org.get("created_at"),
    )


@router.post(
    "/rotate-code",
    response_model=OrgRotateCodeResponse,
    summary="Generate kode undangan baru untuk org user",
    description=(
        "Rotate invite_code org user yang sedang login. Hanya owner org "
        "(= user pendiri) atau admin yang boleh — selain itu dapat 403."
    ),
)
async def rotate_my_org_code(user: dict = Depends(get_current_user)) -> OrgRotateCodeResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    user_org_id = user.get("org_id")
    if not user_org_id:
        raise HTTPException(status_code=400, detail="User belum punya org_id")

    org = await ensure_org_exists(user_org_id, fallback_owner_id=user["_id"])

    # Authorization: owner ATAU admin
    if not (is_org_owner(user, org) or user.get("role") == "admin"):
        raise HTTPException(
            status_code=403,
            detail="Hanya pemilik org atau admin yang boleh ganti kode undangan",
        )

    new_code = await rotate_invite_code(user_org_id)
    debug_info(f"[Org] {user['email']} rotated invite_code for org {user_org_id} → {new_code}")
    return OrgRotateCodeResponse(org_id=str(user_org_id), invite_code=new_code)


@router.post(
    "/join",
    response_model=OrgInfoResponse,
    summary="Self-service join org lain via invite_code",
    description=(
        "User existing pindah ke org lain dengan kode undangan. Data domain "
        "(branches/log/deteksi/jobs) yang sudah dibuat user **tetap di org lama** "
        "— tidak ikut pindah. User mulai dengan data kosong di org baru.\n\n"
        "Kalau user adalah owner org lama, org lama tetap exist (orphan owner) "
        "— anggota lain masih bisa lihat data, tapi rotate code hanya bisa via admin."
    ),
)
async def join_org(
    body: OrgJoinRequest,
    user: dict = Depends(get_current_user),
) -> OrgInfoResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    code = body.invite_code.strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="Kode undangan tidak boleh kosong")

    target_org = await find_org_by_invite_code(code)
    if not target_org:
        raise HTTPException(
            status_code=404,
            detail="Kode undangan tidak ditemukan atau sudah tidak berlaku",
        )

    # No-op kalau user sudah di org target
    if user.get("org_id") == target_org["_id"]:
        raise HTTPException(
            status_code=400,
            detail="Anda sudah di tim ini",
        )

    # Update user.org_id
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"org_id": target_org["_id"]}},
    )
    debug_info(
        f"[Org] {user['email']} self-joined org {target_org['_id']} via code {code}"
    )

    # Build response untuk org baru — list members & owner
    members: list[OrgMemberSummary] = []
    async for doc in db.users.find({"org_id": target_org["_id"]}).sort("created_at", 1):
        members.append(OrgMemberSummary(
            id=str(doc["_id"]),
            email=doc["email"],
            name=doc["name"],
            role=doc.get("role", "operator"),
            is_owner=str(doc["_id"]) == str(target_org.get("created_by")),
        ))
    owner_doc = next((m for m in members if m.is_owner), None)
    return OrgInfoResponse(
        org_id=str(target_org["_id"]),
        name=target_org.get("name"),
        owner_email=owner_doc.email if owner_doc else None,
        owner_id=owner_doc.id if owner_doc else None,
        invite_code=target_org.get("invite_code"),
        member_count=len(members),
        members=members,
        is_owner=False,  # Joiner bukan owner (kecuali user sebelumnya emang owner di org target — unlikely)
        created_at=target_org.get("created_at"),
    )


@router.get(
    "/list",
    response_model=list[OrgSummary],
    summary="List semua org (admin only)",
    description=(
        "Daftar semua org dengan owner email + member count. Dipakai untuk "
        "dropdown ``Pindah Org`` di halaman admin Pengguna."
    ),
)
async def list_orgs(_admin: dict = Depends(require_admin)) -> list[OrgSummary]:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database tidak tersedia")

    results: list[OrgSummary] = []
    async for org in db.orgs.find({}).sort("created_at", 1):
        owner_doc = None
        owner_id = org.get("created_by")
        if owner_id:
            owner_doc = await db.users.find_one(
                {"_id": owner_id},
                projection={"email": 1},
            )
        member_count = await db.users.count_documents({"org_id": org["_id"]})
        results.append(OrgSummary(
            org_id=str(org["_id"]),
            name=org.get("name"),
            owner_email=owner_doc["email"] if owner_doc else None,
            member_count=member_count,
            invite_code=org.get("invite_code"),
        ))
    return results
