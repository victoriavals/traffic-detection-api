"""
Branch (Lokasi) Routes.

Tag: Branches
Endpoints:
- GET    /branches            -> List semua lokasi (auth required)
- GET    /branches/{id}       -> Detail satu lokasi (auth required)
- POST   /branches            -> Create lokasi baru (auth required)
- PATCH  /branches/{id}       -> Update lokasi (auth required)
- DELETE /branches/{id}       -> Hapus lokasi (auth required)

Semua user authenticated (admin & operator) boleh CRUD. Setiap mutasi
mencatat email pelaku ke field audit (`created_by`, `updated_by`,
`updated_at`) sekaligus ke `logs/app.log` via `debug_info`.
"""

from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from constant_var import debug_info
from dependencies.auth import get_current_user
from models.schemas import BranchCreate, BranchResponse, BranchUpdate
from services.database import get_db

router = APIRouter(tags=["🏢 Branches"])


def _doc_to_response(doc: dict) -> BranchResponse:
    return BranchResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        address=doc.get("address", ""),
        created_at=doc.get("created_at", ""),
        created_by=doc.get("created_by", ""),
        updated_by=doc.get("updated_by"),
        updated_at=doc.get("updated_at"),
    )


def _parse_object_id(branch_id: str) -> ObjectId:
    try:
        return ObjectId(branch_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID lokasi tidak valid")


@router.get(
    "/branches",
    response_model=list[BranchResponse],
    summary="List semua lokasi",
    description="Mengembalikan semua lokasi terurut alfabet. Butuh autentikasi.",
)
async def list_branches(_user: dict = Depends(get_current_user)) -> list[BranchResponse]:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    rows: list[BranchResponse] = []
    async for doc in db.branches.find({}).sort("name", 1):
        rows.append(_doc_to_response(doc))
    return rows


@router.get(
    "/branches/{branch_id}",
    response_model=BranchResponse,
    summary="Detail satu lokasi",
)
async def get_branch(
    branch_id: str,
    _user: dict = Depends(get_current_user),
) -> BranchResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    doc = await db.branches.find_one({"_id": _parse_object_id(branch_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Lokasi tidak ditemukan")
    return _doc_to_response(doc)


@router.post(
    "/branches",
    response_model=BranchResponse,
    status_code=201,
    summary="Buat lokasi baru",
    description="Semua user authenticated boleh membuat. Nama lokasi harus unik. Email pelaku dicatat di `created_by`.",
)
async def create_branch(
    payload: BranchCreate,
    user: dict = Depends(get_current_user),
) -> BranchResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Nama lokasi tidak boleh kosong atau hanya spasi")

    doc = {
        "name": name,
        "address": payload.address.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": user["email"],
        "updated_by": None,
        "updated_at": None,
    }
    try:
        result = await db.branches.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Nama lokasi sudah dipakai")

    doc["_id"] = result.inserted_id
    debug_info(f"[Branch] Created '{doc['name']}' by {user['email']}")
    return _doc_to_response(doc)


@router.patch(
    "/branches/{branch_id}",
    response_model=BranchResponse,
    summary="Update lokasi",
    description="Semua user authenticated boleh update. Email pelaku & timestamp dicatat di `updated_by` & `updated_at`.",
)
async def update_branch(
    branch_id: str,
    payload: BranchUpdate,
    user: dict = Depends(get_current_user),
) -> BranchResponse:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    update_doc: dict = {}
    if payload.name is not None:
        stripped_name = payload.name.strip()
        if not stripped_name:
            raise HTTPException(status_code=422, detail="Nama lokasi tidak boleh kosong atau hanya spasi")
        update_doc["name"] = stripped_name
    if payload.address is not None:
        update_doc["address"] = payload.address.strip()

    if not update_doc:
        raise HTTPException(status_code=400, detail="Tidak ada field yang diupdate")

    update_doc["updated_by"] = user["email"]
    update_doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    oid = _parse_object_id(branch_id)
    try:
        result = await db.branches.find_one_and_update(
            {"_id": oid},
            {"$set": update_doc},
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Nama lokasi sudah dipakai")

    if not result:
        raise HTTPException(status_code=404, detail="Lokasi tidak ditemukan")

    debug_info(f"[Branch] Updated '{result['name']}' by {user['email']}")
    return _doc_to_response(result)


@router.delete(
    "/branches/{branch_id}",
    status_code=204,
    summary="Hapus lokasi",
    description="Semua user authenticated boleh hapus. Aksi dicatat di `logs/app.log`. Data deteksi historis tetap tersimpan.",
)
async def delete_branch(
    branch_id: str,
    user: dict = Depends(get_current_user),
) -> None:
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    oid = _parse_object_id(branch_id)
    # Fetch first so the audit log captures the (now-gone) name.
    doc = await db.branches.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Lokasi tidak ditemukan")

    await db.branches.delete_one({"_id": oid})
    debug_info(f"[Branch] Deleted '{doc.get('name')}' by {user['email']}")
    return None
