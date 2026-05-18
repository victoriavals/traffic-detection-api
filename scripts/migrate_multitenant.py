"""
Migration: Single-tenant → Multi-tenant per ``org_id``.

Tujuan
------
Assign field ``org_id`` ke setiap user dan setiap dokumen domain
(branches, activity_logs, hourly_detections, video_jobs) supaya backend bisa
filter data per-org. Strategi: tarik semua user existing + semua data existing
ke org admin pertama (least destructive — admin tidak kehilangan datanya,
operator anggota tim lama tetap melihat data yang sama). Akun baru yang
mendaftar setelah migration akan otomatis dapat ``org_id`` baru via
``services.auth_service.create_user`` dan datanya kosong dari awal.

Idempotent
----------
Boleh dijalankan berkali-kali tanpa efek samping — hanya update dokumen yang
belum punya ``org_id``. Re-run kedua kalinya harus melaporkan ``0 docs updated``.

Cara pakai
----------
.. code-block:: powershell

    cd d:\\computer-vision\\traffic-detection-api
    uv run python scripts/migrate_multitenant.py

Yang dilakukan
--------------
1. Cari admin pertama (sort ``created_at`` ascending). Kalau tidak ada admin
   sama sekali, script abort dengan exit code 1 (perlu register dulu).
2. Pastikan admin punya ``org_id``. Kalau belum, buat ``ObjectId()`` baru dan
   set sebagai ``admin.org_id``.
3. Tarik semua user existing yang belum punya ``org_id`` → set ``org_id =
   admin.org_id``. Asumsi: user existing adalah anggota tim sama.
4. Tarik semua dokumen di 4 collection (branches, activity_logs,
   hourly_detections, video_jobs) yang belum punya ``org_id`` → set ``org_id
   = admin.org_id``.
5. Drop legacy ``branches.name_1`` unique index (kalau ada). Index baru
   (compound ``(org_id, name)`` unique) di-bikin otomatis saat backend start
   via ``services.database.connect_db()``.
6. Print summary jumlah dokumen yang ter-update per collection.

Rollback
--------
Sebelum jalankan, backup dulu:

.. code-block:: powershell

    mongodump --uri="$env:MONGO_URI" --db=traffic_counter --out=backup/pre-multitenant

Kalau hasil migrasi salah, restore dengan ``mongorestore`` ke backup itu.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows supaya em-dash & karakter non-ASCII di summary
# tidak crash dengan ``UnicodeEncodeError: 'charmap' codec can't encode...``.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Ensure parent project root is on sys.path so we can import constant_var.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient


# Use the same env reading pattern as constant_var (importing it ensures
# .env is loaded if dotenv is used). We re-read for the script to keep this
# tool standalone and small.
try:
    from constant_var import MONGO_URI, MONGO_DB_NAME  # noqa: E402
except Exception:  # pragma: no cover — defensive fallback
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "traffic_counter")


DOMAIN_COLLECTIONS = (
    "branches",
    "activity_logs",
    "hourly_detections",
    "video_jobs",
)


import secrets
import string


def _generate_invite_code() -> str:
    """Sama dengan ``services.org_service.generate_invite_code`` — duplikat
    untuk hindari import side-effects (org_service → database → loadtime).
    """
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(chars) for _ in range(6))
    return f"ORG-{suffix}"


async def migrate() -> int:
    client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    db = client[MONGO_DB_NAME]

    # Sanity: ensure server reachable
    await client.admin.command("ping")
    print(f"[migrate] Connected to {MONGO_DB_NAME}")

    # 1. Find first admin
    admin = await db.users.find_one({"role": "admin"}, sort=[("created_at", 1)])
    if not admin:
        print("[migrate] ERROR: Tidak ada user admin di database.")
        print("[migrate]        Daftar satu user lewat /auth/register dulu (user pertama otomatis jadi admin).")
        return 1
    print(f"[migrate] Admin pertama: {admin['email']} (created_at={admin.get('created_at')})")

    # 2. Ensure admin has org_id
    admin_org_id = admin.get("org_id")
    if not admin_org_id:
        admin_org_id = ObjectId()
        await db.users.update_one({"_id": admin["_id"]}, {"$set": {"org_id": admin_org_id}})
        print(f"[migrate] Created new org_id for admin: {admin_org_id}")
    else:
        print(f"[migrate] Admin existing org_id: {admin_org_id}")

    # 3. Backfill users yang belum punya org_id
    users_result = await db.users.update_many(
        {"org_id": {"$exists": False}},
        {"$set": {"org_id": admin_org_id}},
    )
    print(f"[migrate] users updated: {users_result.modified_count}")

    # 4. Backfill 4 collection domain
    totals: dict[str, int] = {}
    for col in DOMAIN_COLLECTIONS:
        result = await db[col].update_many(
            {"org_id": {"$exists": False}},
            {"$set": {"org_id": admin_org_id}},
        )
        totals[col] = result.modified_count
        print(f"[migrate] {col} updated: {result.modified_count}")

    # 5. Drop legacy branches.name_1 unique index (kalau masih ada)
    try:
        await db.branches.drop_index("name_1")
        print("[migrate] Dropped legacy branches.name_1 unique index")
    except Exception:
        print("[migrate] Legacy branches.name_1 index sudah tidak ada — OK")

    # 5b. Ensure ``orgs`` doc untuk admin org (idempotent — skip kalau ada)
    existing_org = await db.orgs.find_one({"_id": admin_org_id})
    if existing_org:
        print(f"[migrate] orgs doc untuk admin org sudah ada (code={existing_org.get('invite_code')})")
    else:
        # Generate code yang dijamin unik (retry kalau collision)
        for _ in range(5):
            code = _generate_invite_code()
            taken = await db.orgs.find_one({"invite_code": code}, projection={"_id": 1})
            if not taken:
                break
        else:
            raise RuntimeError("Gagal generate invite_code unik untuk admin org")
        from datetime import datetime, timezone
        admin_name = (admin.get("name") or admin["email"].split("@")[0]).strip()
        default_name = f"Tim {admin_name}"[:60]
        await db.orgs.insert_one({
            "_id": admin_org_id,
            "name": default_name,
            "invite_code": code,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": admin["_id"],
        })
        print(f"[migrate] Created orgs doc for admin org: name={default_name!r}, invite_code={code}")

    # 5c. Backfill name untuk orgs yang belum punya name (idempotent)
    orgs_without_name = await db.orgs.count_documents({"name": {"$exists": False}})
    if orgs_without_name > 0:
        backfilled = 0
        async for org_doc in db.orgs.find({"name": {"$exists": False}}):
            owner_id = org_doc.get("created_by")
            owner = await db.users.find_one({"_id": owner_id}) if owner_id else None
            if owner:
                owner_display = (owner.get("name") or owner["email"].split("@")[0]).strip()
                default_name = f"Tim {owner_display}"[:60]
            else:
                default_name = f"Tim {str(org_doc['_id'])[-6:]}"
            await db.orgs.update_one(
                {"_id": org_doc["_id"]},
                {"$set": {"name": default_name}},
            )
            backfilled += 1
        print(f"[migrate] Backfilled name untuk {backfilled} org tanpa name")
    else:
        print("[migrate] Semua orgs sudah punya name — OK")

    # 6. Summary
    total_docs = sum(totals.values())
    print("")
    print("[migrate] ── Summary ──────────────────────────────────────────────")
    print(f"[migrate]   admin.org_id        : {admin_org_id}")
    print(f"[migrate]   users updated       : {users_result.modified_count}")
    for col, count in totals.items():
        print(f"[migrate]   {col:<22}: {count}")
    print(f"[migrate]   total domain docs   : {total_docs}")
    print("[migrate] ─────────────────────────────────────────────────────────")
    if users_result.modified_count == 0 and total_docs == 0:
        print("[migrate] Tidak ada perubahan — semua dokumen sudah punya org_id (idempotent OK).")
    else:
        print("[migrate] Migration selesai. Restart backend supaya index compound dibuat ulang.")
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(migrate())
    except KeyboardInterrupt:
        print("[migrate] Aborted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"[migrate] FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
