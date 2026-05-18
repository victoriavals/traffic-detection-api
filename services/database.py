"""
MongoDB Database Service — async Motor client and index management.

Multi-tenancy
-------------
Per ``org_id`` scoping diterapkan di 4 collection domain (branches,
activity_logs, hourly_detections, video_jobs). Setiap user dapat
``org_id`` baru saat register; admin = super user, lihat semua org.
Filtering dilakukan oleh :func:`dependencies.auth.scope_filter`.
Migration script ``scripts/migrate_multitenant.py`` mengisi field
``org_id`` ke dokumen legacy (assign ke admin pertama).

Collections
-----------
activity_logs
    Detection activity entries written after each successful detection.
    Indexes: ``timestamp`` (desc), ``type``, ``(org_id, created_at)`` compound.

hourly_detections
    Hourly vehicle count buckets derived from video recording time ranges.
    Used by the Dashboard hourly/weekly charts. Upsert key adalah compound
    ``(org_id, date, hour, branch_name)``.
    Indexes: ``date``, ``(date, hour)`` compound,
    ``(org_id, date, hour, branch_name)`` unique (partial: hanya enforce
    saat ``org_id`` ada, untuk backward-compat dengan legacy docs).

video_jobs
    Persistent store for URL-based video processing jobs.
    Jobs are created here by ``POST /video/jobs`` and updated by the
    processing daemon threads via fire-and-forget async writes.
    Indexes: ``created_at`` (desc), ``status``, ``(org_id, created_at)`` compound.

users
    Authenticated users (admin/operator). First registered user becomes admin,
    subsequent registrations default to operator. Setiap user punya field
    ``org_id`` (ObjectId baru saat register, atau org existing kalau register
    dengan ``invite_code``).
    Indexes: ``email`` (unique).

orgs
    Org/team metadata: ``{_id, invite_code, created_at, created_by}``. ``_id``
    sama dengan ``users.org_id`` (referensi cross-collection). ``created_by``
    adalah ObjectId user pendiri org — yang berhak rotate ``invite_code``.
    Indexes: ``invite_code`` (unique), ``created_by``.

branches
    Lokasi/cabang yang dipakai sebagai tag pada activity logs & stats.
    CRUD via ``/branches`` — semua operasi (read/create/update/delete) terbuka
    untuk semua user authenticated (admin & operator) di org-nya. Admin (super
    user) bisa lihat branches dari semua org. Audit fields (created_by,
    updated_by, updated_at) dicatat di setiap dokumen.
    Indexes: ``(org_id, name)`` unique (partial: hanya enforce saat ``org_id``
    ada — dua org berbeda bebas pakai nama branch yang sama).

Usage
-----
Call :func:`connect_db` once during application startup (FastAPI lifespan).
Then use :func:`get_db` anywhere to obtain the Motor database handle.
Returns ``None`` when the connection has not been established or failed —
callers must guard against ``None`` before performing queries.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from constant_var import MONGO_URI, MONGO_DB_NAME, debug_info, debug_error

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_db() -> None:
    """Initialize MongoDB connection and create indexes."""
    global _client, _db

    try:
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[MONGO_DB_NAME]

        # Verify connection
        await _client.admin.command("ping")
        debug_info(f"[MongoDB] Connected to {MONGO_DB_NAME}")

        # Create indexes for efficient queries
        await _db.activity_logs.create_index([("timestamp", -1)])
        await _db.activity_logs.create_index("type")
        # Multi-tenant filter + sort
        await _db.activity_logs.create_index([("org_id", 1), ("created_at", -1)])

        # Indexes for hourly_detections (video recording chart data)
        await _db.hourly_detections.create_index("date")
        await _db.hourly_detections.create_index([("date", 1), ("hour", 1)])
        # Upsert key & filter: (org_id, date, hour, branch_name)
        # Partial filter → unique constraint hanya berlaku saat org_id ada,
        # backward-compat dengan dokumen legacy yang belum di-migrate.
        await _db.hourly_detections.create_index(
            [("org_id", 1), ("date", 1), ("hour", 1), ("branch_name", 1)],
            unique=True,
            partialFilterExpression={"org_id": {"$exists": True}},
            name="org_date_hour_branch_unique",
        )

        # Indexes for video_jobs (persistent job store)
        await _db.video_jobs.create_index([("created_at", -1)])
        await _db.video_jobs.create_index("status")
        # Multi-tenant filter + sort
        await _db.video_jobs.create_index([("org_id", 1), ("created_at", -1)])

        # Index for users (auth) — email must be unique
        await _db.users.create_index("email", unique=True)

        # Indexes for orgs (multi-tenant teams).
        # invite_code unique global supaya lookup by code one-shot;
        # sparse=False karena setiap org dijamin punya code (di-generate
        # saat create_org_for_user). created_by butuh index untuk owner-check.
        await _db.orgs.create_index("invite_code", unique=True, name="invite_code_unique")
        await _db.orgs.create_index("created_by", name="created_by_idx")

        # Index for branches (lokasi).
        # Legacy single-tenant index ``name_1`` (unique) di-drop kalau masih ada
        # — dua org berbeda boleh punya nama branch sama setelah multi-tenant.
        try:
            await _db.branches.drop_index("name_1")
            debug_info("[MongoDB] Dropped legacy branches.name_1 unique index")
        except Exception:
            pass  # tidak ada — OK
        await _db.branches.create_index(
            [("org_id", 1), ("name", 1)],
            unique=True,
            partialFilterExpression={"org_id": {"$exists": True}},
            name="org_name_unique",
        )

        debug_info("[MongoDB] Indexes created")
    except Exception as e:
        debug_error(f"[MongoDB] Connection failed: {e}")
        _client = None
        _db = None


async def close_db() -> None:
    """Close MongoDB connection."""
    global _client, _db

    if _client:
        _client.close()
        _client = None
        _db = None
        debug_info("[MongoDB] Connection closed")


def get_db() -> Optional[AsyncIOMotorDatabase]:
    """Get database instance. Returns None if not connected."""
    return _db
