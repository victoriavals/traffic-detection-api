"""
MongoDB Database Service — async Motor client and index management.

Collections
-----------
activity_logs
    Detection activity entries written after each successful detection.
    Indexes: ``timestamp`` (desc), ``type``.

hourly_detections
    Hourly vehicle count buckets derived from video recording time ranges.
    Used by the Dashboard hourly/weekly charts.
    Indexes: ``date``, ``(date, hour)`` compound.

video_jobs
    Persistent store for URL-based video processing jobs.
    Jobs are created here by ``POST /video/jobs`` and updated by the
    processing daemon threads via fire-and-forget async writes.
    Indexes: ``created_at`` (desc), ``status``.

users
    Authenticated users (admin/operator). First registered user becomes admin,
    subsequent registrations default to operator.
    Indexes: ``email`` (unique).

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

        # Indexes for hourly_detections (video recording chart data)
        await _db.hourly_detections.create_index("date")
        await _db.hourly_detections.create_index([("date", 1), ("hour", 1)])

        # Indexes for video_jobs (persistent job store)
        await _db.video_jobs.create_index([("created_at", -1)])
        await _db.video_jobs.create_index("status")

        # Index for users (auth) — email must be unique
        await _db.users.create_index("email", unique=True)

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
