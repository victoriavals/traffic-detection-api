"""
MongoDB Database Service.

Async MongoDB client using Motor for activity log persistence
and detection statistics aggregation.
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
        await _db.activity_logs.create_index("timestamp", expireAfterSeconds=None)
        await _db.activity_logs.create_index([("timestamp", -1)])
        await _db.activity_logs.create_index("type")

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
