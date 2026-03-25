"""
Statistics & Activity Log Routes.

Tag: Stats & Logs
Endpoints:
- POST   /logs           → Create activity log entry
- GET    /logs           → Get activity logs (with pagination)
- DELETE /logs           → Clear all activity logs
- GET    /stats/hourly   → Get hourly detection stats for today
- GET    /stats/summary  → Get overall detection summary
"""

from datetime import datetime, timezone, timedelta

from bson import ObjectId
from fastapi import APIRouter, Query, HTTPException

from constant_var import debug_info, debug_error
from services.database import get_db
from models.schemas import (
    ActivityLogCreate,
    ActivityLogResponse,
    HourlyStatItem,
    ClassCount,
)

router = APIRouter(tags=["📊 Stats & Logs"])


# =============================================
# ACTIVITY LOGS
# =============================================

@router.post(
    "/logs",
    response_model=ActivityLogResponse,
    summary="Create activity log",
    description="Save a new detection activity log entry to the database.",
)
async def create_log(body: ActivityLogCreate) -> ActivityLogResponse:
    """Create a new activity log entry."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    now = datetime.now(timezone.utc)

    doc = {
        "timestamp": now.isoformat(),
        "type": body.type,
        "source": body.source,
        "total_deteksi": body.total_deteksi,
        "counts": body.counts.model_dump() if body.counts else None,
        "created_at": now,
    }

    result = await db.activity_logs.insert_one(doc)
    debug_info(f"[Log] Created: {body.type} — {body.source} ({body.total_deteksi} detections)")

    return ActivityLogResponse(
        id=str(result.inserted_id),
        timestamp=doc["timestamp"],
        type=doc["type"],
        source=doc["source"],
        total_deteksi=doc["total_deteksi"],
        counts=body.counts,
    )


@router.get(
    "/logs",
    response_model=list[ActivityLogResponse],
    summary="Get activity logs",
    description="Retrieve activity logs with pagination, sorted by newest first.",
)
async def get_logs(
    limit: int = Query(50, ge=1, le=200, description="Max number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    type: str | None = Query(None, description="Filter by type (Gambar, Video, RTSP, EZVIZ)"),
) -> list[ActivityLogResponse]:
    """Get paginated activity logs."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    query = {}
    if type:
        query["type"] = type

    cursor = db.activity_logs.find(query).sort("created_at", -1).skip(offset).limit(limit)

    logs = []
    async for doc in cursor:
        counts = None
        if doc.get("counts"):
            counts = ClassCount(**doc["counts"])

        logs.append(ActivityLogResponse(
            id=str(doc["_id"]),
            timestamp=doc["timestamp"],
            type=doc["type"],
            source=doc["source"],
            total_deteksi=doc["total_deteksi"],
            counts=counts,
        ))

    return logs


@router.delete(
    "/logs",
    summary="Clear all activity logs",
    description="Delete all activity log entries from the database.",
)
async def clear_logs() -> dict:
    """Clear all activity logs."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    result = await db.activity_logs.delete_many({})
    debug_info(f"[Log] Cleared {result.deleted_count} logs")

    return {"success": True, "deleted_count": result.deleted_count}


# =============================================
# HOURLY STATS
# =============================================

@router.get(
    "/stats/hourly",
    response_model=list[HourlyStatItem],
    summary="Get hourly detection stats",
    description="Aggregate detection counts per hour for today (UTC). Returns 24 hours (00:00–23:00).",
)
async def get_hourly_stats(
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (default: today UTC)"),
) -> list[HourlyStatItem]:
    """Get hourly aggregated stats for a given date."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    # Parse target date
    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    else:
        target = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    next_day = target + timedelta(days=1)

    # Aggregation pipeline: group by hour, sum counts
    pipeline = [
        {
            "$match": {
                "created_at": {"$gte": target, "$lt": next_day},
                "counts": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": {"$hour": "$created_at"},
                "big_vehicle": {"$sum": "$counts.big_vehicle"},
                "car": {"$sum": "$counts.car"},
                "pedestrian": {"$sum": "$counts.pedestrian"},
                "two_wheeler": {"$sum": "$counts.two_wheeler"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    hour_data: dict[int, dict] = {}
    async for doc in db.activity_logs.aggregate(pipeline):
        hour_data[doc["_id"]] = {
            "big_vehicle": doc["big_vehicle"],
            "car": doc["car"],
            "pedestrian": doc["pedestrian"],
            "two_wheeler": doc["two_wheeler"],
        }

    # Build complete 24-hour result
    result = []
    for h in range(24):
        data = hour_data.get(h, {})
        result.append(HourlyStatItem(
            hour=f"{h:02d}:00",
            big_vehicle=data.get("big_vehicle", 0),
            car=data.get("car", 0),
            pedestrian=data.get("pedestrian", 0),
            two_wheeler=data.get("two_wheeler", 0),
        ))

    return result


# =============================================
# SUMMARY STATS
# =============================================

@router.get(
    "/stats/summary",
    summary="Get overall detection summary",
    description="Get total detection counts across all logs, plus total log count.",
)
async def get_summary_stats() -> dict:
    """Get overall summary of all detection logs."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    pipeline = [
        {"$match": {"counts": {"$ne": None}}},
        {
            "$group": {
                "_id": None,
                "big_vehicle": {"$sum": "$counts.big_vehicle"},
                "car": {"$sum": "$counts.car"},
                "pedestrian": {"$sum": "$counts.pedestrian"},
                "two_wheeler": {"$sum": "$counts.two_wheeler"},
                "total_logs": {"$sum": 1},
                "total_deteksi": {"$sum": "$total_deteksi"},
            }
        },
    ]

    result = None
    async for doc in db.activity_logs.aggregate(pipeline):
        result = doc

    if not result:
        return {
            "counts": {"big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0, "total": 0},
            "total_logs": 0,
            "total_deteksi": 0,
        }

    total = result["big_vehicle"] + result["car"] + result["pedestrian"] + result["two_wheeler"]

    return {
        "counts": {
            "big_vehicle": result["big_vehicle"],
            "car": result["car"],
            "pedestrian": result["pedestrian"],
            "two_wheeler": result["two_wheeler"],
            "total": total,
        },
        "total_logs": result["total_logs"],
        "total_deteksi": result["total_deteksi"],
    }
