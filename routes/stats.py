"""
Statistics & Activity Log Routes.

Tag: Stats & Logs
Endpoints:
- POST   /logs                      → Create activity log entry
- GET    /logs                      → Get activity logs (with pagination)
- DELETE /logs                      → Clear all activity logs
- GET    /stats/hourly              → Get hourly detection stats for today (activity_logs)
- GET    /stats/summary             → Get overall detection summary
- POST   /stats/video-detection     → Save distributed video detection counts
- GET    /stats/hourly-detections   → Get per-hour counts from video recordings
- GET    /stats/weekly-detections   → Get per-day counts for current week
"""

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel

from constant_var import debug_info
from dependencies.auth import get_current_user, require_admin
from services.database import get_db
from models.schemas import (
    ActivityLogCreate,
    ActivityLogResponse,
    HourlyStatItem,
    ClassCount,
    VideoDetectionSave,
    WeeklyStatItem,
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
async def create_log(
    body: ActivityLogCreate,
    _user: dict = Depends(get_current_user),
) -> ActivityLogResponse:
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
        "branch_name": body.branch_name,
        "address": body.address,
        "created_at": now,
    }

    result = await db.activity_logs.insert_one(doc)
    loc = f" [{body.branch_name}]" if body.branch_name else ""
    debug_info(f"[Log] Created: {body.type} — {body.source} ({body.total_deteksi} detections){loc}")

    return ActivityLogResponse(
        id=str(result.inserted_id),
        timestamp=doc["timestamp"],
        type=doc["type"],
        source=doc["source"],
        total_deteksi=doc["total_deteksi"],
        counts=body.counts,
        branch_name=body.branch_name,
        address=body.address,
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
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
) -> list[ActivityLogResponse]:
    """Get paginated activity logs."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    query = {}
    if type:
        query["type"] = type
    if branch_name:
        query["branch_name"] = branch_name

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
            branch_name=doc.get("branch_name"),
            address=doc.get("address"),
        ))

    return logs


@router.delete(
    "/logs",
    summary="Clear all activity logs",
    description="Delete all activity log entries from the database.",
)
async def clear_logs(_admin: dict = Depends(require_admin)) -> dict:
    """Clear all activity logs. Admin only."""
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
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
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

    match_stage: dict = {
        "created_at": {"$gte": target, "$lt": next_day},
        "counts": {"$ne": None},
    }
    if branch_name:
        match_stage["branch_name"] = branch_name

    # Aggregation pipeline: group by hour, sum counts
    pipeline = [
        {"$match": match_stage},
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
async def get_summary_stats(
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
) -> dict:
    """Get overall summary of all detection logs."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    match_stage: dict = {"counts": {"$ne": None}}
    if branch_name:
        match_stage["branch_name"] = branch_name

    pipeline = [
        {"$match": match_stage},
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


# =============================================
# VIDEO DETECTION CHART DATA
# =============================================

def _distribute_counts(
    counts: dict,
    recording_start: datetime,
    duration_seconds: float,
) -> list[dict]:
    """Distribute detection counts proportionally across hours covered by video.

    A video recorded from 08:30 for 90 min gets:
    - hour 08: 30/90 fraction
    - hour 09: 60/90 fraction
    """
    recording_end = recording_start + timedelta(seconds=duration_seconds)
    current = recording_start.replace(minute=0, second=0, microsecond=0)

    results = []
    while current < recording_end:
        hour_end = current + timedelta(hours=1)
        overlap_start = max(recording_start, current)
        overlap_end = min(recording_end, hour_end)
        overlap_secs = (overlap_end - overlap_start).total_seconds()

        if overlap_secs > 0:
            frac = overlap_secs / duration_seconds
            results.append({
                "date": current.strftime("%Y-%m-%d"),
                "hour": current.hour,
                "big_vehicle": round(counts["big_vehicle"] * frac),
                "car": round(counts["car"] * frac),
                "pedestrian": round(counts["pedestrian"] * frac),
                "two_wheeler": round(counts["two_wheeler"] * frac),
            })

        current = hour_end

    return results


@router.post(
    "/stats/video-detection",
    summary="Save video detection to hourly chart",
    description="Distribute vehicle counts from a video across hours based on recording start time and duration, then upsert into hourly_detections collection.",
)
async def save_video_detection(
    body: VideoDetectionSave,
    _user: dict = Depends(get_current_user),
) -> dict:
    """Receive detection results and distribute across hourly buckets."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        recording_start = datetime.fromisoformat(body.recording_start)
        if recording_start.tzinfo is None:
            recording_start = recording_start.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid recording_start format. Use ISO 8601.")

    counts = {
        "big_vehicle": body.big_vehicle,
        "car": body.car,
        "pedestrian": body.pedestrian,
        "two_wheeler": body.two_wheeler,
    }

    buckets = _distribute_counts(counts, recording_start, body.duration_seconds)
    now = datetime.now(timezone.utc)

    branch = body.branch_name or ""

    for bucket in buckets:
        doc_id = f"{bucket['date']}_{bucket['hour']:02d}"
        if branch:
            doc_id = f"{doc_id}_{branch}"
        await db.hourly_detections.update_one(
            {"_id": doc_id},
            {
                "$inc": {
                    "big_vehicle": bucket["big_vehicle"],
                    "car": bucket["car"],
                    "pedestrian": bucket["pedestrian"],
                    "two_wheeler": bucket["two_wheeler"],
                },
                "$set": {
                    "date": bucket["date"],
                    "hour": bucket["hour"],
                    "branch_name": branch or None,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    loc = f" [{branch}]" if branch else ""
    debug_info(f"[Stats] Saved {len(buckets)} hourly bucket(s) for recording at {body.recording_start}{loc}")
    return {"success": True, "buckets_saved": len(buckets)}


@router.get(
    "/stats/hourly-detections",
    response_model=list[HourlyStatItem],
    summary="Get hourly detection chart data",
    description="Returns 24 per-hour buckets from video recordings for a given date (based on recording time, not upload time).",
)
async def get_hourly_detections(
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (default: today UTC)"),
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
) -> list[HourlyStatItem]:
    """Get hourly chart data from hourly_detections collection."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        target_date = date
    else:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    query: dict = {"date": target_date}
    if branch_name:
        query["branch_name"] = branch_name

    hour_data: dict[int, dict] = {}
    async for doc in db.hourly_detections.find(query):
        hour_data[doc["hour"]] = {
            "big_vehicle": doc.get("big_vehicle", 0),
            "car": doc.get("car", 0),
            "pedestrian": doc.get("pedestrian", 0),
            "two_wheeler": doc.get("two_wheeler", 0),
        }

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


@router.get(
    "/stats/weekly-detections",
    response_model=list[WeeklyStatItem],
    summary="Get weekly detection chart data",
    description="Returns 7-day (Mon-Sun) totals from video recordings for the current week.",
)
async def get_weekly_detections(
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
) -> list[WeeklyStatItem]:
    """Get per-day totals for current week from hourly_detections collection."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    today = datetime.now(timezone.utc).date()
    monday = today - timedelta(days=today.weekday())

    day_labels = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    results = []

    for i in range(7):
        day_date = monday + timedelta(days=i)
        date_str = day_date.strftime("%Y-%m-%d")

        match_stage: dict = {"date": date_str}
        if branch_name:
            match_stage["branch_name"] = branch_name

        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": None,
                    "big_vehicle": {"$sum": "$big_vehicle"},
                    "car": {"$sum": "$car"},
                    "pedestrian": {"$sum": "$pedestrian"},
                    "two_wheeler": {"$sum": "$two_wheeler"},
                }
            },
        ]

        doc = None
        async for d in db.hourly_detections.aggregate(pipeline):
            doc = d

        if doc:
            total = doc["big_vehicle"] + doc["car"] + doc["pedestrian"] + doc["two_wheeler"]
            results.append(WeeklyStatItem(
                day=day_labels[i],
                date=date_str,
                total=total,
                big_vehicle=doc["big_vehicle"],
                car=doc["car"],
                pedestrian=doc["pedestrian"],
                two_wheeler=doc["two_wheeler"],
            ))
        else:
            results.append(WeeklyStatItem(day=day_labels[i], date=date_str))

    return results


class ReportRow(BaseModel):
    date: str
    hour: int
    branch_name: str | None
    big_vehicle: int
    car: int
    pedestrian: int
    two_wheeler: int


@router.get(
    "/stats/report",
    response_model=list[ReportRow],
    summary="Get report data",
    description="Raw hourly_detections rows for a date range, hour range, and optional branch filter. Frontend handles daily/hourly aggregation.",
    tags=["📊 Stats"],
)
async def get_report_data(
    date_from: str = Query(..., description="Start date YYYY-MM-DD"),
    date_to: str = Query(..., description="End date YYYY-MM-DD"),
    hour_from: int = Query(0, ge=0, le=23, description="Start hour (inclusive, 0–23)"),
    hour_to: int = Query(23, ge=0, le=23, description="End hour (inclusive, 0–23)"),
    branch_name: str | None = Query(None, description="Filter by branch/location name"),
    _user: dict = Depends(get_current_user),
) -> list[ReportRow]:
    """Fetch raw hourly_detections rows for the given filters."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        datetime.strptime(date_from, "%Y-%m-%d")
        datetime.strptime(date_to, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if date_from > date_to:
        raise HTTPException(status_code=400, detail="date_from must be <= date_to.")

    query: dict = {
        "date": {"$gte": date_from, "$lte": date_to},
        "hour": {"$gte": hour_from, "$lte": hour_to},
    }
    if branch_name:
        query["branch_name"] = branch_name

    rows: list[ReportRow] = []
    async for doc in db.hourly_detections.find(query).sort([("date", 1), ("hour", 1)]):
        rows.append(ReportRow(
            date=doc["date"],
            hour=doc["hour"],
            branch_name=doc.get("branch_name"),
            big_vehicle=doc.get("big_vehicle", 0),
            car=doc.get("car", 0),
            pedestrian=doc.get("pedestrian", 0),
            two_wheeler=doc.get("two_wheeler", 0),
        ))
    return rows


@router.delete(
    "/stats/video-detections",
    summary="Delete video detection chart data",
    description=(
        "Hapus data deteksi video dari grafik hourly.\n\n"
        "- Tanpa parameter `date`: hapus **semua** data.\n"
        "- Dengan `date=YYYY-MM-DD`: hapus data **satu hari** saja (berguna jika salah input timestamp)."
    ),
)
async def delete_video_detections(
    date: str | None = Query(None, description="Tanggal yang dihapus (YYYY-MM-DD). Kosongkan untuk hapus semua."),
    _admin: dict = Depends(require_admin),
) -> dict:
    """Delete hourly_detections data — all or by specific date. Admin only."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        result = await db.hourly_detections.delete_many({"date": date})
        debug_info(f"[Stats] Deleted {result.deleted_count} hourly bucket(s) for date {date}")
        return {"success": True, "deleted_count": result.deleted_count, "scope": f"date={date}"}
    else:
        result = await db.hourly_detections.delete_many({})
        debug_info(f"[Stats] Deleted all {result.deleted_count} hourly bucket(s)")
        return {"success": True, "deleted_count": result.deleted_count, "scope": "all"}
