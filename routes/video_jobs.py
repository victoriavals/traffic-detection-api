"""
Video Job Routes — URL-based async video processing with MongoDB persistence.

Architecture
------------
Jobs are processed in daemon threads and tracked in two stores:

1. **In-memory dict** ``_jobs`` — fast, transient. Lost on server restart.
2. **MongoDB collection** ``video_jobs`` — persistent. Survives restarts.

Every state change is written to MongoDB via :func:`_persist_job` using
``asyncio.run_coroutine_threadsafe`` so the sync processing thread can
schedule async DB writes without blocking.

On **server startup**, :func:`recover_interrupted_jobs` marks any
``pending / downloading / processing`` documents as ``error`` — these are
jobs that were killed mid-flight by the previous server shutdown and
cannot be resumed.

On **GET /video/jobs/{id}**, the endpoint checks memory first (O(1) dict
lookup), then falls back to MongoDB for jobs that completed before the last
restart. This means the frontend can always reconnect to a finished job
even after a server restart, as long as it has the ``job_id`` in
``localStorage``.

Endpoints
---------
- ``POST /video/jobs``        — Submit URL video processing job (202 Accepted)
- ``GET  /video/jobs/{id}``   — Poll status / result (memory → MongoDB)
- ``GET  /video/jobs``        — List all jobs (memory + MongoDB, dedup)
- ``DELETE /video/jobs/{id}`` — Cancel / delete from both stores

Scalability notes
-----------------
- ``_MAX_JOBS = 50`` caps in-memory jobs; oldest is evicted (FIFO) when full.
- MongoDB has no cap — completed jobs accumulate and are queryable historically.
- ``_persist_job`` is fire-and-forget; DB write failures are logged but never
  propagate to the user.
- Progress updates are written to MongoDB every ~2 s during processing.
  For very high throughput, consider throttling these writes.
"""

import asyncio
import os
import re
import time
import uuid
import threading
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import cv2
import numpy as np
import supervision as sv
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from constant_var import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_SIZE,
    DEFAULT_LINE_START_X,
    DEFAULT_LINE_START_Y,
    DEFAULT_LINE_END_X,
    DEFAULT_LINE_END_Y,
    CLASS_NAMES,
    TEMP_DIR,
    debug_info,
    debug_error,
)
from models.schemas import VideoJobRequest, VideoJobStatus, ClassCount
from services.detector_service import DetectorService
from services.database import get_db
from utils.pedestrian_filter import filter_pedestrian_on_vehicle


router = APIRouter(prefix="/video", tags=["🎬 Video Processing"])

# Service instances (shared, thread-safe for inference)
_detector: DetectorService = DetectorService()

# ─── Event Loop Reference (for thread → async DB writes) ──────────────────────

_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the running asyncio event loop so daemon threads can schedule DB writes."""
    global _event_loop
    _event_loop = loop


# ─── In-memory Job Store ───────────────────────────────────────────────────────

@dataclass
class _Job:
    """Internal mutable job state (not serialized directly)."""
    job_id: str
    video_url: str = ""
    status: Literal["pending", "downloading", "processing", "done", "error"] = "pending"
    progress: float = 0.0
    message: str = "Job terdaftar"
    counts: ClassCount | None = None
    video_info: dict | None = None
    inference_config: dict | None = None
    error: str | None = None
    temp_path: str | None = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()

# Maximum number of jobs to keep in memory
_MAX_JOBS: int = 50


# ─── URL Normalizer ────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Convert sharing URLs to direct download URLs.

    Supports:
    - Google Drive sharing links → direct download
    - Dropbox sharing links → direct download
    - All other URLs returned as-is.

    Args:
        url: Raw URL string from user.

    Returns:
        Direct download URL string.
    """
    url = url.strip()

    # ── Google Drive ──────────────────────────────────────────────────────────
    # Sharing: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    # Open:    https://drive.google.com/open?id=FILE_ID
    gd_file_match = re.search(r"drive\.google\.com/file/d/([^/?&]+)", url)
    gd_open_match = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
    gdrive_id: str | None = None

    if gd_file_match:
        gdrive_id = gd_file_match.group(1)
    elif gd_open_match:
        gdrive_id = gd_open_match.group(1)

    if gdrive_id:
        return f"https://drive.usercontent.google.com/download?id={gdrive_id}&export=download&confirm=t"

    # ── Dropbox ───────────────────────────────────────────────────────────────
    # Change dl=0 → dl=1 for direct download
    if "dropbox.com" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        qs["dl"] = ["1"]
        new_query = urlencode({k: v[0] for k, v in qs.items()})
        return urlunparse(parsed._replace(query=new_query))

    return url


# ─── Background Worker ────────────────────────────────────────────────────────

def _run_job(job: _Job, request: VideoJobRequest) -> None:
    """Full job lifecycle: stream-download → sample frames → detect → update job.

    Runs in a daemon thread. Updates `job` state in place.

    Args:
        job: Mutable _Job object to update.
        request: VideoJobRequest with all processing parameters.
    """
    temp_path: str = str(TEMP_DIR / f"urljob_{job.job_id[:8]}.mp4")
    job.temp_path = temp_path
    download_url: str = _normalize_url(request.url)

    try:
        # ── Phase 1: Download ─────────────────────────────────────────────────
        _update_job(job, status="downloading", progress=0.0, message="Mengunduh video dari URL...")
        debug_info(f"[JOB/{job.job_id[:8]}] Downloading: {download_url}")

        with httpx.Client(follow_redirects=True, timeout=None) as client:
            with client.stream("GET", download_url) as response:
                response.raise_for_status()
                total_bytes: int = int(response.headers.get("content-length", 0))
                downloaded: int = 0

                with open(temp_path, "wb") as fp:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):  # 1 MB chunks
                        fp.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes > 0:
                            pct: float = min(downloaded / total_bytes * 30.0, 30.0)  # 0–30%
                            size_mb: float = downloaded / (1024 * 1024)
                            _update_job(job, progress=pct, message=f"Mengunduh... {size_mb:.1f} MB")

        debug_info(f"[JOB/{job.job_id[:8]}] Download complete: {os.path.getsize(temp_path) / 1e6:.1f} MB")

        # ── Phase 2: Open video ───────────────────────────────────────────────
        _update_job(job, status="processing", progress=30.0, message="Membuka video...")
        cap: cv2.VideoCapture = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError("Tidak dapat membuka file video yang diunduh")

        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec: float = total_frames / video_fps if video_fps > 0 else 0

        debug_info(
            f"[JOB/{job.job_id[:8]}] Video: {width}x{height} @ {video_fps:.1f} FPS, "
            f"{total_frames} frames ({duration_sec:.1f}s)"
        )

        # ── Phase 3: Setup tracker & line zone ────────────────────────────────
        tracker: sv.ByteTrack = sv.ByteTrack(
            lost_track_buffer=30,
            frame_rate=int(video_fps),
        )
        line_start: sv.Point = sv.Point(int(width * request.line_start_x), int(height * request.line_start_y))
        line_end: sv.Point = sv.Point(int(width * request.line_end_x), int(height * request.line_end_y))
        line_zone: sv.LineZone = sv.LineZone(start=line_start, end=line_end)

        counts: dict[str, int] = {
            "big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0,
        }
        counted_ids: set[int] = set()
        frame_count: int = 0
        process_start: float = time.time()
        last_progress_update: float = 0.0

        # ── Phase 4: Process ALL frames sequentially ──────────────────────────
        # ByteTrack requires consecutive frames for accurate tracking and
        # line-crossing detection. Frame sampling breaks tracker ID continuity.
        try:
            while True:
                ret: bool
                frame: np.ndarray
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect → filter → track → count
                detections: sv.Detections = _detector.detect(
                    frame, request.confidence, request.iou, request.model_size
                )
                detections = filter_pedestrian_on_vehicle(detections)
                detections = tracker.update_with_detections(detections)

                crossed_in: np.ndarray
                crossed_out: np.ndarray
                crossed_in, crossed_out = line_zone.trigger(detections)

                for i in range(len(detections)):
                    if crossed_in[i] or crossed_out[i]:
                        t_id: int = int(detections.tracker_id[i])
                        if t_id in counted_ids:
                            continue
                        counted_ids.add(t_id)
                        cls_name: str = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                        key: str = cls_name.replace("-", "_")
                        if key in counts:
                            counts[key] += 1

                # Update progress every ~2 seconds to avoid excessive locking
                elapsed: float = time.time() - process_start
                if elapsed - last_progress_update >= 2.0:
                    last_progress_update = elapsed
                    processing_pct: float = 30.0 + (frame_count / max(total_frames, 1)) * 70.0
                    fps_proc: float = frame_count / max(elapsed, 0.001)
                    remaining_frames: int = max(0, total_frames - frame_count)
                    eta_sec: int = int(remaining_frames / fps_proc) if fps_proc > 0 else 0
                    eta_str: str = f"{eta_sec // 60}m {eta_sec % 60}s" if eta_sec > 60 else f"{eta_sec}s"
                    current_count: int = sum(counts.values())

                    _update_job(
                        job,
                        progress=min(processing_pct, 99.0),
                        message=(
                            f"Frame {frame_count}/{total_frames} "
                            f"({fps_proc:.1f} FPS) • "
                            f"{current_count} kendaraan • "
                            f"ETA {eta_str}"
                        ),
                    )

        finally:
            cap.release()

        # ── Phase 5: Finalize ─────────────────────────────────────────────────
        process_time: float = time.time() - process_start
        total_count: int = sum(counts.values())

        _update_job(
            job,
            status="done",
            progress=100.0,
            message=f"Selesai! {total_count} kendaraan terdeteksi dari {frame_count} frame.",
            counts=ClassCount(
                big_vehicle=counts["big_vehicle"],
                car=counts["car"],
                pedestrian=counts["pedestrian"],
                two_wheeler=counts["two_wheeler"],
                total=total_count,
            ),
            video_info={
                "resolution": f"{width}x{height}",
                "fps": round(video_fps, 1),
                "total_frames": total_frames,
                "frames_processed": frame_count,
                "duration_seconds": round(duration_sec, 1),
                "processing_time_seconds": round(process_time, 2),
            },
            inference_config={
                "model": f"YOLOv11{'s' if request.model_size == 'SMALL' else 'm'}",
                "device": _detector.get_device_info(),
                "image_size": 640,
                "line_start": [request.line_start_x, request.line_start_y],
                "line_end": [request.line_end_x, request.line_end_y],
            },
        )
        debug_info(f"[JOB/{job.job_id[:8]}] Done: {total_count} vehicles in {process_time:.1f}s")

    except Exception as exc:
        debug_error(f"[JOB/{job.job_id[:8]}] Error: {exc}")
        _update_job(job, status="error", message="Job gagal", error=str(exc))

    finally:
        # Clean up downloaded temp file
        if job.temp_path and os.path.exists(job.temp_path):
            try:
                os.remove(job.temp_path)
                debug_info(f"[JOB/{job.job_id[:8]}] Cleaned temp: {job.temp_path}")
            except OSError:
                pass


def _on_persist_done(future: asyncio.Future) -> None:
    """Log MongoDB write errors without blocking the caller."""
    try:
        future.result()
    except Exception as exc:
        debug_error(f"[DB] Job persist failed: {exc}")


def _persist_job(job_id: str, fields: dict) -> None:
    """Fire-and-forget MongoDB upsert from a sync daemon thread.

    Schedules an async coroutine on the main event loop without blocking
    the calling thread. Silently no-ops if DB or event loop is unavailable.
    """
    db = get_db()
    if db is None or _event_loop is None:
        return

    # Serialize Pydantic models (e.g. ClassCount) to plain dicts for MongoDB
    mongo_fields: dict = {}
    for key, val in fields.items():
        mongo_fields[key] = val.model_dump() if hasattr(val, "model_dump") else val
    mongo_fields["updated_at"] = datetime.now(timezone.utc)

    async def _do_upsert() -> None:
        await db.video_jobs.update_one(  # type: ignore[union-attr]
            {"_id": job_id},
            {"$set": mongo_fields},
            upsert=True,
        )

    future = asyncio.run_coroutine_threadsafe(_do_upsert(), _event_loop)
    future.add_done_callback(_on_persist_done)


def _update_job(job: _Job, **kwargs: object) -> None:
    """Thread-safe update of job fields and fire-and-forget MongoDB persist.

    Args:
        job: Job to update.
        **kwargs: Field name → new value pairs.
    """
    with _jobs_lock:
        for key, val in kwargs.items():
            setattr(job, key, val)

    # Persist to MongoDB (non-blocking from thread)
    _persist_job(job.job_id, dict(kwargs))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/jobs",
    response_model=VideoJobStatus,
    status_code=202,
    summary="Submit URL video processing job",
    description="""
Kirim URL video publik untuk diproses secara asinkron (background job).

**Cara kerja:**
1. Paste URL dari Google Drive, Dropbox, atau direct video link
2. Backend stream-download video → proses frame sampling → deteksi kendaraan
3. Poll `GET /video/jobs/{job_id}` setiap beberapa detik untuk lihat progress & hasil

**Supported URL types:**
- Google Drive sharing link (`drive.google.com/file/d/...`)
- Dropbox link (auto-converted ke direct download)
- Direct MP4/AVI/MOV URL publik

**Frame Sampling:** Untuk video panjang (>1 jam), gunakan `sample_every_n_seconds=5` atau lebih besar.
    """,
)
async def create_video_job(request: VideoJobRequest) -> VideoJobStatus:
    """Create and start a background video processing job from a public URL."""
    # Evict oldest jobs if store is full
    with _jobs_lock:
        if len(_jobs) >= _MAX_JOBS:
            oldest_id: str = min(_jobs, key=lambda jid: _jobs[jid].created_at)
            old_job = _jobs.pop(oldest_id)
            # Clean up temp file if still exists
            if old_job.temp_path and os.path.exists(old_job.temp_path):
                try:
                    os.remove(old_job.temp_path)
                except OSError:
                    pass
            debug_info(f"[JOBS] Evicted oldest job: {oldest_id[:8]}")

    job_id: str = uuid.uuid4().hex
    job = _Job(job_id=job_id, video_url=request.url)

    with _jobs_lock:
        _jobs[job_id] = job

    # Persist initial job document to MongoDB
    db = get_db()
    if db is not None:
        await db.video_jobs.insert_one({
            "_id": job_id,
            "video_url": request.url,
            "status": "pending",
            "progress": 0.0,
            "message": "Job terdaftar",
            "counts": None,
            "video_info": None,
            "inference_config": None,
            "error": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })

    # Launch background thread (daemon=True → auto-killed when server exits)
    thread = threading.Thread(target=_run_job, args=(job, request), daemon=True, name=f"job-{job_id[:8]}")
    thread.start()

    debug_info(f"[JOBS] Created job {job_id[:8]} for URL: {request.url[:60]}...")

    return _job_to_status(job)


@router.get(
    "/jobs/{job_id}",
    response_model=VideoJobStatus,
    summary="Get video job status / result",
)
async def get_video_job(job_id: str) -> VideoJobStatus:
    """Poll a video job for its current status and result.

    Checks in-memory store first (fast path), then falls back to MongoDB
    (handles server restarts where job result was persisted but memory was cleared).

    Args:
        job_id: Job ID returned by POST /video/jobs.

    Returns:
        Current job status and result (if done).
    """
    # Fast path: in-memory store
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return _job_to_status(job)

    # Fallback: MongoDB (job completed before server restart, or evicted from memory)
    db = get_db()
    if db is not None:
        doc = await db.video_jobs.find_one({"_id": job_id})
        if doc is not None:
            counts = ClassCount(**doc["counts"]) if doc.get("counts") else None
            return VideoJobStatus(
                job_id=doc["_id"],
                status=doc["status"],
                progress=doc.get("progress", 0.0),
                message=doc.get("message", ""),
                counts=counts,
                video_info=doc.get("video_info"),
                inference_config=doc.get("inference_config"),
                error=doc.get("error"),
            )

    raise HTTPException(status_code=404, detail=f"Job '{job_id}' tidak ditemukan")


@router.get(
    "/jobs",
    response_model=list[VideoJobStatus],
    summary="List all video jobs",
)
async def list_video_jobs() -> list[VideoJobStatus]:
    """Return all jobs (memory + MongoDB), newest first, deduplicated."""
    with _jobs_lock:
        memory_statuses = {j.job_id: _job_to_status(j) for j in _jobs.values()}

    results: dict[str, VideoJobStatus] = dict(memory_statuses)

    # Merge MongoDB jobs (historical — not in memory)
    db = get_db()
    if db is not None:
        async for doc in db.video_jobs.find(
            {"_id": {"$nin": list(results.keys())}},
            sort=[("created_at", -1)],
            limit=200,
        ):
            counts = ClassCount(**doc["counts"]) if doc.get("counts") else None
            results[doc["_id"]] = VideoJobStatus(
                job_id=doc["_id"],
                status=doc["status"],
                progress=doc.get("progress", 0.0),
                message=doc.get("message", ""),
                counts=counts,
                video_info=doc.get("video_info"),
                inference_config=doc.get("inference_config"),
                error=doc.get("error"),
            )

    return list(results.values())


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    summary="Delete a video job",
)
async def delete_video_job(job_id: str) -> None:
    """Remove a job from the store and clean up its temp files.

    Removes from both in-memory store and MongoDB. If the job is only in
    MongoDB (e.g. loaded after a server restart), it is still deleted.

    Args:
        job_id: Job ID to delete.
    """
    with _jobs_lock:
        job = _jobs.pop(job_id, None)

    # Clean up temp file if job was in memory
    if job is not None and job.temp_path and os.path.exists(job.temp_path):
        try:
            os.remove(job.temp_path)
        except OSError:
            pass

    # Always try to delete from MongoDB
    db = get_db()
    db_result = None
    if db is not None:
        db_result = await db.video_jobs.delete_one({"_id": job_id})

    # 404 only if not found in either store
    if job is None and (db_result is None or db_result.deleted_count == 0):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' tidak ditemukan")

    debug_info(f"[JOBS] Deleted job {job_id[:8]}")


@router.post(
    "/preview-frame",
    summary="Extract first frame from URL video",
    description="Ekstrak frame pertama dari URL video publik menggunakan ffmpeg, return sebagai JPEG.",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "First frame as JPEG"},
        400: {"description": "Invalid URL or cannot open video"},
    },
)
async def preview_frame(
    url: str,
) -> Response:
    """Extract the first frame from a URL video and return as JPEG.

    Uses ffmpeg to stream directly from the URL. This avoids the
    'moov atom not found' issue that occurs when partially downloading
    MP4 files (moov atom is typically at the end of the file).

    Args:
        url: Public video URL (Google Drive, Dropbox, direct link).

    Returns:
        JPEG image of the first video frame.
    """
    import shutil
    import subprocess

    download_url: str = _normalize_url(url)
    output_path: str = str(TEMP_DIR / f"preview_{uuid.uuid4().hex[:8]}.jpg")

    ffmpeg_bin: str | None = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise HTTPException(status_code=500, detail="ffmpeg tidak ditemukan di server")

    try:
        # Use ffmpeg to extract first frame directly from URL
        # ffmpeg handles HTTP redirects, range requests, and moov atom seeking
        cmd: list[str] = [
            ffmpeg_bin,
            "-y",                      # overwrite output
            "-i", download_url,        # input from URL (ffmpeg handles HTTP)
            "-vframes", "1",           # extract only 1 frame
            "-q:v", "2",               # JPEG quality (2 = high quality)
            output_path,
        ]

        debug_info(f"[PREVIEW] Extracting frame via ffmpeg from: {download_url[:80]}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout for first frame
        )

        if result.returncode != 0:
            stderr_snippet: str = result.stderr[-500:] if result.stderr else ""
            debug_error(f"[PREVIEW] ffmpeg failed: {result.returncode}\n{stderr_snippet}")

            # Provide specific guidance based on URL type and error
            detail: str
            if "drive.google.com/drive/folders" in url:
                detail = "URL yang dimasukkan adalah folder Google Drive, bukan file video. Buka file videonya, lalu salin link share-nya."
            elif "drive.google.com" in url and "file/d/" not in url:
                detail = "Format URL Google Drive tidak dikenali. Gunakan link share dari file video (bukan folder)."
            elif "Invalid data" in stderr_snippet or "Error opening input" in stderr_snippet:
                detail = "URL tidak mengarah ke file video yang valid. Pastikan link dapat diakses secara publik dan merupakan file video (MP4, AVI, dll)."
            elif "403" in stderr_snippet or "forbidden" in stderr_snippet.lower():
                detail = "Akses ditolak (403). Pastikan file tidak dibatasi aksesnya — gunakan 'Anyone with the link' di Google Drive."
            elif "404" in stderr_snippet or "not found" in stderr_snippet.lower():
                detail = "File tidak ditemukan (404). Periksa kembali URL yang dimasukkan."
            else:
                detail = f"Tidak dapat membuka video dari URL ini. Pastikan URL valid dan dapat diakses publik."

            raise HTTPException(status_code=400, detail=detail)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=400, detail="ffmpeg tidak menghasilkan output frame")

        with open(output_path, "rb") as fp:
            jpeg_bytes: bytes = fp.read()

        debug_info(f"[PREVIEW] Extracted frame ({len(jpeg_bytes) / 1024:.0f} KB)")

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=300"},
        )

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Timeout: Video URL terlalu lambat untuk diakses")
    except Exception as e:
        debug_error(f"[PREVIEW] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Preview gagal: {str(e)}")
    finally:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


# ─── Startup Recovery ─────────────────────────────────────────────────────────

async def recover_interrupted_jobs() -> None:
    """On server startup, mark any interrupted jobs (pending/downloading/processing) as error.

    These jobs were running when the server last shut down. Since daemon threads are
    killed on exit and video data is not stored, they cannot be resumed — so we mark
    them as error so the frontend can show a clear message instead of polling forever.
    """
    db = get_db()
    if db is None:
        return

    result = await db.video_jobs.update_many(
        {"status": {"$in": ["pending", "downloading", "processing"]}},
        {"$set": {
            "status": "error",
            "error": "Server direstart saat job sedang berjalan. Silakan kirim ulang video.",
            "progress": 0.0,
            "updated_at": datetime.now(timezone.utc),
        }},
    )

    if result.modified_count > 0:
        debug_info(f"[JOBS] Marked {result.modified_count} interrupted job(s) as error on startup")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _job_to_status(job: _Job) -> VideoJobStatus:
    """Convert internal _Job to the public VideoJobStatus schema.

    Args:
        job: Internal job object.

    Returns:
        Serializable VideoJobStatus.
    """
    with _jobs_lock:
        return VideoJobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            counts=job.counts,
            video_info=job.video_info,
            inference_config=job.inference_config,
            error=job.error,
        )
