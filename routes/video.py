"""
Video Processing Routes.

Tag: Video Processing
Endpoints:
- POST /video/detect   → JSON counting per class (crossing line)
- POST /video/annotate → Annotated video file (MP4)
"""

import os
import time
import uuid
from typing import Literal

import cv2
import numpy as np
import supervision as sv
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

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
from models.schemas import VideoDetectResponse, ClassCount
from services.detector_service import DetectorService
from services.annotation_service import AnnotationService
from utils.pedestrian_filter import filter_pedestrian_on_vehicle


router = APIRouter(prefix="/video", tags=["🎬 Video Processing"])

# Service instances
detector: DetectorService = DetectorService()
annotator: AnnotationService = AnnotationService()


def _process_video(
    video_path: str,
    confidence: float,
    iou: float,
    model_size: str,
    line_start_x: float,
    line_start_y: float,
    line_end_x: float,
    line_end_y: float,
    annotate: bool = False,
    output_path: str | None = None,
) -> dict:
    """Process video file: detect, track, count, and optionally annotate.

    Args:
        video_path: Path to input video file.
        confidence: Detection confidence threshold.
        iou: IoU threshold for NMS.
        model_size: Model size to use.
        line_start_x: Counting line start X (percentage).
        line_start_y: Counting line start Y (percentage).
        line_end_x: Counting line end X (percentage).
        line_end_y: Counting line end Y (percentage).
        annotate: Whether to write annotated output video.
        output_path: Path for output video (required if annotate=True).

    Returns:
        Dictionary with counts, video_info, and processing stats.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup ByteTrack tracker
    tracker: sv.ByteTrack = sv.ByteTrack(
        lost_track_buffer=30,
        frame_rate=int(video_fps),
    )

    # Setup LineZone (using percentage-based coordinates)
    line_start: sv.Point = sv.Point(int(width * line_start_x), int(height * line_start_y))
    line_end: sv.Point = sv.Point(int(width * line_end_x), int(height * line_end_y))
    line_zone: sv.LineZone = sv.LineZone(start=line_start, end=line_end)

    # Setup video writer for annotated output
    writer: cv2.VideoWriter | None = None
    if annotate and output_path:
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    # Counting state
    counts: dict[str, int] = {
        "big_vehicle": 0,
        "car": 0,
        "pedestrian": 0,
        "two_wheeler": 0,
    }
    counted_tracker_ids: set[int] = set()

    frame_count: int = 0
    process_start: float = time.time()

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detect
            detections: sv.Detections = detector.detect(
                frame, confidence, iou, model_size
            )

            # Filter drivers/riders
            detections = filter_pedestrian_on_vehicle(detections)

            # Track
            detections = tracker.update_with_detections(detections)

            # Count line crossings
            crossed_in: np.ndarray
            crossed_out: np.ndarray
            crossed_in, crossed_out = line_zone.trigger(detections)

            for i in range(len(detections)):
                if crossed_in[i] or crossed_out[i]:
                    tracker_id: int = int(detections.tracker_id[i])
                    if tracker_id in counted_tracker_ids:
                        continue

                    counted_tracker_ids.add(tracker_id)
                    class_id: int = int(detections.class_id[i])
                    class_name: str = CLASS_NAMES.get(class_id, "unknown")
                    key: str = class_name.replace("-", "_")
                    if key in counts:
                        counts[key] += 1

            # Write annotated frame
            if annotate and writer:
                fps_val: float = frame_count / max(time.time() - process_start, 0.001)
                annotated: np.ndarray = annotator.annotate_detections(
                    frame, detections, show_tracker_id=True
                )
                annotator.draw_counting_line(annotated, line_zone)
                annotator.draw_stats_overlay(annotated, counts, fps_val)
                writer.write(annotated)

    finally:
        cap.release()
        if writer:
            writer.release()

    process_time: float = time.time() - process_start
    total_count: int = sum(counts.values())

    return {
        "counts": counts,
        "total": total_count,
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": round(video_fps, 1),
            "total_frames": total_frames,
            "frames_processed": frame_count,
            "duration_seconds": round(total_frames / video_fps, 1) if video_fps > 0 else 0,
            "processing_time_seconds": round(process_time, 2),
        },
    }


@router.post(
    "/detect",
    response_model=VideoDetectResponse,
    summary="Count vehicles in video (JSON)",
    description="""
Upload sebuah video dan dapatkan hasil counting kendaraan dalam format JSON.

**Cara kerja:**
1. Video diproses frame-by-frame menggunakan YOLO detection
2. Objek dilacak menggunakan ByteTrack multi-object tracker
3. Kendaraan dihitung saat melewati counting line (diagonal)
4. Driver/rider otomatis difilter dari hitungan pedestrian

**Response berisi:**
- Jumlah per kelas: big-vehicle, car, pedestrian, two-wheeler
- Info video: resolusi, FPS, total frames, durasi
- Waktu processing

**Supported formats:** MP4, AVI, MOV, MKV

**Parameter opsional:**
- `confidence` — Threshold deteksi (default: 0.45)
- `iou` — IoU threshold untuk NMS (default: 0.5)
- `model_size` — SMALL (cepat, cocok real-time) atau MEDIUM (akurat, lebih lambat)
- `line_start_x/y`, `line_end_x/y` — Posisi counting line sebagai persentase (0.0-1.0)

**Default counting line:** Diagonal dari kiri-atas (0%, 15%) ke kanan-bawah (100%, 65%).
Objek dihitung saat centroid-nya melewati garis ini.
    """,
)
async def detect_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0, description="Detection confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    model_size: Literal["SMALL", "MEDIUM"] = Query(DEFAULT_MODEL_SIZE, description="Model: SMALL (faster) or MEDIUM (more accurate)"),
    line_start_x: float = Query(DEFAULT_LINE_START_X, ge=0.0, le=1.0, description="Counting line start X (0.0=left, 1.0=right)"),
    line_start_y: float = Query(DEFAULT_LINE_START_Y, ge=0.0, le=1.0, description="Counting line start Y (0.0=top, 1.0=bottom)"),
    line_end_x: float = Query(DEFAULT_LINE_END_X, ge=0.0, le=1.0, description="Counting line end X"),
    line_end_y: float = Query(DEFAULT_LINE_END_Y, ge=0.0, le=1.0, description="Counting line end Y"),
) -> VideoDetectResponse:
    """Upload video and return vehicle counting results as JSON."""
    debug_info(f"[VIDEO/DETECT] Processing: {file.filename} (conf={confidence}, model={model_size})")

    # Save uploaded file to temp
    temp_input: str = str(TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4")

    try:
        content: bytes = await file.read()
        with open(temp_input, "wb") as f:
            f.write(content)

        result: dict = _process_video(
            video_path=temp_input,
            confidence=confidence,
            iou=iou,
            model_size=model_size,
            line_start_x=line_start_x,
            line_start_y=line_start_y,
            line_end_x=line_end_x,
            line_end_y=line_end_y,
            annotate=False,
        )

        counts: dict = result["counts"]

        response = VideoDetectResponse(
            success=True,
            message=f"Processed {result['video_info']['frames_processed']} frames in {result['video_info']['processing_time_seconds']}s",
            counts=ClassCount(
                big_vehicle=counts["big_vehicle"],
                car=counts["car"],
                pedestrian=counts["pedestrian"],
                two_wheeler=counts["two_wheeler"],
                total=result["total"],
            ),
            video_info=result["video_info"],
            inference_config={
                "confidence": confidence,
                "iou": iou,
                "model_size": model_size,
                "device": detector.get_device_info(),
                "counting_line": {
                    "start": f"({line_start_x}, {line_start_y})",
                    "end": f"({line_end_x}, {line_end_y})",
                },
            },
        )

        debug_info(f"[VIDEO/DETECT] Done: {result['total']} total vehicles counted")
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        debug_error(f"[VIDEO/DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        # Cleanup input temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)


@router.post(
    "/annotate",
    summary="Annotate video with detections (MP4)",
    description="""
Upload sebuah video dan dapatkan video hasil anotasi dengan bounding box, tracking IDs, counting line, dan stats overlay.

**Cara kerja:**
1. Video diproses frame-by-frame (YOLO detection + ByteTrack tracking)
2. Setiap frame dianotasi dengan bounding box, label, dan ID tracker
3. Counting line diagonal ditambahkan
4. Stats overlay (jumlah per kelas + FPS) ditampilkan di sudut kiri atas
5. Output: video MP4 dengan anotasi lengkap

**Response:** File video MP4 dengan:
- Bounding box berwarna per kelas
- Label: #TrackerID ClassName Confidence
- Garis counting diagonal
- Stats overlay: FPS, jumlah per kelas, total

**Supported formats input:** MP4, AVI, MOV, MKV
**Format output:** MP4 (codec mp4v)

**⚠️ Warning:** Processing bisa memakan waktu beberapa menit untuk video panjang.
    """,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Annotated video file"},
        400: {"description": "Invalid video file"},
        500: {"description": "Processing failed"},
    },
)
async def annotate_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0, description="Detection confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    model_size: Literal["SMALL", "MEDIUM"] = Query(DEFAULT_MODEL_SIZE, description="Model: SMALL (faster) or MEDIUM (more accurate)"),
    line_start_x: float = Query(DEFAULT_LINE_START_X, ge=0.0, le=1.0, description="Counting line start X"),
    line_start_y: float = Query(DEFAULT_LINE_START_Y, ge=0.0, le=1.0, description="Counting line start Y"),
    line_end_x: float = Query(DEFAULT_LINE_END_X, ge=0.0, le=1.0, description="Counting line end X"),
    line_end_y: float = Query(DEFAULT_LINE_END_Y, ge=0.0, le=1.0, description="Counting line end Y"),
) -> FileResponse:
    """Upload video and return annotated MP4 with detections, tracking, and counting."""
    debug_info(f"[VIDEO/ANNOTATE] Processing: {file.filename} (conf={confidence}, model={model_size})")

    temp_input: str = str(TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4")
    temp_output: str = str(TEMP_DIR / f"annotated_{uuid.uuid4().hex[:8]}.mp4")

    try:
        content: bytes = await file.read()
        with open(temp_input, "wb") as f:
            f.write(content)

        result: dict = _process_video(
            video_path=temp_input,
            confidence=confidence,
            iou=iou,
            model_size=model_size,
            line_start_x=line_start_x,
            line_start_y=line_start_y,
            line_end_x=line_end_x,
            line_end_y=line_end_y,
            annotate=True,
            output_path=temp_output,
        )

        if not os.path.exists(temp_output):
            raise HTTPException(status_code=500, detail="Failed to generate annotated video")

        debug_info(
            f"[VIDEO/ANNOTATE] Done: {result['total']} vehicles, "
            f"{result['video_info']['frames_processed']} frames"
        )

        original_name: str = os.path.splitext(file.filename or "video")[0]

        return FileResponse(
            path=temp_output,
            media_type="video/mp4",
            filename=f"annotated_{original_name}.mp4",
            headers={
                "X-Total-Count": str(result["total"]),
                "X-Frames-Processed": str(result["video_info"]["frames_processed"]),
                "X-Processing-Time": str(result["video_info"]["processing_time_seconds"]),
            },
            background=BackgroundTask(_cleanup_files, temp_input, temp_output),
        )

    except HTTPException:
        raise
    except ValueError as e:
        _safe_remove(temp_input, temp_output)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _safe_remove(temp_input, temp_output)
        debug_error(f"[VIDEO/ANNOTATE] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Video annotation failed: {str(e)}")


def _safe_remove(*paths: str) -> None:
    """Safely remove files, ignoring errors.

    Args:
        paths: File paths to remove.
    """
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _cleanup_files(*paths: str) -> None:
    """Remove temporary files after response is sent.

    Args:
        paths: File paths to remove.
    """
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                debug_info(f"[CLEANUP] Removed temp file: {path}")
        except OSError as e:
            debug_error(f"[CLEANUP] Failed to remove {path}: {e}")
