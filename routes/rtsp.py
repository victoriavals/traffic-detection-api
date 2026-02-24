"""
RTSP Stream Routes.

Tag: RTSP Stream
Endpoints:
- POST /rtsp/detect  → JSON counting dari N frame snapshot
- WS   /rtsp/stream  → Real-time annotated frames via WebSocket
"""

import asyncio
import os
import time
import base64
import json

import cv2
import numpy as np
import supervision as sv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from constant_var import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_SIZE,
    DEFAULT_LINE_START_X,
    DEFAULT_LINE_START_Y,
    DEFAULT_LINE_END_X,
    DEFAULT_LINE_END_Y,
    CLASS_NAMES,
    WEBSOCKET_FPS_LIMIT,
    debug_info,
    debug_error,
)
from models.schemas import RTSPRequest, RTSPDetectResponse, ClassCount
from services.detector_service import DetectorService
from services.annotation_service import AnnotationService
from utils.pedestrian_filter import filter_pedestrian_on_vehicle


router = APIRouter(prefix="/rtsp", tags=["📡 RTSP Stream"])

# Service instances
detector: DetectorService = DetectorService()
annotator: AnnotationService = AnnotationService()

# WebSocket stream stability constants
_MAX_CONSECUTIVE_FAILURES: int = 30    # tolerate N bad reads before reconnect
_MAX_RECONNECT_ATTEMPTS: int = 5       # max RTSP reconnect attempts
_RECONNECT_DELAY_SEC: float = 2.0      # delay between reconnect attempts


def _open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """Open RTSP stream with TCP transport for stability.

    Args:
        url: RTSP stream URL.

    Returns:
        Opened VideoCapture object.

    Raises:
        ValueError: If stream cannot be opened.
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap: cv2.VideoCapture = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Cannot open RTSP stream: {url}")
    return cap


def _read_frame_sync(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Read one frame synchronously (intended to run in thread executor).

    Args:
        cap: Opened VideoCapture object.

    Returns:
        Tuple of (success, frame).
    """
    ret: bool
    frame: np.ndarray
    ret, frame = cap.read()
    return ret, frame if ret else None


@router.post(
    "/detect",
    response_model=RTSPDetectResponse,
    summary="Detect vehicles from RTSP snapshot (JSON)",
    description="""
Koneksi ke RTSP stream, capture sejumlah frame, proses deteksi + tracking + counting, lalu return hasil JSON.

**Request body (JSON):**
```json
{
    "url": "rtsp://user:pass@192.168.1.72:554/H.264",
    "confidence": 0.45,
    "iou": 0.5,
    "model_size": "SMALL",
    "frame_count": 150,
    "line_config": {
        "start_x": 0.0, "start_y": 0.15,
        "end_x": 1.0, "end_y": 0.65
    }
}
```

**Tips:**
- Gunakan `frame_count:150` untuk ~5 detik sampling pada 30 FPS
- Naikkan `frame_count` untuk sampling lebih lama
- Pastikan device terhubung ke jaringan yang sama dengan CCTV

**⚠️ Timeout:** Koneksi RTSP bisa gagal jika IP/port/credentials salah.
    """,
)
async def detect_rtsp(request: RTSPRequest) -> RTSPDetectResponse:
    """Connect to RTSP stream, capture frames, and return counting results."""
    debug_info(f"[RTSP/DETECT] Connecting to: {request.url} ({request.frame_count} frames)")

    try:
        cap: cv2.VideoCapture = _open_rtsp_stream(request.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Line config
        lc = request.line_config
        lsx: float = lc.start_x if lc else DEFAULT_LINE_START_X
        lsy: float = lc.start_y if lc else DEFAULT_LINE_START_Y
        lex: float = lc.end_x if lc else DEFAULT_LINE_END_X
        ley: float = lc.end_y if lc else DEFAULT_LINE_END_Y

        # Setup tracking & counting
        tracker: sv.ByteTrack = sv.ByteTrack(
            lost_track_buffer=30,
            frame_rate=int(video_fps),
        )
        line_start: sv.Point = sv.Point(int(width * lsx), int(height * lsy))
        line_end: sv.Point = sv.Point(int(width * lex), int(height * ley))
        line_zone: sv.LineZone = sv.LineZone(start=line_start, end=line_end)

        counts: dict[str, int] = {
            "big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0,
        }
        counted_ids: set[int] = set()

        frame_count: int = 0
        process_start: float = time.time()

        while frame_count < request.frame_count:
            ret: bool
            frame: np.ndarray | None
            ret, frame = cap.read()

            if not ret or frame is None:
                debug_info("[RTSP/DETECT] Stream ended or broken")
                break

            frame_count += 1

            # Detect → filter → track → count
            detections: sv.Detections = detector.detect(
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
                    cls_id: int = int(detections.class_id[i])
                    cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
                    key: str = cls_name.replace("-", "_")
                    if key in counts:
                        counts[key] += 1

        process_time: float = time.time() - process_start
        total: int = sum(counts.values())

        debug_info(f"[RTSP/DETECT] Done: {total} vehicles in {frame_count} frames ({process_time:.1f}s)")

        return RTSPDetectResponse(
            success=True,
            message=f"Processed {frame_count} frames from RTSP stream in {process_time:.1f}s",
            counts=ClassCount(
                big_vehicle=counts["big_vehicle"],
                car=counts["car"],
                pedestrian=counts["pedestrian"],
                two_wheeler=counts["two_wheeler"],
                total=total,
            ),
            stream_info={
                "resolution": f"{width}x{height}",
                "fps": round(video_fps, 1),
                "frames_processed": frame_count,
                "processing_time_seconds": round(process_time, 2),
            },
            inference_config={
                "confidence": request.confidence,
                "iou": request.iou,
                "model_size": request.model_size,
                "device": detector.get_device_info(),
                "counting_line": {
                    "start": f"({lsx}, {lsy})",
                    "end": f"({lex}, {ley})",
                },
            },
        )

    except Exception as e:
        debug_error(f"[RTSP/DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"RTSP processing failed: {str(e)}")
    finally:
        cap.release()


@router.websocket("/stream")
async def stream_rtsp(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time RTSP stream with annotations.

    Stability features:
    - cap.read() runs in asyncio thread executor (non-blocking, improves FPS)
    - Frame retry tolerance: tolerates up to _MAX_CONSECUTIVE_FAILURES bad reads
    - Auto-reconnect: reopens RTSP connection up to _MAX_RECONNECT_ATTEMPTS times

    Protocol:
    1. Client connects to WebSocket
    2. Client sends JSON config:
       {"url": "...", "confidence": 0.45, "iou": 0.5, "model_size": "SMALL",
        "line_config": {"start_x": 0.0, "start_y": 0.15, "end_x": 1.0, "end_y": 0.65}}
    3. Server streams annotated frames as JSON:
       {"type": "frame", "frame": "<base64>", "counts": {...}, "fps": 15.2, "frame_number": 100}
    4. Client sends {"action": "stop"} to stop
    5. Disconnect → cleanup automatically
    """
    await ws.accept()
    debug_info("[RTSP/STREAM] WebSocket connected")

    loop = asyncio.get_event_loop()
    cap: cv2.VideoCapture | None = None

    try:
        # 1. Receive config from client
        config_raw: str = await ws.receive_text()
        config: dict = json.loads(config_raw)

        url: str = config.get("url", "")
        if not url:
            await ws.send_json({"type": "error", "message": "RTSP URL is required"})
            await ws.close()
            return

        confidence: float = float(config.get("confidence", DEFAULT_CONFIDENCE))
        iou_val: float = float(config.get("iou", DEFAULT_IOU))
        model_size: str = str(config.get("model_size", DEFAULT_MODEL_SIZE))
        send_frame: bool = bool(config.get("send_frame", True))

        lc: dict = config.get("line_config") or {}
        lsx: float = float(lc.get("start_x", DEFAULT_LINE_START_X))
        lsy: float = float(lc.get("start_y", DEFAULT_LINE_START_Y))
        lex: float = float(lc.get("end_x", DEFAULT_LINE_END_X))
        ley: float = float(lc.get("end_y", DEFAULT_LINE_END_Y))

        # 2. Open RTSP stream
        debug_info(f"[RTSP/STREAM] Opening: {url}")
        try:
            cap = _open_rtsp_stream(url)
        except ValueError as e:
            await ws.send_json({"type": "error", "message": str(e)})
            await ws.close()
            return

        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Setup tracker & line zone (persist across reconnects)
        tracker: sv.ByteTrack = sv.ByteTrack(
            lost_track_buffer=30,
            frame_rate=int(video_fps),
        )
        line_start: sv.Point = sv.Point(int(width * lsx), int(height * lsy))
        line_end: sv.Point = sv.Point(int(width * lex), int(height * ley))
        line_zone: sv.LineZone = sv.LineZone(start=line_start, end=line_end)

        counts: dict[str, int] = {
            "big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0,
        }
        counted_ids: set[int] = set()

        # Send initial connection info
        await ws.send_json({
            "type": "info",
            "message": f"Connected: {width}x{height} @ {video_fps:.0f} FPS",
        })

        # 3. Stream loop variables
        frame_number: int = 0
        fps_start: float = time.time()
        fps_count: int = 0
        display_fps: float = 0.0
        consecutive_failures: int = 0
        frame_skip: int = max(1, int(video_fps / WEBSOCKET_FPS_LIMIT))

        while True:
            # --- Check for stop command (non-blocking 1ms timeout) ---
            try:
                msg_raw: str = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg_data: dict = json.loads(msg_raw)
                if msg_data.get("action") == "stop":
                    debug_info("[RTSP/STREAM] Stop requested by client")
                    break
            except (asyncio.TimeoutError, Exception):
                pass  # No message, continue

            # --- Read frame in thread executor (non-blocking) ---
            ret: bool
            frame: np.ndarray | None
            ret, frame = await loop.run_in_executor(None, _read_frame_sync, cap)

            if not ret or frame is None:
                consecutive_failures += 1
                debug_info(
                    f"[RTSP/STREAM] Frame read failed "
                    f"({consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES})"
                )

                if consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
                    # Tolerate transient failures — skip frame and retry
                    await asyncio.sleep(0.05)
                    continue

                # Too many failures → attempt RTSP reconnect
                debug_info("[RTSP/STREAM] Max failures reached, attempting reconnect...")
                cap.release()
                cap = None

                reconnected: bool = False
                for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
                    await ws.send_json({
                        "type": "info",
                        "message": f"Reconnecting... ({attempt}/{_MAX_RECONNECT_ATTEMPTS})",
                    })
                    try:
                        await asyncio.sleep(_RECONNECT_DELAY_SEC)
                        cap = _open_rtsp_stream(url)
                        new_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frame_skip = max(1, int(new_fps / WEBSOCKET_FPS_LIMIT))
                        consecutive_failures = 0
                        reconnected = True
                        debug_info(f"[RTSP/STREAM] Reconnected (attempt {attempt})")
                        await ws.send_json({
                            "type": "info",
                            "message": "Reconnected to stream",
                        })
                        break
                    except ValueError:
                        debug_info(f"[RTSP/STREAM] Reconnect attempt {attempt} failed")

                if not reconnected:
                    await ws.send_json({
                        "type": "error",
                        "message": "Stream terputus — gagal reconnect setelah beberapa percobaan",
                    })
                    break

                continue  # Restart loop with new cap

            # --- Successful frame read ---
            consecutive_failures = 0
            frame_number += 1
            fps_count += 1

            # FPS calculation (update every 1 second)
            elapsed_fps: float = time.time() - fps_start
            if elapsed_fps >= 1.0:
                display_fps = fps_count / elapsed_fps
                fps_count = 0
                fps_start = time.time()

            # Skip frames to match WEBSOCKET_FPS_LIMIT
            if frame_number % frame_skip != 0:
                continue

            # --- Detect → filter → track → count ---
            detections: sv.Detections = detector.detect(frame, confidence, iou_val, model_size)
            detections = filter_pedestrian_on_vehicle(detections)
            detections = tracker.update_with_detections(detections)

            crossed_in_arr: np.ndarray
            crossed_out_arr: np.ndarray
            crossed_in_arr, crossed_out_arr = line_zone.trigger(detections)

            for i in range(len(detections)):
                if crossed_in_arr[i] or crossed_out_arr[i]:
                    t_id: int = int(detections.tracker_id[i])
                    if t_id in counted_ids:
                        continue
                    counted_ids.add(t_id)
                    cls_name: str = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                    key: str = cls_name.replace("-", "_")
                    if key in counts:
                        counts[key] += 1

            # --- Annotate frame & encode ---
            frame_b64: str | None = None
            if send_frame:
                annotated: np.ndarray = annotator.annotate_detections(
                    frame, detections, show_tracker_id=True
                )
                annotator.draw_counting_line(annotated, line_zone)
                annotator.draw_stats_overlay(annotated, counts, display_fps)

                success: bool
                buffer: np.ndarray
                success, buffer = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if success:
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # --- Send frame message ---
            await ws.send_json({
                "type": "frame",
                "frame": frame_b64,
                "counts": {
                    "big_vehicle": counts["big_vehicle"],
                    "car": counts["car"],
                    "pedestrian": counts["pedestrian"],
                    "two_wheeler": counts["two_wheeler"],
                    "total": sum(counts.values()),
                },
                "fps": round(display_fps, 1),
                "frame_number": frame_number,
            })

    except WebSocketDisconnect:
        debug_info("[RTSP/STREAM] Client disconnected")
    except json.JSONDecodeError:
        debug_error("[RTSP/STREAM] Invalid JSON config from client")
        try:
            await ws.send_json({"type": "error", "message": "Invalid JSON config"})
        except Exception:
            pass
    except Exception as e:
        debug_error(f"[RTSP/STREAM] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()
        debug_info("[RTSP/STREAM] Cleanup done")
