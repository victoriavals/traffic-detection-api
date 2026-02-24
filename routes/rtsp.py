"""
RTSP Stream Routes.

Tag: RTSP Stream
Endpoints:
- POST /rtsp/detect  → JSON counting dari N frame snapshot
- WS   /rtsp/stream  → Real-time annotated frames via WebSocket
"""

import os
import time
import base64
import json
from typing import Literal

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
    RTSP_DEFAULT_FRAME_COUNT,
    RTSP_MAX_FRAME_COUNT,
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


def _open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """Open RTSP stream with TCP transport for stability.

    Args:
        url: RTSP stream URL.

    Returns:
        Opened VideoCapture object.

    Raises:
        ValueError: If stream cannot be opened.
    """
    # Set TCP transport for RTSP stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap: cv2.VideoCapture = cv2.VideoCapture(url)

    if not cap.isOpened():
        raise ValueError(f"Cannot open RTSP stream: {url}")

    return cap


@router.post(
    "/detect",
    response_model=RTSPDetectResponse,
    summary="Detect vehicles from RTSP snapshot (JSON)",
    description="""
Koneksi ke RTSP stream, capture sejumlah frame, proses deteksi + tracking + counting, lalu return hasil JSON.

**Cara kerja:**
1. Buka koneksi ke RTSP stream via URL yang diberikan
2. Capture frame sebanyak `frame_count` (default: 150 frame)
3. Setiap frame: YOLO detection → filter pedestrian → ByteTrack tracking → LineZone counting
4. Tutup koneksi → return aggregate counting per kelas

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
            frame: np.ndarray
            ret, frame = cap.read()

            if not ret:
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

    **Protocol:**
    1. Client connects ke WebSocket
    2. Client mengirim JSON config:
       ```json
       {
           "url": "rtsp://user:pass@ip:554/path",
           "confidence": 0.45,
           "iou": 0.5,
           "model_size": "SMALL",
           "line_config": {
               "start_x": 0.0, "start_y": 0.15,
               "end_x": 1.0, "end_y": 0.65
           }
       }
       ```
    3. Server mulai streaming annotated frames sebagai JSON messages:
       ```json
       {
           "type": "frame",
           "frame": "<base64 JPEG>",
           "counts": {"big_vehicle": 0, "car": 5, ...},
           "fps": 15.2,
           "frame_number": 100
       }
       ```
    4. Client bisa mengirim `{"action": "stop"}` kapan saja
    5. Disconnect → cleanup otomatis
    """
    await ws.accept()
    debug_info("[RTSP/STREAM] WebSocket connected")

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

        confidence: float = config.get("confidence", DEFAULT_CONFIDENCE)
        iou_val: float = config.get("iou", DEFAULT_IOU)
        model_size: str = config.get("model_size", DEFAULT_MODEL_SIZE)

        lc: dict = config.get("line_config", {})
        lsx: float = lc.get("start_x", DEFAULT_LINE_START_X)
        lsy: float = lc.get("start_y", DEFAULT_LINE_START_Y)
        lex: float = lc.get("end_x", DEFAULT_LINE_END_X)
        ley: float = lc.get("end_y", DEFAULT_LINE_END_Y)

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

        # Setup tracker & line zone
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

        # Send stream info
        await ws.send_json({
            "type": "info",
            "message": f"Connected to RTSP stream: {width}x{height} @ {video_fps:.0f} FPS",
        })

        # 3. Stream loop
        frame_number: int = 0
        fps_start: float = time.time()
        fps_count: int = 0
        display_fps: float = 0.0
        frame_interval: float = 1.0 / WEBSOCKET_FPS_LIMIT  # Limit FPS for WebSocket

        while True:
            # Check for stop command (non-blocking)
            try:
                import asyncio
                msg: str = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg_data: dict = json.loads(msg)
                if msg_data.get("action") == "stop":
                    debug_info("[RTSP/STREAM] Stop requested by client")
                    break
            except Exception:
                pass  # No message received, continue

            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()

            if not ret:
                await ws.send_json({"type": "error", "message": "RTSP stream ended or broken"})
                break

            frame_number += 1
            fps_count += 1

            # FPS calculation
            elapsed: float = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # Skip frames to match FPS limit
            if frame_number % max(1, int(video_fps / WEBSOCKET_FPS_LIMIT)) != 0:
                continue

            # Detect → filter → track → count
            detections: sv.Detections = detector.detect(
                frame, confidence, iou_val, model_size
            )
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
                    cls_id: int = int(detections.class_id[i])
                    cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
                    key: str = cls_name.replace("-", "_")
                    if key in counts:
                        counts[key] += 1

            # Annotate frame
            annotated: np.ndarray = annotator.annotate_detections(
                frame, detections, show_tracker_id=True
            )
            annotator.draw_counting_line(annotated, line_zone)
            annotator.draw_stats_overlay(annotated, counts, display_fps)

            # Encode to base64 JPEG
            success: bool
            buffer: np.ndarray
            success, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            if not success:
                continue

            frame_b64: str = base64.b64encode(buffer.tobytes()).decode("utf-8")
            total: int = sum(counts.values())

            # Send frame message
            await ws.send_json({
                "type": "frame",
                "frame": frame_b64,
                "counts": {
                    "big_vehicle": counts["big_vehicle"],
                    "car": counts["car"],
                    "pedestrian": counts["pedestrian"],
                    "two_wheeler": counts["two_wheeler"],
                    "total": total,
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
