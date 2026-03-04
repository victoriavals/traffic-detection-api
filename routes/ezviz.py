"""
EZVIZ Cloud Capture Routes.

Tag: EZVIZ Cloud
Endpoints:
- GET  /ezviz/status    → Check EZVIZ credentials & connection
- GET  /ezviz/devices   → List all devices on account
- POST /ezviz/detect    → Capture-based detection (multiple captures)
- WS   /ezviz/stream    → Real-time capture-based stream via WebSocket

Uses the EZVIZ capture API (/api/lapp/device/capture) which is more reliable
than HLS/RTMP streaming for many camera models (e.g., H8c returns error 9053
on live stream but capture works perfectly).
"""

import asyncio
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
    CLASS_NAMES,
    debug_info,
    debug_error,
)
from models.schemas import EzvizDetectRequest, EzvizDetectResponse, ClassCount
from services.detector_service import DetectorService
from services.annotation_service import AnnotationService
from services.ezviz_service import EzvizService
from utils.pedestrian_filter import filter_pedestrian_on_vehicle


router = APIRouter(prefix="/ezviz", tags=["☁️ EZVIZ Cloud"])

# Service instances
detector: DetectorService = DetectorService()
annotator: AnnotationService = AnnotationService()
ezviz: EzvizService = EzvizService()

# Minimum interval between captures (seconds) — prevents hammering the API
_MIN_CAPTURE_INTERVAL_SEC: float = 0.5


@router.get(
    "/status",
    summary="Check EZVIZ connection status",
    description="Verifikasi apakah credentials EZVIZ sudah dikonfigurasi dan token valid.",
)
async def ezviz_status() -> dict:
    """Check EZVIZ API configuration and connectivity."""
    if not ezviz.is_configured():
        return {
            "configured": False,
            "connected": False,
            "message": "EZVIZ credentials belum dikonfigurasi. Set EZVIZ_APP_KEY dan EZVIZ_APP_SECRET di file .env",
        }

    try:
        token = await ezviz.get_access_token()
        return {
            "configured": True,
            "connected": True,
            "message": "EZVIZ Cloud terhubung",
            "token_preview": f"{token[:12]}...{token[-6:]}",
        }
    except Exception as e:
        return {
            "configured": True,
            "connected": False,
            "message": f"Gagal konek ke EZVIZ Cloud: {str(e)}",
        }


@router.get(
    "/devices",
    summary="List EZVIZ devices",
    description="Ambil daftar semua device/kamera yang terdaftar di akun EZVIZ.",
)
async def list_devices() -> dict:
    """List all devices registered on the EZVIZ account."""
    if not ezviz.is_configured():
        raise HTTPException(
            status_code=400,
            detail="EZVIZ credentials belum dikonfigurasi. Set EZVIZ_APP_KEY dan EZVIZ_APP_SECRET di file .env",
        )

    try:
        devices = await ezviz.get_device_list()
        return {
            "success": True,
            "count": len(devices),
            "devices": devices,
        }
    except Exception as e:
        debug_error(f"[EZVIZ/DEVICES] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/detect",
    response_model=EzvizDetectResponse,
    summary="Detect vehicles from EZVIZ cloud captures (JSON)",
    description="""
Capture beberapa gambar dari kamera EZVIZ via cloud API, lalu proses deteksi per-frame.

**Cara kerja:**
1. Setiap iterasi memanggil EZVIZ capture API untuk mendapatkan snapshot terbaru
2. Gambar di-download dan diproses dengan YOLOv11
3. Hasil deteksi dikumpulkan (tanpa tracking karena capture-based)

**Tidak perlu WiFi yang sama dengan kamera** — capture dilakukan melalui EZVIZ Cloud API.

**Request body (JSON):**
```json
{
    "device_serial": "BF7564746",
    "channel_no": 1,
    "confidence": 0.45,
    "iou": 0.5,
    "model_size": "SMALL",
    "frame_count": 5
}
```

**Note:** `frame_count` di sini berarti jumlah capture (bukan jumlah frame video).
Setiap capture membutuhkan ~2-3 detik. Disarankan gunakan 3-10.
    """,
)
async def detect_ezviz(request: EzvizDetectRequest) -> EzvizDetectResponse:
    """Capture multiple images from EZVIZ cloud and run detection on each."""
    # Clamp frame_count for capture mode (each capture takes ~2-3s)
    capture_count: int = min(request.frame_count, 20)
    debug_info(f"[EZVIZ/DETECT] Device: {request.device_serial} ({capture_count} captures)")

    if not ezviz.is_configured():
        raise HTTPException(
            status_code=400,
            detail="EZVIZ credentials belum dikonfigurasi",
        )

    counts: dict[str, int] = {
        "big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0,
    }
    frames_processed: int = 0
    process_start: float = time.time()
    last_frame_shape: tuple = (0, 0)

    try:
        for i in range(capture_count):
            try:
                frame: np.ndarray = await ezviz.capture_image(
                    device_serial=request.device_serial,
                    channel_no=request.channel_no,
                )
                last_frame_shape = (frame.shape[1], frame.shape[0])
            except Exception as e:
                debug_error(f"[EZVIZ/DETECT] Capture {i+1} failed: {e}")
                continue

            frames_processed += 1

            # Detect → filter (no tracking for capture-based — each frame is independent)
            detections: sv.Detections = detector.detect(
                frame, request.confidence, request.iou, request.model_size
            )
            detections = filter_pedestrian_on_vehicle(detections)

            # Count detections per class
            for j in range(len(detections)):
                cls_id: int = int(detections.class_id[j])
                cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
                key: str = cls_name.replace("-", "_")
                if key in counts:
                    counts[key] += 1

            # Small delay to avoid API rate limiting (except last)
            if i < capture_count - 1:
                await asyncio.sleep(_MIN_CAPTURE_INTERVAL_SEC)

        process_time: float = time.time() - process_start
        total: int = sum(counts.values())

        debug_info(f"[EZVIZ/DETECT] Done: {total} detections in {frames_processed} captures ({process_time:.1f}s)")

        width, height = last_frame_shape

        return EzvizDetectResponse(
            success=True,
            message=f"Processed {frames_processed} captures from EZVIZ cloud in {process_time:.1f}s",
            counts=ClassCount(
                big_vehicle=counts["big_vehicle"],
                car=counts["car"],
                pedestrian=counts["pedestrian"],
                two_wheeler=counts["two_wheeler"],
                total=total,
            ),
            stream_info={
                "resolution": f"{width}x{height}" if width > 0 else "unknown",
                "captures_requested": capture_count,
                "captures_processed": frames_processed,
                "processing_time_seconds": round(process_time, 2),
                "source": "ezviz_cloud_capture",
            },
            inference_config={
                "confidence": request.confidence,
                "iou": request.iou,
                "model_size": request.model_size,
                "device": detector.get_device_info(),
            },
            device_info={
                "device_serial": request.device_serial,
                "channel_no": request.channel_no,
            },
        )

    except Exception as e:
        debug_error(f"[EZVIZ/DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"EZVIZ processing failed: {str(e)}")


@router.websocket("/stream")
async def stream_ezviz(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time EZVIZ cloud capture-based stream.

    Uses repeated capture API calls instead of HLS/RTMP streaming.
    Each capture takes ~2-3 seconds, so effective FPS is ~0.3-0.5.
    Detection + annotation is run on each captured frame.

    Protocol:
    1. Client connects to WebSocket
    2. Client sends JSON config:
       {"device_serial": "BF7564746", "channel_no": 1, "confidence": 0.45,
        "iou": 0.5, "model_size": "SMALL", "send_frame": true}
    3. Server repeatedly captures images from EZVIZ cloud
    4. Server streams annotated frames as JSON:
       {"type": "frame", "frame": "<base64>", "counts": {...}, "fps": 0.3}
    5. Client sends {"action": "stop"} to stop
    """
    await ws.accept()
    debug_info("[EZVIZ/STREAM] WebSocket connected")

    try:
        # 1. Receive config from client
        config_raw: str = await ws.receive_text()
        config: dict = json.loads(config_raw)

        device_serial: str = config.get("device_serial", "")
        if not device_serial:
            await ws.send_json({"type": "error", "message": "Device serial is required"})
            await ws.close()
            return

        channel_no: int = int(config.get("channel_no", 1))
        confidence: float = float(config.get("confidence", DEFAULT_CONFIDENCE))
        iou_val: float = float(config.get("iou", DEFAULT_IOU))
        model_size: str = str(config.get("model_size", DEFAULT_MODEL_SIZE))
        send_frame: bool = bool(config.get("send_frame", True))

        # 2. Verify EZVIZ is configured
        if not ezviz.is_configured():
            await ws.send_json({
                "type": "error",
                "message": "EZVIZ credentials belum dikonfigurasi di .env",
            })
            await ws.close()
            return

        # 3. Verify connectivity with first capture
        await ws.send_json({
            "type": "info",
            "message": "Connecting to EZVIZ Cloud...",
        })

        try:
            first_frame: np.ndarray = await ezviz.capture_image(device_serial, channel_no)
            height, width = first_frame.shape[:2]
        except Exception as e:
            await ws.send_json({
                "type": "error",
                "message": f"Gagal capture dari EZVIZ: {str(e)}",
            })
            await ws.close()
            return

        await ws.send_json({
            "type": "info",
            "message": f"Connected: {width}x{height} (EZVIZ Cloud Capture Mode)",
        })

        # 4. Optimized capture loop
        # Connection pooling in EzvizService gives ~1s/capture (vs ~5s before).
        # No artificial sleep — capture as fast as the EZVIZ API allows.
        # Retry logic handles "Device response timeout" from camera rate-limiting.
        counts: dict[str, int] = {
            "big_vehicle": 0, "car": 0, "pedestrian": 0, "two_wheeler": 0,
        }
        frame_number: int = 0
        fps_start: float = time.time()
        fps_count: int = 0
        display_fps: float = 0.0
        consecutive_failures: int = 0

        frame: np.ndarray = first_frame

        while True:
            # Check for stop command (non-blocking)
            try:
                msg_raw: str = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                msg_data: dict = json.loads(msg_raw)
                if msg_data.get("action") == "stop":
                    debug_info("[EZVIZ/STREAM] Stop requested by client")
                    break
            except (asyncio.TimeoutError, Exception):
                pass

            # Detect → filter on current frame (GPU inference ~0.1s)
            detections: sv.Detections = detector.detect(frame, confidence, iou_val, model_size)
            detections = filter_pedestrian_on_vehicle(detections)

            # Count detections per class
            for j in range(len(detections)):
                cls_id: int = int(detections.class_id[j])
                cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
                key: str = cls_name.replace("-", "_")
                if key in counts:
                    counts[key] += 1

            frame_number += 1
            fps_count += 1

            # FPS calculation
            elapsed_fps: float = time.time() - fps_start
            if elapsed_fps >= 1.0:
                display_fps = fps_count / elapsed_fps
                fps_count = 0
                fps_start = time.time()

            # Annotate frame & encode
            frame_b64: str | None = None
            if send_frame:
                annotated: np.ndarray = annotator.annotate_detections(
                    frame, detections, show_tracker_id=False
                )

                success: bool
                buffer: np.ndarray
                success, buffer = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if success:
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # Send frame message
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

            # Capture next frame (EzvizService handles retries for camera timeouts)
            try:
                frame = await ezviz.capture_image(device_serial, channel_no)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                debug_info(f"[EZVIZ/STREAM] Capture failed ({consecutive_failures}/5): {e}")
                if consecutive_failures >= 5:
                    await ws.send_json({
                        "type": "error",
                        "message": "Gagal capture 5x berturut-turut — stream dihentikan",
                    })
                    break
                await ws.send_json({
                    "type": "info",
                    "message": f"Capture gagal, retry... ({consecutive_failures}/5)",
                })

    except WebSocketDisconnect:
        debug_info("[EZVIZ/STREAM] Client disconnected")
    except json.JSONDecodeError:
        debug_error("[EZVIZ/STREAM] Invalid JSON config from client")
        try:
            await ws.send_json({"type": "error", "message": "Invalid JSON config"})
        except Exception:
            pass
    except Exception as e:
        debug_error(f"[EZVIZ/STREAM] Error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        debug_info("[EZVIZ/STREAM] Cleanup done")
