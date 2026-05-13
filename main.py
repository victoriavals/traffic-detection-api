"""
Traffic Counter API — FastAPI Application.

Real-time traffic detection, tracking, and counting API
powered by YOLOv11s/m trained model.

Classes: big-vehicle, car, pedestrian, two-wheeler
Supports: Image upload, Video upload, RTSP stream

Author: Naufal Firdaus
"""

import asyncio
import os
import glob
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from constant_var import TEMP_DIR, JWT_SECRET_KEY, debug_info, debug_warning, debug_error, mask_rtsp, mask_secrets, log_http_request
from services.detector_service import DetectorService
from services.database import connect_db, close_db
from routes.image import router as image_router
from routes.video import router as video_router
from routes.video_jobs import router as video_jobs_router, set_event_loop, recover_interrupted_jobs
from routes.rtsp import router as rtsp_router
from routes.ezviz import router as ezviz_router
from routes.stats import router as stats_router
from routes.auth import router as auth_router
from routes.branches import router as branches_router
from routes.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events.

    Startup:
        - Preload YOLO model (SMALL) into memory for faster first request.

    Shutdown:
        - Cleanup temp directory (remove leftover files).
    """
    # --- STARTUP ---
    debug_info("=" * 60)
    debug_info("TRAFFIC COUNTER API — Starting up...")
    debug_info("=" * 60)

    # Preload default model
    try:
        detector: DetectorService = DetectorService()
        detector.preload("SMALL")
        debug_info(f"Device: {detector.get_device_info()}")
    except Exception as e:
        debug_error(f"Model preload failed: {e}")

    # Connect to MongoDB
    await connect_db()

    # Warn if JWT secret is still the dev default
    if JWT_SECRET_KEY == "dev-secret-change-me":
        debug_warning(
            "[Auth] JWT_SECRET_KEY is using the default dev value — "
            "set a strong secret in .env for production!"
        )

    # Store running event loop so video-job daemon threads can schedule
    # async MongoDB writes via asyncio.run_coroutine_threadsafe.
    set_event_loop(asyncio.get_running_loop())

    # On restart, mark any in-progress jobs (pending/downloading/processing)
    # as "error" — they cannot be resumed because the daemon threads were
    # killed when the server exited.  The frontend will show a clear error
    # message and prompt the user to re-submit.
    await recover_interrupted_jobs()

    debug_info("API is ready! Docs: http://localhost:3219/docs")
    debug_info("=" * 60)

    yield

    # --- SHUTDOWN ---
    debug_info("Shutting down... cleaning up temp files")
    await close_db()
    _cleanup_temp_dir()
    debug_info("Goodbye!")


def _cleanup_temp_dir() -> None:
    """Remove all temporary files from temp directory."""
    temp_files: list[str] = glob.glob(str(TEMP_DIR / "*"))
    for f in temp_files:
        try:
            os.remove(f)
            debug_info(f"Cleaned up: {f}")
        except OSError:
            pass


# =============================================
# REQUEST LOGGING MIDDLEWARE
# =============================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request/response to requests.log.

    Captures: client IP, method, path, query params, status code, latency,
    JSON request body (with RTSP credentials masked), file upload metadata
    (set on request.state.upload_info by route handlers), and JSON response body.
    Binary responses (images, videos) are not buffered — only their status is logged.
    WebSocket upgrades are passed through; those are logged by the route handlers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # WebSocket upgrades — handled individually in route handlers
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # CORS preflight — skip (no business logic, would be noisy)
        if request.method == "OPTIONS":
            return await call_next(request)

        start: float = time.perf_counter()

        # Client IP — prefer X-Forwarded-For when behind Nginx proxy
        forwarded: str = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        client_ip: str = forwarded or (request.client.host if request.client else "unknown")

        # Buffer JSON request body only — never touch multipart/file uploads.
        # mask_secrets strips passwords and tokens before they hit the log file
        # (otherwise POST /auth/login would write the user's plaintext password
        # into requests.log).
        req_body_str: str | None = None
        if "application/json" in request.headers.get("content-type", ""):
            raw_bytes: bytes = await request.body()
            if raw_bytes:
                req_body_str = mask_secrets(mask_rtsp(raw_bytes.decode("utf-8", errors="replace")))

        # Query string
        query_str: str | None = str(request.query_params) if request.query_params else None

        # Run the actual route handler
        response: Response = await call_next(request)

        latency_ms: int = round((time.perf_counter() - start) * 1000)

        # File/video upload info set by route handlers via request.state
        upload_info: dict | None = getattr(request.state, "upload_info", None)

        # Buffer JSON response only — skip binary (image/video) responses
        res_body_str: str | None = None
        if "application/json" in response.headers.get("content-type", ""):
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            res_bytes: bytes = b"".join(chunks)
            # Mask access_token / refresh_token in the LOGGED copy only — the
            # response sent to the client (res_bytes below) keeps the real
            # values intact.
            res_body_str = mask_secrets(res_bytes.decode("utf-8", errors="replace"))
            # Reconstruct response (body iterator was consumed)
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            response = Response(
                content=res_bytes,
                status_code=response.status_code,
                headers=headers,
                media_type="application/json",
            )

        log_http_request(
            ip=client_ip,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
            query=query_str,
            req_body=req_body_str,
            upload_info=upload_info,
            res_body=res_body_str,
        )

        return response


# =============================================
# FASTAPI APPLICATION
# =============================================

app: FastAPI = FastAPI(
    title="🚦 Traffic Counter API",
    description="""
## API Penghitung Lalu Lintas Real-Time

Menggunakan model **YOLOv11s/m** yang sudah di-training khusus untuk deteksi kendaraan Indonesia.

### Classes yang Dideteksi
| Class | Deskripsi |
|-------|-----------|
| `big-vehicle` | Truk, bus, kendaraan besar |
| `car` | Mobil, sedan, SUV |
| `pedestrian` | Pejalan kaki |
| `two-wheeler` | Motor, sepeda |

### Fitur Utama
- 🖼️ **Image Detection** — Upload gambar, dapatkan deteksi JSON atau annotated image
- 🎬 **Video Processing** — Upload video, dapatkan counting JSON atau annotated video
- 📡 **RTSP Stream** — Koneksi ke CCTV, dapatkan counting snapshot atau real-time WebSocket stream
- ⚙️ **Configurable** — Atur confidence, IoU, model size, dan posisi counting line

### Konfigurasi Model
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `SMALL` (YOLOv11s) | 19 MB | ⚡ Fast | Good |
| `MEDIUM` (YOLOv11m) | 40 MB | 🐢 Slower | Better |

### Smart Features
- **Pedestrian Filter** — Otomatis menghilangkan driver/rider dari hitungan pedestrian
- **ByteTrack** — Multi-object tracking untuk menghindari double counting
- **GPU Acceleration** — Otomatis menggunakan CUDA GPU jika tersedia
    """,
    version="1.6.0",
    lifespan=lifespan,
)

# =============================================
# MIDDLEWARE STACK
# =============================================
# Middleware is applied in reverse registration order:
# last registered = outermost (runs first).
# We register RequestLogging first so CORS ends up outermost,
# meaning CORS handles OPTIONS preflight before logging sees it.

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Detections-Count",
        "X-Total-Count",
        "X-Frames-Processed",
        "X-Processing-Time",
    ],
)

# =============================================
# INCLUDE ROUTERS
# =============================================

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(branches_router)
app.include_router(image_router)
app.include_router(video_router)
app.include_router(video_jobs_router)
app.include_router(rtsp_router)
app.include_router(ezviz_router)
app.include_router(stats_router)


# =============================================
# ROOT ENDPOINT
# =============================================

@app.get(
    "/",
    summary="API Health Check",
    description="Root endpoint — returns API info and health status.",
    tags=["🏠 General"],
)
async def root() -> dict:
    """API health check and info endpoint.

    Returns:
        Dictionary with API name, version, status, and endpoint links.
    """
    detector: DetectorService = DetectorService()

    return {
        "name": "Traffic Counter API",
        "version": "1.6.0",
        "status": "running",
        "model": detector.get_active_model_name(),
        "device": detector.get_device_info(),
        "endpoints": {
            "docs": "/docs",
            "auth_register": "POST /auth/register",
            "auth_login": "POST /auth/login",
            "auth_refresh": "POST /auth/refresh",
            "auth_me": "GET /auth/me",
            "branches": "GET/POST/PATCH/DELETE /branches",
            "image_detect": "POST /image/detect",
            "image_annotate": "POST /image/annotate",
            "video_detect": "POST /video/detect",
            "video_annotate": "POST /video/annotate",
            "rtsp_detect": "POST /rtsp/detect",
            "rtsp_stream": "WS /rtsp/stream",
            "ezviz_status": "GET /ezviz/status",
            "ezviz_devices": "GET /ezviz/devices",
            "ezviz_detect": "POST /ezviz/detect",
            "ezviz_stream": "WS /ezviz/stream",
            "logs": "POST/GET/DELETE /logs",
            "stats_hourly": "GET /stats/hourly",
            "stats_summary": "GET /stats/summary",
        },
        "classes": ["big-vehicle", "car", "pedestrian", "two-wheeler"],
        "models": {
            "SMALL": "YOLOv11s (best-s.pt) — 19 MB, faster",
            "MEDIUM": "YOLOv11m (best-m.pt) — 40 MB, more accurate",
        },
    }


# =============================================
# RUN SERVER
# =============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3219,
        reload=True,
    )
