"""
Traffic Counter API — FastAPI Application.

Real-time traffic detection, tracking, and counting API
powered by YOLOv11s/m trained model.

Classes: big-vehicle, car, pedestrian, two-wheeler
Supports: Image upload, Video upload, RTSP stream

Author: Naufal Firdaus
"""

import os
import glob
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from constant_var import TEMP_DIR, debug_info, debug_error
from services.detector_service import DetectorService
from routes.image import router as image_router
from routes.video import router as video_router
from routes.video_jobs import router as video_jobs_router
from routes.rtsp import router as rtsp_router


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

    debug_info("API is ready! Docs: http://localhost:8000/docs")
    debug_info("=" * 60)

    yield

    # --- SHUTDOWN ---
    debug_info("Shutting down... cleaning up temp files")
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
    version="1.0.0",
    lifespan=lifespan,
)

# =============================================
# CORS MIDDLEWARE (for frontend web)
# =============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
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

app.include_router(image_router)
app.include_router(video_router)
app.include_router(video_jobs_router)
app.include_router(rtsp_router)


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
        "version": "1.0.0",
        "status": "running",
        "device": detector.get_device_info(),
        "endpoints": {
            "docs": "/docs",
            "image_detect": "POST /image/detect",
            "image_annotate": "POST /image/annotate",
            "video_detect": "POST /video/detect",
            "video_annotate": "POST /video/annotate",
            "rtsp_detect": "POST /rtsp/detect",
            "rtsp_stream": "WS /rtsp/stream",
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
        port=8000,
        reload=True,
    )
