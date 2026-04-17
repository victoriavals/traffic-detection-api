# 🚦 Traffic Detection API

> Backend API untuk sistem deteksi dan penghitungan kendaraan lalu lintas berbasis **YOLOv11**, dibangun sebagai bagian dari **Projek Sarjana Muda (PSM)**.

## Tech Stack

| Teknologi | Peran |
|---|---|
| **FastAPI** ≥0.115 | REST API + WebSocket framework |
| **Uvicorn** ≥0.34 | ASGI server with auto-reload |
| **Ultralytics YOLO** ≥8.3 | YOLOv11s/m object detection |
| **Supervision** ≥0.25 | ByteTrack tracking + LineZone counting |
| **OpenCV** ≥4.10 | Image/video processing |
| **PyTorch** ≥2.0 | GPU inference engine |
| **NumPy** ≥1.26 | Array/matrix operations |

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🖼️ **Image Detection** | Upload gambar → deteksi kendaraan (JSON + annotated JPEG) |
| 🎬 **Video Processing** | Upload video → hitung kendaraan + annotated MP4 |
| 📡 **RTSP Streaming** | Koneksi CCTV real-time via WebSocket |
| 🎯 **Pedestrian Filter** | Otomatis filter driver/rider dari hitungan pedestrian |
| 🧠 **Multi-Model** | Pilih SMALL (cepat) atau MEDIUM (akurat) |
| 📊 **File Logging** | 3-level log: `app.log`, `details.log`, `errors.log` |

### Kelas Kendaraan

- 🚛 **big-vehicle** — Truk, Bus
- 🚗 **car** — Mobil, Sedan, SUV
- 🚶 **pedestrian** — Pejalan Kaki
- 🏍️ **two-wheeler** — Motor, Sepeda

## Setup & Installation

### Prerequisites

- **Python** ≥ 3.10
- **NVIDIA GPU** (recommended, CUDA-compatible) atau CPU
- **Model weights**: `best-s.pt` dan/atau `best-m.pt` di root folder

### Installation

```bash
# Clone repository
git clone <YOUR_GIT_URL>
cd api-traffic-counter

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Place model weights in root folder
# best-s.pt  → YOLOv11s (19 MB)
# best-m.pt  → YOLOv11m (40 MB)
```

### Run Server

```bash
python main.py
# → Server running at http://localhost:3219
# → Swagger UI at http://localhost:3219/docs
```

## API Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| `GET` | `/` | — | Health check + device info |
| `POST` | `/image/detect` | Image file + params | JSON: detections + summary |
| `POST` | `/image/annotate` | Image file + params | JPEG: annotated image |
| `POST` | `/video/detect` | Video file + params | JSON: counting per class |
| `POST` | `/video/annotate` | Video file + params | MP4: annotated video |
| `POST` | `/rtsp/detect` | JSON body (RTSP URL) | JSON: snapshot counting |
| `WS` | `/rtsp/stream` | JSON config → frames | Real-time annotated stream |

### Parameters (Query)

| Parameter | Type | Default | Deskripsi |
|-----------|------|---------|-----------|
| `confidence` | float | 0.45 | Detection threshold (0.0-1.0) |
| `iou` | float | 0.5 | NMS IoU threshold (0.0-1.0) |
| `model_size` | string | SMALL | `SMALL` (cepat) atau `MEDIUM` (akurat) |
| `line_start_x/y` | float | 0.0/0.15 | Posisi awal counting line (%) |
| `line_end_x/y` | float | 1.0/0.65 | Posisi akhir counting line (%) |

## Struktur Project

```
api-traffic-counter/
├── main.py                    # FastAPI entry point + CORS + lifespan
├── constant_var.py            # Config hub + logging (3 file handlers)
├── requirements.txt           # Python dependencies
├── best-s.pt                  # YOLOv11s model weights (19 MB)
├── best-m.pt                  # YOLOv11m model weights (40 MB)
├── models/
│   └── schemas.py             # Pydantic request/response schemas
├── services/
│   ├── detector_service.py    # Singleton YOLO loader + inference
│   └── annotation_service.py  # Frame annotation (Supervision)
├── routes/
│   ├── image.py               # /image/detect, /image/annotate
│   ├── video.py               # /video/detect, /video/annotate
│   └── rtsp.py                # /rtsp/detect, WS /rtsp/stream
├── utils/
│   └── pedestrian_filter.py   # IoA-based driver/rider filter
├── logs/                      # app.log, details.log, errors.log
├── temp/                      # Temporary annotated video files
└── data/                      # Test images & videos
```

## Logging

Log files tersimpan di folder `logs/`:

| File | Level | Konten |
|------|-------|--------|
| `app.log` | INFO+ | Request processing, results |
| `details.log` | DEBUG+ | Verbose debug info |
| `errors.log` | ERROR+ | Error-only logs |

## Environment Variables

Tidak ada environment variable yang diperlukan. Semua konfigurasi terdefinisi di `constant_var.py`.

## License

© 2026 PSM — Traffic Detection API v1.0.0