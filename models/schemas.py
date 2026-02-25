"""
Pydantic Models (Schemas) for Traffic Counter API.

Request and response models for all endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# =============================================
# CONFIG MODELS
# =============================================

class LineConfig(BaseModel):
    """Konfigurasi posisi counting line (sebagai persentase 0.0 - 1.0).

    Attributes:
        start_x: Posisi X awal line (0.0 = kiri, 1.0 = kanan).
        start_y: Posisi Y awal line (0.0 = atas, 1.0 = bawah).
        end_x: Posisi X akhir line.
        end_y: Posisi Y akhir line.
    """

    start_x: float = Field(0.0, ge=0.0, le=1.0, description="Start X position (0.0=left, 1.0=right)")
    start_y: float = Field(0.15, ge=0.0, le=1.0, description="Start Y position (0.0=top, 1.0=bottom)")
    end_x: float = Field(1.0, ge=0.0, le=1.0, description="End X position")
    end_y: float = Field(0.65, ge=0.0, le=1.0, description="End Y position")


class RTSPRequest(BaseModel):
    """Request body untuk RTSP detect endpoint.

    Attributes:
        url: URL RTSP stream (e.g., rtsp://user:pass@ip:port/path).
        confidence: Confidence threshold untuk deteksi (0.0 - 1.0).
        iou: IoU threshold untuk NMS (0.0 - 1.0).
        model_size: Ukuran model ('SMALL' = YOLOv11s, 'MEDIUM' = YOLOv11m).
        frame_count: Jumlah frame yang akan diproses.
        line_config: Konfigurasi posisi counting line.
    """

    url: str = Field(..., description="RTSP stream URL (e.g., rtsp://user:pass@ip:554/path)")
    confidence: float = Field(0.45, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou: float = Field(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")
    model_size: Literal["SMALL", "MEDIUM"] = Field("SMALL", description="Model size: SMALL (faster) or MEDIUM (more accurate)")
    frame_count: int = Field(150, ge=1, le=1000, description="Number of frames to process")
    line_config: Optional[LineConfig] = Field(None, description="Counting line position config (optional)")


# =============================================
# DETECTION ITEMS
# =============================================

class BoundingBox(BaseModel):
    """Bounding box koordinat dalam pixel.

    Attributes:
        x1: Koordinat X kiri atas.
        y1: Koordinat Y kiri atas.
        x2: Koordinat X kanan bawah.
        y2: Koordinat Y kanan bawah.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    """Single detected object.

    Attributes:
        class_name: Nama kelas objek (e.g., 'car', 'big-vehicle').
        class_id: ID kelas objek.
        confidence: Skor confidence deteksi.
        bbox: Bounding box koordinat.
        tracker_id: ID tracker (jika tracking aktif, None untuk image).
    """

    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    tracker_id: Optional[int] = None


# =============================================
# RESPONSE MODELS
# =============================================

class ClassCount(BaseModel):
    """Jumlah deteksi per kelas.

    Attributes:
        big_vehicle: Jumlah big-vehicle terdeteksi.
        car: Jumlah car terdeteksi.
        pedestrian: Jumlah pedestrian terdeteksi.
        two_wheeler: Jumlah two-wheeler terdeteksi.
        total: Total semua kelas.
    """

    big_vehicle: int = 0
    car: int = 0
    pedestrian: int = 0
    two_wheeler: int = 0
    total: int = 0


class ImageDetectResponse(BaseModel):
    """Response untuk image detection endpoint.

    Attributes:
        success: Apakah deteksi berhasil.
        message: Pesan status.
        detections: List semua objek terdeteksi.
        summary: Ringkasan jumlah per kelas.
        inference_config: Konfigurasi yang digunakan.
    """

    success: bool
    message: str
    detections: list[DetectionItem]
    summary: ClassCount
    inference_config: dict


class VideoDetectResponse(BaseModel):
    """Response untuk video detection endpoint.

    Attributes:
        success: Apakah proses berhasil.
        message: Pesan status.
        counts: Jumlah kendaraan per kelas yang melewati counting line.
        video_info: Informasi video (resolution, fps, total frames).
        inference_config: Konfigurasi yang digunakan.
    """

    success: bool
    message: str
    counts: ClassCount
    video_info: dict
    inference_config: dict


class RTSPDetectResponse(BaseModel):
    """Response untuk RTSP snapshot detect endpoint.

    Attributes:
        success: Apakah proses berhasil.
        message: Pesan status.
        counts: Jumlah kendaraan per kelas yang melewati counting line.
        stream_info: Informasi stream (resolution, fps, frames processed).
        inference_config: Konfigurasi yang digunakan.
    """

    success: bool
    message: str
    counts: ClassCount
    stream_info: dict
    inference_config: dict


class RTSPStreamMessage(BaseModel):
    """WebSocket message untuk RTSP stream.

    Dikirim dari server ke client setiap frame.

    Attributes:
        type: Tipe message ('frame', 'error', 'info').
        frame: Base64 encoded JPEG frame (jika type='frame').
        counts: Live counting data.
        fps: Processing FPS saat ini.
        frame_number: Nomor frame saat ini.
        message: Pesan text (jika type='info' atau 'error').
    """

    type: Literal["frame", "error", "info"]
    frame: Optional[str] = None
    counts: Optional[ClassCount] = None
    fps: Optional[float] = None
    frame_number: Optional[int] = None
    message: Optional[str] = None


# =============================================
# VIDEO JOB MODELS (URL-based async processing)
# =============================================

class VideoJobRequest(BaseModel):
    """Request body untuk membuat video processing job dari URL publik.

    Attributes:
        url: URL publik video (Google Drive, Dropbox, direct link, dll).
        confidence: Confidence threshold untuk deteksi.
        iou: IoU threshold untuk NMS.
        model_size: Ukuran model YOLO.
        line_start_x: Posisi X awal counting line.
        line_start_y: Posisi Y awal counting line.
        line_end_x: Posisi X akhir counting line.
        line_end_y: Posisi Y akhir counting line.
    """

    url: str = Field(..., description="Public video URL (Google Drive, Dropbox, direct MP4 link, etc.)")
    confidence: float = Field(0.45, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou: float = Field(0.5, ge=0.0, le=1.0, description="IoU threshold for NMS")
    model_size: Literal["SMALL", "MEDIUM"] = Field("SMALL", description="Model: SMALL (faster) or MEDIUM (more accurate)")
    line_start_x: float = Field(0.0, ge=0.0, le=1.0)
    line_start_y: float = Field(0.15, ge=0.0, le=1.0)
    line_end_x: float = Field(1.0, ge=0.0, le=1.0)
    line_end_y: float = Field(0.65, ge=0.0, le=1.0)


class VideoJobStatus(BaseModel):
    """Status dan hasil dari video processing job.

    Attributes:
        job_id: Unique job identifier.
        status: Status saat ini ('pending', 'downloading', 'processing', 'done', 'error').
        progress: Persentase penyelesaian (0-100).
        message: Pesan status terbaru.
        counts: Hasil counting (tersedia jika status='done').
        video_info: Informasi video yang diproses.
        inference_config: Konfigurasi inference yang digunakan.
        error: Pesan error (tersedia jika status='error').
    """

    job_id: str
    status: Literal["pending", "downloading", "processing", "done", "error"]
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    counts: Optional[ClassCount] = None
    video_info: Optional[dict] = None
    inference_config: Optional[dict] = None
    error: Optional[str] = None
