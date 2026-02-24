"""
Image Detection Routes.

Tag: Image Detection
Endpoints:
- POST /image/detect   → JSON deteksi per objek
- POST /image/annotate → Annotated image (JPEG)
"""

import io
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse

from constant_var import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_SIZE,
    CLASS_NAMES,
    debug_info,
    debug_error,
)
from models.schemas import (
    ImageDetectResponse,
    DetectionItem,
    BoundingBox,
    ClassCount,
)
from services.detector_service import DetectorService
from services.annotation_service import AnnotationService
from utils.pedestrian_filter import filter_pedestrian_on_vehicle


router = APIRouter(prefix="/image", tags=["🖼️ Image Detection"])

# Service instances
detector: DetectorService = DetectorService()
annotator: AnnotationService = AnnotationService()


async def _read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded image file into numpy array.

    Args:
        file: Uploaded image file.

    Returns:
        Image as BGR numpy array.

    Raises:
        HTTPException: If file is not a valid image.
    """
    content: bytes = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    np_arr: np.ndarray = np.frombuffer(content, np.uint8)
    image: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Supported formats: JPEG, PNG, BMP, TIFF"
        )

    return image


def _build_detection_result(
    detections,
    confidence: float,
    iou: float,
    model_size: str,
    image_shape: tuple,
) -> ImageDetectResponse:
    """Build JSON response from detections.

    Args:
        detections: Supervision Detections object.
        confidence: Confidence threshold used.
        iou: IoU threshold used.
        model_size: Model size used.
        image_shape: Original image shape (H, W, C).

    Returns:
        ImageDetectResponse with all detection details.
    """
    items: list[DetectionItem] = []
    counts: dict[str, int] = {
        "big_vehicle": 0,
        "car": 0,
        "pedestrian": 0,
        "two_wheeler": 0,
    }

    for i in range(len(detections)):
        cls_id: int = int(detections.class_id[i])
        cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
        conf: float = float(detections.confidence[i])
        box: np.ndarray = detections.xyxy[i]

        items.append(DetectionItem(
            class_name=cls_name,
            class_id=cls_id,
            confidence=round(conf, 4),
            bbox=BoundingBox(
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
            ),
        ))

        # Count per class
        key: str = cls_name.replace("-", "_")
        if key in counts:
            counts[key] += 1

    total: int = sum(counts.values())

    return ImageDetectResponse(
        success=True,
        message=f"Detected {len(items)} objects in image ({image_shape[1]}x{image_shape[0]})",
        detections=items,
        summary=ClassCount(
            big_vehicle=counts["big_vehicle"],
            car=counts["car"],
            pedestrian=counts["pedestrian"],
            two_wheeler=counts["two_wheeler"],
            total=total,
        ),
        inference_config={
            "confidence": confidence,
            "iou": iou,
            "model_size": model_size,
            "device": detector.get_device_info(),
            "image_size": f"{image_shape[1]}x{image_shape[0]}",
        },
    )


@router.post(
    "/detect",
    response_model=ImageDetectResponse,
    summary="Detect objects in image (JSON)",
    description="""
Upload sebuah gambar dan dapatkan hasil deteksi dalam format JSON.

**Response berisi:**
- List semua objek terdeteksi (class, confidence, bounding box)
- Summary jumlah per kelas (big-vehicle, car, pedestrian, two-wheeler)
- Konfigurasi inference yang digunakan

**Supported formats:** JPEG, PNG, BMP, TIFF

**Parameter opsional:**
- `confidence` — Threshold deteksi (default: 0.45, range: 0.0-1.0). Semakin tinggi = semakin ketat.
- `iou` — IoU threshold untuk NMS (default: 0.5). Mengontrol overlap antar detection box.
- `model_size` — Pilihan model: SMALL (lebih cepat) atau MEDIUM (lebih akurat).

**Note:** Endpoint ini tidak menggunakan tracking/counting karena hanya memproses 1 frame.
Pedestrian yang overlap dengan kendaraan (driver/rider) secara otomatis difilter.
    """,
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, TIFF)"),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0, description="Detection confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    model_size: Literal["SMALL", "MEDIUM"] = Query(DEFAULT_MODEL_SIZE, description="Model size: SMALL (faster) or MEDIUM (more accurate)"),
) -> ImageDetectResponse:
    """Detect objects in uploaded image and return JSON results."""
    debug_info(f"[IMAGE/DETECT] Processing: {file.filename} (conf={confidence}, model={model_size})")

    try:
        image: np.ndarray = await _read_image(file)
        detections = detector.detect(image, confidence, iou, model_size)
        detections = filter_pedestrian_on_vehicle(detections)

        response: ImageDetectResponse = _build_detection_result(
            detections, confidence, iou, model_size, image.shape
        )

        debug_info(f"[IMAGE/DETECT] Done: {len(detections)} objects detected")
        return response

    except HTTPException:
        raise
    except Exception as e:
        debug_error(f"[IMAGE/DETECT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post(
    "/annotate",
    summary="Detect and annotate image (JPEG)",
    description="""
Upload sebuah gambar dan dapatkan gambar hasil anotasi dengan bounding box dan label.

**Response:** Gambar JPEG dengan:
- Bounding box berwarna per kelas
- Label: nama kelas + confidence score
- Otomatis filter pedestrian yang overlap dengan kendaraan (driver/rider)

**Supported formats:** JPEG, PNG, BMP, TIFF

**Parameter opsional:**
- `confidence` — Threshold deteksi (default: 0.45, range: 0.0-1.0)
- `iou` — IoU threshold untuk NMS (default: 0.5)
- `model_size` — SMALL (lebih cepat) atau MEDIUM (lebih akurat)

**Content-Type response:** `image/jpeg`
    """,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Annotated image"},
        400: {"description": "Invalid image file"},
        500: {"description": "Detection failed"},
    },
)
async def annotate_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, TIFF)"),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.0, le=1.0, description="Detection confidence threshold"),
    iou: float = Query(DEFAULT_IOU, ge=0.0, le=1.0, description="IoU threshold for NMS"),
    model_size: Literal["SMALL", "MEDIUM"] = Query(DEFAULT_MODEL_SIZE, description="Model size: SMALL (faster) or MEDIUM (more accurate)"),
) -> StreamingResponse:
    """Detect objects in uploaded image and return annotated JPEG."""
    debug_info(f"[IMAGE/ANNOTATE] Processing: {file.filename} (conf={confidence}, model={model_size})")

    try:
        image: np.ndarray = await _read_image(file)
        detections = detector.detect(image, confidence, iou, model_size)
        detections = filter_pedestrian_on_vehicle(detections)

        # Annotate frame
        annotated: np.ndarray = annotator.annotate_detections(
            image, detections, show_tracker_id=False
        )

        # Encode to JPEG
        success: bool
        buffer: np.ndarray
        success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image")

        debug_info(f"[IMAGE/ANNOTATE] Done: {len(detections)} objects annotated")

        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'inline; filename="annotated_{file.filename}"',
                "X-Detections-Count": str(len(detections)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        debug_error(f"[IMAGE/ANNOTATE] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")
