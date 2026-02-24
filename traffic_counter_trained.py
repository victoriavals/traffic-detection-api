"""
Traffic Counter menggunakan Model YOLOv11s yang Telah Ditraining.

Model: best.pt (YOLOv11s, trained on traffic-merged.v2 dataset)
Classes: big-vehicle, car, pedestrian, two-wheeler

Menggunakan supervision library untuk ByteTrack tracking dan LineZone counting.
"""

import cv2
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import os
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO
 
# =============================================
# VIDEO SOURCE CONFIGURATION
# =============================================
# Pilih source yang ingin digunakan:
# "RTSP"    -> CCTV EZVIZ Real-time (192.168.1.72)
# "ROMLAH1" -> File video data/video-romlah1.mp4
# "ROMLAH2" -> File video data/test romlah.mp4
# "YOUTUBE" -> Livestream YouTube
SOURCE_SELECTION = "RTSP" 

# Dictionary Sumber Video
SOURCES = {
    "RTSP": "rtsp://admin:1Triliun@192.168.1.72:554/H.264",
    "ROMLAH1": "data/video-romlah1.mp4",
    "ROMLAH2": "data/test romlah.mp4",
    "YOUTUBE": "https://www.youtube.com/watch?v=2Xi9VIWiv6A"
}

# =============================================
# MODEL SELECTION
# =============================================
# Pilih model yang ingin digunakan:
# "SMALL"  -> YOLOv11s (best-s.pt) - Lebih cepat, cocok untuk real-time
# "MEDIUM" -> YOLOv11m (best-m.pt) - Lebih akurat, tapi lebih lambat
MODEL_SELECTION = "SMALL"

MODELS = {
    "SMALL": "best-s.pt",
    "MEDIUM": "best-m.pt",
}

# =============================================
# CONFIGURATION
# =============================================

# Path ke model yang sudah ditraining
MODEL_PATH: str = str(
    Path(__file__).resolve().parent / MODELS.get(MODEL_SELECTION, "best-s.pt")
)

# Set VIDEO_SOURCE based on selection
VIDEO_SOURCE: str = SOURCES.get(SOURCE_SELECTION, SOURCES["RTSP"])

# Set environment variable untuk RTSP (biar stabil pakai TCP)
if SOURCE_SELECTION == "RTSP":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Inference settings
CONFIDENCE_THRESHOLD: float = 0.45  # Dinaikkan dari 0.3 berdasarkan F1 curve analysis
IOU_THRESHOLD: float = 0.5

# Custom class names dari training
CLASS_NAMES: dict[int, str] = {
    0: "big-vehicle",
    1: "car",
    2: "pedestrian",
    3: "two-wheeler",
}

# Warna per kelas (BGR format)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 0),    # big-vehicle  → biru
    1: (255, 0, 0),      # car          → biru tua
    2: (0, 255, 0),      # pedestrian   → hijau
    3: (0, 165, 255),    # two-wheeler  → orange
}

# Kategori untuk counting
VEHICLE_CLASS_IDS: list[int] = [0, 1, 3]  # big-vehicle, car, two-wheeler


# =============================================
# DATA MODELS
# =============================================

@dataclass
class TrafficCount:
    """Counter statistik traffic per kelas.

    Attributes:
        big_vehicle (int): Jumlah big-vehicle yang terdeteksi.
        car (int): Jumlah car yang terdeteksi.
        pedestrian (int): Jumlah pedestrian yang terdeteksi.
        two_wheeler (int): Jumlah two-wheeler yang terdeteksi.
    """
    big_vehicle: int = 0
    car: int = 0
    pedestrian: int = 0
    two_wheeler: int = 0

    def total(self) -> int:
        """Hitung total semua kelas.

        Returns:
            int: Total count semua kelas.
        """
        return self.big_vehicle + self.car + self.pedestrian + self.two_wheeler

    def increment(self, class_name: str) -> None:
        """Tambah count untuk kelas tertentu.

        Args:
            class_name (str): Nama kelas yang akan ditambah.
        """
        if class_name == "big-vehicle":
            self.big_vehicle += 1
        elif class_name == "car":
            self.car += 1
        elif class_name == "pedestrian":
            self.pedestrian += 1
        elif class_name == "two-wheeler":
            self.two_wheeler += 1


# =============================================
# HELPER FUNCTIONS
# =============================================

def detect_device() -> int | str:
    """Deteksi device terbaik untuk inference (GPU/CPU).

    Returns:
        int | str: Device index (0 untuk GPU) atau 'cpu'.
    """
    if torch.cuda.is_available():
        gpu_name: str = torch.cuda.get_device_name(0)
        gpu_mem: float = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
        return 0
    print("[CPU] No GPU detected, using CPU (slower inference)")
    return "cpu"


def get_youtube_stream_url(youtube_url: str) -> Optional[str]:
    """Resolve YouTube URL ke direct stream URL menggunakan yt-dlp.

    Args:
        youtube_url (str): URL YouTube video/livestream.

    Returns:
        Optional[str]: Direct stream URL, atau None jika gagal.
    """
    try:
        print("[INFO] Resolving YouTube stream URL...")

        # Prefer 720p untuk performa optimal
        cmd: list[str] = [
            "yt-dlp", "-f", "best[height<=720]", "-g", youtube_url
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0 and result.stdout.strip().startswith("http"):
            print("[INFO] Stream URL obtained (720p)")
            return result.stdout.strip()

        # Fallback ke 'best'
        print("[WARN] 720p not found, trying 'best'...")
        cmd = ["yt-dlp", "-f", "best", "-g", youtube_url]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0 and result.stdout.strip().startswith("http"):
            return result.stdout.strip()

    except FileNotFoundError:
        print("[ERROR] yt-dlp not found. Install: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        print("[ERROR] yt-dlp timeout.")
    except Exception as e:
        print(f"[ERROR] Stream resolution failed: {e}")

    return None


def open_video_source(source: str) -> Optional[cv2.VideoCapture]:
    """Buka video source (YouTube URL, HTTP/RTSP stream, atau file lokal).

    Args:
        source (str): URL stream atau path ke video file.

    Returns:
        Optional[cv2.VideoCapture]: VideoCapture object, atau None jika gagal.
    """
    cap: Optional[cv2.VideoCapture] = None

    if "youtube.com" in source or "youtu.be" in source:
        stream_url: Optional[str] = get_youtube_stream_url(source)
        if stream_url:
            cap = cv2.VideoCapture(stream_url)
    elif source.startswith(("rtsp://", "http://", "https://", "rtsps://")):
        # Direct stream URL (RTSP/HTTP/HTTPS)
        print(f"[INFO] Opening stream: {source}")
        cap = cv2.VideoCapture(source)
    else:
        video_path: Path = Path(source)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {source}")
            return None
        cap = cv2.VideoCapture(str(video_path))

    if cap is not None and not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return None

    return cap


def filter_pedestrian_on_vehicle(
    detections: sv.Detections,
    overlap_threshold: float = 0.4,
) -> sv.Detections:
    """Filter pedestrian yang overlap dengan kendaraan (menghilangkan driver/rider).

    Args:
        detections (sv.Detections): Detections dari YOLO.
        overlap_threshold (float): Threshold overlap IoA (default 0.4).

    Returns:
        sv.Detections: Detections yang sudah difilter.
    """
    if len(detections) == 0:
        return detections

    class_ids: np.ndarray = detections.class_id
    xyxy: np.ndarray = detections.xyxy

    # Cari index pedestrian dan vehicle
    pedestrian_mask: np.ndarray = class_ids == 2  # pedestrian class_id = 2
    vehicle_mask: np.ndarray = np.isin(class_ids, VEHICLE_CLASS_IDS)

    # Jika tidak ada pedestrian atau vehicle, skip
    if not np.any(pedestrian_mask) or not np.any(vehicle_mask):
        return detections

    # Hitung IoA (Intersection over Area) untuk setiap pedestrian vs vehicle
    keep_mask: np.ndarray = np.ones(len(detections), dtype=bool)

    pedestrian_indices: np.ndarray = np.where(pedestrian_mask)[0]
    vehicle_indices: np.ndarray = np.where(vehicle_mask)[0]

    for p_idx in pedestrian_indices:
        p_box: np.ndarray = xyxy[p_idx]
        p_area: float = float(
            (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
        )

        if p_area <= 0:
            continue

        for v_idx in vehicle_indices:
            v_box: np.ndarray = xyxy[v_idx]

            # Hitung intersection
            xi1: float = max(float(p_box[0]), float(v_box[0]))
            yi1: float = max(float(p_box[1]), float(v_box[1]))
            xi2: float = min(float(p_box[2]), float(v_box[2]))
            yi2: float = min(float(p_box[3]), float(v_box[3]))

            if xi2 <= xi1 or yi2 <= yi1:
                continue

            intersection: float = (xi2 - xi1) * (yi2 - yi1)
            ioa: float = intersection / p_area

            if ioa > overlap_threshold:
                keep_mask[p_idx] = False
                break

    return detections[keep_mask]


# =============================================
# MAIN APPLICATION
# =============================================

def main() -> None:
    """Main traffic counter menggunakan model YOLOv11s yang telah ditraining."""

    print("=" * 60)
    print("TRAFFIC COUNTER - TRAINED YOLOv11s MODEL")
    print("=" * 60)

    # --- 1. LOAD MODEL ---
    model_path: Path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[HINT] Pastikan training sudah selesai dan best.pt tersedia.")
        sys.exit(1)

    device: int | str = detect_device()
    print(f"[INFO] Loading model: {model_path.name}")

    model: YOLO = YOLO(str(model_path))
    print(f"[INFO] Model loaded: {model.model_name}")
    print(f"[INFO] Classes: {list(CLASS_NAMES.values())}")
    print(f"[INFO] Confidence: {CONFIDENCE_THRESHOLD}")

    # --- 2. OPEN VIDEO SOURCE ---
    print(f"\n[INFO] Source: {VIDEO_SOURCE}")
    cap: Optional[cv2.VideoCapture] = open_video_source(VIDEO_SOURCE)

    if cap is None:
        print("[ERROR] Failed to open video source.")
        sys.exit(1)

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps: float = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Stream: {width}x{height} @ {video_fps:.1f} FPS")

    # --- 3. SETUP TRACKING & COUNTING ---
    tracker: sv.ByteTrack = sv.ByteTrack(
        lost_track_buffer=30,
        frame_rate=int(video_fps) if video_fps > 0 else 30,
    )

    # LineZone diagonal: kiri-atas ke kanan-bawah
    line_start: sv.Point = sv.Point(0, int(height * 0.15))
    line_end: sv.Point = sv.Point(width, int(height * 0.65))
    line_zone: sv.LineZone = sv.LineZone(start=line_start, end=line_end)

    # Annotators
    box_annotator: sv.BoxAnnotator = sv.BoxAnnotator(thickness=2)
    label_annotator: sv.LabelAnnotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    line_annotator: sv.LineZoneAnnotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.8,
    )

    # Traffic counter
    traffic_count: TrafficCount = TrafficCount()
    counted_tracker_ids: set[int] = set()

    # FPS tracking
    frame_count: int = 0
    fps_start: float = time.time()
    fps_frame_count: int = 0
    display_fps: float = 0.0

    print(f"\n[INFO] Counting line: ({line_start.x},{line_start.y}) → ({line_end.x},{line_end.y}) [diagonal]")
    print("[INFO] Press 'q' to quit")
    print("-" * 60)

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()

            if not ret:
                print("\n[INFO] Video stream ended.")
                break

            frame_count += 1
            fps_frame_count += 1

            # --- A. FPS CALCULATION ---
            elapsed: float = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start = time.time()

            # --- B. YOLO INFERENCE ---
            results = model(
                frame,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
            )

            # --- C. CONVERT TO SUPERVISION ---
            detections: sv.Detections = sv.Detections.from_ultralytics(
                results[0]
            )

            # --- D. FILTER DRIVERS ---
            detections = filter_pedestrian_on_vehicle(detections)

            # --- E. TRACKING ---
            detections = tracker.update_with_detections(detections)

            # --- F. LINE COUNTING ---
            crossed_in: np.ndarray
            crossed_out: np.ndarray
            crossed_in, crossed_out = line_zone.trigger(detections)

            # Hitung objek yang crossing (in atau out)
            for i in range(len(detections)):
                if crossed_in[i] or crossed_out[i]:
                    tracker_id: int = int(detections.tracker_id[i])

                    # Hindari double counting
                    if tracker_id in counted_tracker_ids:
                        continue

                    counted_tracker_ids.add(tracker_id)
                    class_id: int = int(detections.class_id[i])
                    class_name: str = CLASS_NAMES.get(class_id, "unknown")

                    traffic_count.increment(class_name)
                    direction: str = "IN" if crossed_in[i] else "OUT"
                    conf: float = float(detections.confidence[i])
                    print(
                        f"[COUNT] {direction} | {class_name} "
                        f"(conf={conf:.2f}) | Total: {traffic_count.total()}"
                    )

            # --- G. VISUALIZATION ---
            # Draw bounding boxes dengan warna per kelas
            labels: list[str] = []
            for i in range(len(detections)):
                cls_id: int = int(detections.class_id[i])
                cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
                conf_val: float = float(detections.confidence[i])
                t_id: int = int(detections.tracker_id[i])
                labels.append(f"#{t_id} {cls_name} {conf_val:.2f}")

            annotated: np.ndarray = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

            # Draw counting line
            line_annotator.annotate(
                frame=annotated, line_counter=line_zone
            )

            # --- H. STATS OVERLAY ---
            overlay_h: int = 200
            cv2.rectangle(annotated, (5, 5), (300, overlay_h), (0, 0, 0), -1)
            cv2.rectangle(annotated, (5, 5), (300, overlay_h), (255, 255, 255), 1)

            cv2.putText(
                annotated, f"FPS: {display_fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                annotated, f"Big-Vehicle: {traffic_count.big_vehicle}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                CLASS_COLORS[0], 2,
            )
            cv2.putText(
                annotated, f"Car: {traffic_count.car}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                CLASS_COLORS[1], 2,
            )
            cv2.putText(
                annotated, f"Pedestrian: {traffic_count.pedestrian}",
                (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                CLASS_COLORS[2], 2,
            )
            cv2.putText(
                annotated, f"Two-Wheeler: {traffic_count.two_wheeler}",
                (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                CLASS_COLORS[3], 2,
            )
            cv2.putText(
                annotated, f"TOTAL: {traffic_count.total()}",
                (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2,
            )

            # Display
            cv2.imshow("Traffic Counter - YOLOv11s Trained", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # --- FINAL REPORT ---
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Model:        {model_path.name}")
        print(f"  Confidence:   {CONFIDENCE_THRESHOLD}")
        print(f"  Frames:       {frame_count}")
        print("-" * 60)
        print(f"  Big-Vehicle:  {traffic_count.big_vehicle}")
        print(f"  Car:          {traffic_count.car}")
        print(f"  Pedestrian:   {traffic_count.pedestrian}")
        print(f"  Two-Wheeler:  {traffic_count.two_wheeler}")
        print("-" * 60)
        print(f"  TOTAL:        {traffic_count.total()}")
        print("=" * 60)


if __name__ == "__main__":
    main()
