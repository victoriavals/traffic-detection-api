"""
Pedestrian-on-Vehicle Filter Utility.

Filters out pedestrian detections that overlap with vehicles,
which typically indicates drivers or riders rather than pedestrians.

Extracted from traffic_counter_trained.py for reuse across API endpoints.
"""

import numpy as np
import supervision as sv

from constant_var import VEHICLE_CLASS_IDS


def filter_pedestrian_on_vehicle(
    detections: sv.Detections,
    overlap_threshold: float = 0.4,
) -> sv.Detections:
    """Filter pedestrian yang overlap dengan kendaraan (menghilangkan driver/rider).

    Menggunakan IoA (Intersection over Area) untuk menentukan apakah
    bounding box pedestrian berada di dalam bounding box kendaraan.

    Args:
        detections: Detections dari YOLO inference.
        overlap_threshold: Threshold overlap IoA (default 0.4).

    Returns:
        Detections yang sudah difilter (tanpa driver/rider).
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
