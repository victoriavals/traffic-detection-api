"""
Frame Annotation Service.

Handles drawing bounding boxes, labels, counting lines,
and stats overlays on video frames.
"""

import numpy as np
import supervision as sv

from constant_var import CLASS_NAMES, CLASS_COLORS


class AnnotationService:
    """Service for annotating frames with detection results.

    Provides methods to draw bounding boxes, labels, counting lines,
    and statistics overlays using the Supervision library.
    """

    def __init__(self) -> None:
        """Initialize annotators."""
        self._box_annotator: sv.BoxAnnotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator: sv.LabelAnnotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1
        )
        self._line_annotator: sv.LineZoneAnnotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=0.8,
        )

    def annotate_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        show_tracker_id: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame (BGR numpy array).
            detections: Detected objects.
            show_tracker_id: Whether to show tracker IDs in labels.

        Returns:
            Annotated frame with boxes and labels.
        """
        labels: list[str] = []
        for i in range(len(detections)):
            cls_id: int = int(detections.class_id[i])
            cls_name: str = CLASS_NAMES.get(cls_id, "unknown")
            conf: float = float(detections.confidence[i])

            if show_tracker_id and detections.tracker_id is not None:
                t_id: int = int(detections.tracker_id[i])
                labels.append(f"#{t_id} {cls_name} {conf:.2f}")
            else:
                labels.append(f"{cls_name} {conf:.2f}")

        annotated: np.ndarray = self._box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated = self._label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated

    def draw_counting_line(
        self,
        frame: np.ndarray,
        line_zone: sv.LineZone,
    ) -> np.ndarray:
        """Draw the counting line on frame.

        Args:
            frame: Input frame (BGR numpy array).
            line_zone: Supervision LineZone object.

        Returns:
            Frame with counting line drawn.
        """
        self._line_annotator.annotate(frame=frame, line_counter=line_zone)
        return frame

    @staticmethod
    def draw_stats_overlay(
        frame: np.ndarray,
        counts: dict[str, int],
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw statistics overlay box on frame.

        Args:
            frame: Input frame (BGR numpy array).
            counts: Dictionary of class counts {'big_vehicle': N, ...}.
            fps: Current processing FPS.

        Returns:
            Frame with stats overlay.
        """
        import cv2

        overlay_h: int = 200
        cv2.rectangle(frame, (5, 5), (300, overlay_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, overlay_h), (255, 255, 255), 1)

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Big-Vehicle: {counts.get('big_vehicle', 0)}",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            CLASS_COLORS[0], 2,
        )
        cv2.putText(
            frame, f"Car: {counts.get('car', 0)}",
            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            CLASS_COLORS[1], 2,
        )
        cv2.putText(
            frame, f"Pedestrian: {counts.get('pedestrian', 0)}",
            (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            CLASS_COLORS[2], 2,
        )
        cv2.putText(
            frame, f"Two-Wheeler: {counts.get('two_wheeler', 0)}",
            (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            CLASS_COLORS[3], 2,
        )

        total: int = sum(counts.values())
        cv2.putText(
            frame, f"TOTAL: {total}",
            (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2,
        )

        return frame
