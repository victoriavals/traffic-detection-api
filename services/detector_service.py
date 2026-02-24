"""
YOLO Detector Service.

Singleton service that manages YOLO model loading and inference.
Supports lazy loading and caching of multiple model sizes (SMALL/MEDIUM).
"""

from typing import Optional, Literal

import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO

from constant_var import (
    MODEL_PATHS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    debug_info,
    debug_error,
)


class DetectorService:
    """YOLO object detection service with model caching.

    Manages YOLO model lifecycle: lazy loading, GPU detection, and inference.
    Models are cached in memory for reuse across requests.

    Attributes:
        _models: Cache of loaded YOLO models keyed by size.
        _device: Auto-detected device (GPU index or 'cpu').
    """

    _instance: Optional["DetectorService"] = None

    def __new__(cls) -> "DetectorService":
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._device = cls._detect_device()
        return cls._instance

    @staticmethod
    def _detect_device() -> int | str:
        """Detect best available device for inference.

        Returns:
            Device index (0 for GPU) or 'cpu'.
        """
        if torch.cuda.is_available():
            gpu_name: str = torch.cuda.get_device_name(0)
            gpu_mem: float = torch.cuda.get_device_properties(0).total_memory / 1024**3
            debug_info(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
            return 0
        debug_info("[CPU] No GPU detected, using CPU")
        return "cpu"

    def load_model(self, model_size: Literal["SMALL", "MEDIUM"] = "SMALL") -> YOLO:
        """Load and cache YOLO model by size.

        Args:
            model_size: Model variant to load ('SMALL' or 'MEDIUM').

        Returns:
            Loaded YOLO model instance.

        Raises:
            FileNotFoundError: If model weights file doesn't exist.
        """
        if model_size in self._models:
            return self._models[model_size]

        model_path: str = MODEL_PATHS.get(model_size, MODEL_PATHS["SMALL"])
        debug_info(f"Loading YOLO model: {model_size} ({model_path})")

        model: YOLO = YOLO(model_path)
        self._models[model_size] = model

        debug_info(f"Model loaded: {model.model_name}")
        return model

    def detect(
        self,
        frame: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        model_size: Literal["SMALL", "MEDIUM"] = "SMALL",
    ) -> sv.Detections:
        """Run YOLO inference on a single frame.

        Args:
            frame: Input image as numpy array (BGR format).
            confidence: Confidence threshold (0.0 - 1.0).
            iou: IoU threshold for NMS (0.0 - 1.0).
            model_size: Which model to use.

        Returns:
            Supervision Detections object with all detected objects.
        """
        model: YOLO = self.load_model(model_size)

        results = model(
            frame,
            verbose=False,
            conf=confidence,
            iou=iou,
            device=self._device,
        )

        detections: sv.Detections = sv.Detections.from_ultralytics(results[0])
        return detections

    def get_device_info(self) -> str:
        """Get current device info string.

        Returns:
            Device description string.
        """
        if isinstance(self._device, int):
            return f"GPU: {torch.cuda.get_device_name(self._device)}"
        return "CPU"

    def preload(self, model_size: Literal["SMALL", "MEDIUM"] = "SMALL") -> None:
        """Preload model into memory (call during app startup).

        Args:
            model_size: Which model to preload.
        """
        try:
            self.load_model(model_size)
            debug_info(f"Model {model_size} preloaded successfully")
        except Exception as e:
            debug_error(f"Failed to preload model {model_size}: {e}")
