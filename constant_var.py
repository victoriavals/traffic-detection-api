"""
Central Configuration Hub for Traffic Counter API.

All environment variables, model paths, constants, and logging
wrappers are centralized here.
"""

import os
import logging
from pathlib import Path
from typing import Final

# =============================================
# PATHS
# =============================================

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
SMALL_MODEL_PATH: Final[str] = str(BASE_DIR / "best-s.pt")
MEDIUM_MODEL_PATH: Final[str] = str(BASE_DIR / "best-m.pt")
TEMP_DIR: Final[Path] = BASE_DIR / "temp"

# Ensure temp directory exists
TEMP_DIR.mkdir(exist_ok=True)

# =============================================
# MODEL CONFIGURATION
# =============================================

MODEL_PATHS: Final[dict[str, str]] = {
    "SMALL": SMALL_MODEL_PATH,
    "MEDIUM": MEDIUM_MODEL_PATH,
}

DEFAULT_MODEL_SIZE: Final[str] = "SMALL"
DEFAULT_CONFIDENCE: Final[float] = 0.45
DEFAULT_IOU: Final[float] = 0.5

# =============================================
# CLASS DEFINITIONS (from trained model)
# =============================================

CLASS_NAMES: Final[dict[int, str]] = {
    0: "big-vehicle",
    1: "car",
    2: "pedestrian",
    3: "two-wheeler",
}

CLASS_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    0: (255, 100, 0),    # big-vehicle  → blue
    1: (255, 0, 0),      # car          → dark blue
    2: (0, 255, 0),      # pedestrian   → green
    3: (0, 165, 255),    # two-wheeler  → orange
}

VEHICLE_CLASS_IDS: Final[list[int]] = [0, 1, 3]  # big-vehicle, car, two-wheeler

# =============================================
# LINE ZONE DEFAULTS (as percentage 0.0 - 1.0)
# =============================================

DEFAULT_LINE_START_X: Final[float] = 0.0
DEFAULT_LINE_START_Y: Final[float] = 0.15
DEFAULT_LINE_END_X: Final[float] = 1.0
DEFAULT_LINE_END_Y: Final[float] = 0.65

# =============================================
# API LIMITS
# =============================================

MAX_UPLOAD_SIZE_MB: Final[int] = 500
MAX_IMAGE_SIZE_MB: Final[int] = 50
RTSP_DEFAULT_FRAME_COUNT: Final[int] = 150
RTSP_MAX_FRAME_COUNT: Final[int] = 1000
WEBSOCKET_FPS_LIMIT: Final[int] = 15

# =============================================
# LOGGING (File + Console)
# =============================================

LOGS_DIR: Final[Path] = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log format
_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(message)s"
_LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_formatter: logging.Formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

# Logger instance
logger: logging.Logger = logging.getLogger("traffic-counter-api")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Console handler (INFO+)
_console_handler: logging.StreamHandler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

# File handler: app.log (INFO+) — General application logs
_app_file_handler: logging.FileHandler = logging.FileHandler(
    str(LOGS_DIR / "app.log"), encoding="utf-8"
)
_app_file_handler.setLevel(logging.INFO)
_app_file_handler.setFormatter(_formatter)
logger.addHandler(_app_file_handler)

# File handler: details.log (DEBUG+) — Verbose debugging
_detail_file_handler: logging.FileHandler = logging.FileHandler(
    str(LOGS_DIR / "details.log"), encoding="utf-8"
)
_detail_file_handler.setLevel(logging.DEBUG)
_detail_file_handler.setFormatter(_formatter)
logger.addHandler(_detail_file_handler)

# File handler: errors.log (ERROR+) — Error-only logs
_error_file_handler: logging.FileHandler = logging.FileHandler(
    str(LOGS_DIR / "errors.log"), encoding="utf-8"
)
_error_file_handler.setLevel(logging.ERROR)
_error_file_handler.setFormatter(_formatter)
logger.addHandler(_error_file_handler)


def debug_info(msg: str) -> None:
    """Log info-level message (console + app.log + details.log)."""
    logger.info(msg)


def detail_debug(msg: str) -> None:
    """Log debug-level message (details.log only)."""
    logger.debug(msg)


def debug_error(msg: str) -> None:
    """Log error-level message (console + all log files)."""
    logger.error(msg)

