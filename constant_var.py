"""
Central Configuration Hub for Traffic Counter API.

All environment variables, model paths, constants, and logging
wrappers are centralized here.
"""

import json as _json
import os
import re as _re
import sys as _sys
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

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
# MONGODB
# =============================================

MONGO_URI: Final[str] = os.getenv("MONGO_URI", "")
MONGO_DB_NAME: Final[str] = os.getenv("MONGO_DB_NAME", "traffic_counter")

# =============================================
# JWT AUTHENTICATION
# =============================================

JWT_SECRET_KEY: Final[str] = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_TO_A_RANDOM_256BIT_SECRET_IN_PRODUCTION")
JWT_ALGORITHM: Final[str] = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 jam

# =============================================
# EZVIZ CLOUD API
# =============================================

EZVIZ_APP_KEY: Final[str] = os.getenv("EZVIZ_APP_KEY", "")
EZVIZ_APP_SECRET: Final[str] = os.getenv("EZVIZ_APP_SECRET", "")
EZVIZ_API_BASE: Final[str] = "https://isgpopen.ezvizlife.com"
EZVIZ_TOKEN_URL: Final[str] = "https://open.ezvizlife.com/api/lapp/token/get"
EZVIZ_MAX_CAPTURE_FAILURES: Final[int] = 5  # Max consecutive capture failures before auto-stop

# =============================================
# LOGGING CONFIGURATION
# =============================================

# Set LOG_FORMAT=json in .env for production (Grafana/Loki-ready single-line JSON).
# Default "text" is human-readable for development.
LOG_FORMAT_TYPE: Final[str] = os.getenv("LOG_FORMAT", "text")
_LOG_MAX_BYTES: Final[int] = 50 * 1024 * 1024   # 50 MB per file before rotation
_LOG_BACKUP_COUNT: Final[int] = 5                # keep 5 rotated backups

LOGS_DIR: Final[Path] = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)-7s] %(message)s"
_LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


class _JsonFormatter(logging.Formatter):
    """Formats app log records as single-line JSON (for production log aggregators)."""
    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["traceback"] = self.formatException(record.exc_info)
        return _json.dumps(obj, ensure_ascii=False)


class _PassthroughFormatter(logging.Formatter):
    """For requests_logger — caller pre-formats the message, formatter just passes it through."""
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def _make_app_formatter() -> logging.Formatter:
    if LOG_FORMAT_TYPE == "json":
        return _JsonFormatter()
    return logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)


# ─── App logger (app.log / details.log / errors.log / console) ───────────────

logger: logging.Logger = logging.getLogger("traffic-counter")
logger.setLevel(logging.DEBUG)
logger.propagate = False

_app_fmt = _make_app_formatter()

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_app_fmt)
logger.addHandler(_console_handler)

_app_handler = RotatingFileHandler(
    str(LOGS_DIR / "app.log"), maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT, encoding="utf-8",
)
_app_handler.setLevel(logging.INFO)
_app_handler.setFormatter(_app_fmt)
logger.addHandler(_app_handler)

_detail_handler = RotatingFileHandler(
    str(LOGS_DIR / "details.log"), maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT, encoding="utf-8",
)
_detail_handler.setLevel(logging.DEBUG)
_detail_handler.setFormatter(_app_fmt)
logger.addHandler(_detail_handler)

_error_handler = RotatingFileHandler(
    str(LOGS_DIR / "errors.log"), maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT, encoding="utf-8",
)
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(_app_fmt)
logger.addHandler(_error_handler)


# ─── Requests logger (requests.log) ──────────────────────────────────────────
# Separate logger dedicated to HTTP access logs and WebSocket session events.
# Logs every request: IP, method, path, status, latency, JSON bodies, file info.

requests_logger: logging.Logger = logging.getLogger("traffic-counter-requests")
requests_logger.setLevel(logging.DEBUG)
requests_logger.propagate = False

_req_handler = RotatingFileHandler(
    str(LOGS_DIR / "requests.log"), maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT, encoding="utf-8",
)
_req_handler.setLevel(logging.DEBUG)
_req_handler.setFormatter(_PassthroughFormatter())
requests_logger.addHandler(_req_handler)


# ─── RTSP credential masking ─────────────────────────────────────────────────

_RTSP_CRED_RE = _re.compile(r"(rtsp://)[^:@/\"\\]*:[^@/\"\\]*@", _re.IGNORECASE)


def mask_rtsp(text: str) -> str:
    """Mask RTSP credentials: rtsp://user:pass@host → rtsp://***:***@host."""
    return _RTSP_CRED_RE.sub(r"\1***:***@", text)


# ─── App log wrappers ─────────────────────────────────────────────────────────

def debug_info(msg: str) -> None:
    """INFO — console + app.log + details.log."""
    logger.info(msg, stacklevel=2)


def debug_warning(msg: str) -> None:
    """WARNING — console + app.log + details.log. Use for recoverable issues."""
    logger.warning(msg, stacklevel=2)


def debug_error(msg: str) -> None:
    """ERROR — console + all log files. Auto-includes traceback inside except blocks."""
    logger.error(msg, stacklevel=2, exc_info=_sys.exc_info()[0] is not None)


def detail_debug(msg: str) -> None:
    """DEBUG — details.log only. Use for verbose per-frame or per-request data."""
    logger.debug(msg, stacklevel=2)


# ─── Structured request log helpers ──────────────────────────────────────────

def _ts_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log_http_request(
    *,
    ip: str,
    method: str,
    path: str,
    status: int,
    latency_ms: int,
    query: str | None = None,
    req_body: str | None = None,
    upload_info: dict | None = None,
    res_body: str | None = None,
) -> None:
    """Write one HTTP request/response entry to requests.log.

    Call from middleware after the response is ready.
    upload_info is set on request.state by route handlers that process file uploads.
    """
    if LOG_FORMAT_TYPE == "json":
        obj: dict = {
            "ts": _ts_utc(),
            "event": "http",
            "method": method,
            "path": path,
            "ip": ip,
            "status": status,
            "latency_ms": latency_ms,
        }
        if query:
            obj["query"] = query
        if req_body:
            try:
                obj["req_body"] = _json.loads(req_body)
            except Exception:
                obj["req_body"] = req_body
        if upload_info:
            obj["upload"] = upload_info
        if res_body:
            try:
                obj["res_body"] = _json.loads(res_body)
            except Exception:
                obj["res_body"] = res_body
        requests_logger.info(_json.dumps(obj, ensure_ascii=False))
    else:
        lines = [
            f"[HTTP] {_ts_local()} | {method} {path} | ip={ip} | {status} | {latency_ms}ms"
        ]
        if query:
            lines.append(f"  QUERY  : {query}")
        if req_body:
            lines.append(f"  REQ    : {req_body}")
        if upload_info:
            name = upload_info.get("filename", "?")
            size = upload_info.get("size_mb", 0)
            mime = upload_info.get("mime", "?")
            dur = upload_info.get("duration_s")
            dur_str = f" | {dur}s" if dur is not None else ""
            lines.append(f"  FILE   : {name} | {size:.2f} MB | {mime}{dur_str}")
        if res_body:
            lines.append(f"  RES    : {res_body}")
        requests_logger.info("\n".join(lines))


def log_ws_event(
    *,
    event: str,        # "connect" | "disconnect"
    ip: str,
    path: str,
    config: dict | None = None,
    duration_s: int | None = None,
    frames: int | None = None,
) -> None:
    """Write one WebSocket connect or disconnect entry to requests.log."""
    if LOG_FORMAT_TYPE == "json":
        obj = {
            "ts": _ts_utc(),
            "event": f"ws_{event}",
            "path": path,
            "ip": ip,
        }
        if config:
            obj["config"] = config
        if duration_s is not None:
            obj["duration_s"] = duration_s
        if frames is not None:
            obj["frames"] = frames
        requests_logger.info(_json.dumps(obj, ensure_ascii=False))
    else:
        ts = _ts_local()
        if event == "connect":
            lines = [f"[WS-CONNECT   ] {ts} | {path} | ip={ip}"]
            if config:
                lines.append(f"  CONFIG : {_json.dumps(config, ensure_ascii=False)}")
        else:
            suffix = ""
            if duration_s is not None:
                m, s = divmod(duration_s, 60)
                suffix += f" | duration={m}m{s}s"
            if frames is not None:
                suffix += f" | frames={frames}"
            lines = [f"[WS-DISCONNECT ] {ts} | {path} | ip={ip}{suffix}"]
        requests_logger.info("\n".join(lines))

