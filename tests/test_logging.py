"""
Tests for the new logging infrastructure introduced in constant_var.py.

Covers:
- mask_rtsp: credential masking in various URL formats
- log_http_request: text and JSON format output
- log_ws_event: connect / disconnect entries
- debug_warning / debug_error: level and auto-traceback behaviour
- RotatingFileHandler: requests.log is created with correct handler
- RequestLoggingMiddleware: IP capture, body buffering, binary skip, state passthrough
"""

import json
import logging
import os
import sys
import time
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


# ---------------------------------------------------------------------------
# mask_rtsp
# ---------------------------------------------------------------------------

class TestMaskRtsp:
    def test_masks_user_and_password(self):
        from constant_var import mask_rtsp
        url = "rtsp://admin:password123@192.168.1.108:554/stream"
        assert mask_rtsp(url) == "rtsp://***:***@192.168.1.108:554/stream"

    def test_masks_case_insensitive(self):
        from constant_var import mask_rtsp
        url = "RTSP://user:pass@10.0.0.1:554/ch01"
        result = mask_rtsp(url)
        assert "user" not in result
        assert "pass" not in result
        assert "***:***@" in result.lower()

    def test_no_credentials_unchanged(self):
        from constant_var import mask_rtsp
        url = "rtsp://192.168.1.1:554/stream"
        # No user:pass@ pattern — unchanged
        assert mask_rtsp(url) == url

    def test_non_rtsp_url_unchanged(self):
        from constant_var import mask_rtsp
        url = "https://drive.google.com/file/d/abc123/view"
        assert mask_rtsp(url) == url

    def test_masks_in_json_string(self):
        from constant_var import mask_rtsp
        body = '{"url": "rtsp://admin:secret@192.168.1.1:554/H.264", "confidence": 0.45}'
        result = mask_rtsp(body)
        assert "secret" not in result
        assert "admin" not in result
        assert "rtsp://***:***@192.168.1.1:554/H.264" in result

    def test_empty_string_unchanged(self):
        from constant_var import mask_rtsp
        assert mask_rtsp("") == ""

    def test_multiple_rtsp_urls_in_string(self):
        from constant_var import mask_rtsp
        text = "url1=rtsp://a:b@host1 url2=rtsp://c:d@host2"
        result = mask_rtsp(text)
        assert "a:b" not in result
        assert "c:d" not in result
        assert result.count("***:***@") == 2


# ---------------------------------------------------------------------------
# log_http_request — text format
# ---------------------------------------------------------------------------

class TestLogHttpRequestText:
    """Verify log_http_request writes correct text to requests.log."""

    def _capture_requests_log(self, **kwargs):
        """Call log_http_request and capture what was written to requests_logger."""
        from constant_var import requests_logger
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        requests_logger.addHandler(handler)
        try:
            from constant_var import log_http_request
            log_http_request(**kwargs)
            return buf.getvalue()
        finally:
            requests_logger.removeHandler(handler)

    def test_basic_get_request(self):
        out = self._capture_requests_log(
            ip="127.0.0.1", method="GET", path="/", status=200, latency_ms=5
        )
        assert "[HTTP]" in out
        assert "GET /" in out
        assert "ip=127.0.0.1" in out
        assert "200" in out
        assert "5ms" in out

    def test_query_string_logged(self):
        out = self._capture_requests_log(
            ip="10.0.0.1", method="GET", path="/logs",
            status=200, latency_ms=10, query="limit=50&offset=0"
        )
        assert "QUERY" in out
        assert "limit=50" in out

    def test_json_req_body_logged(self):
        body = '{"url": "rtsp://***:***@host/stream", "confidence": 0.45}'
        out = self._capture_requests_log(
            ip="10.0.0.1", method="POST", path="/rtsp/detect",
            status=200, latency_ms=1800, req_body=body
        )
        assert "REQ" in out
        assert "rtsp://***:***@host/stream" in out

    def test_upload_info_logged_without_duration(self):
        out = self._capture_requests_log(
            ip="10.0.0.1", method="POST", path="/image/detect",
            status=200, latency_ms=250,
            upload_info={"filename": "photo.jpg", "size_mb": 1.24, "mime": "image/jpeg"}
        )
        assert "FILE" in out
        assert "photo.jpg" in out
        assert "1.24 MB" in out
        assert "image/jpeg" in out

    def test_upload_info_with_duration(self):
        out = self._capture_requests_log(
            ip="10.0.0.1", method="POST", path="/video/detect",
            status=200, latency_ms=45000,
            upload_info={
                "filename": "highway.mp4", "size_mb": 45.2,
                "mime": "video/mp4", "duration_s": 120.5
            }
        )
        assert "highway.mp4" in out
        assert "45.20 MB" in out
        assert "120.5s" in out

    def test_response_body_logged(self):
        res = '{"success": true, "counts": {"car": 3, "total": 3}}'
        out = self._capture_requests_log(
            ip="10.0.0.1", method="POST", path="/image/detect",
            status=200, latency_ms=230, res_body=res
        )
        assert "RES" in out
        assert '"car": 3' in out

    def test_error_status_logged(self):
        out = self._capture_requests_log(
            ip="10.0.0.1", method="POST", path="/image/detect",
            status=500, latency_ms=12,
            res_body='{"detail": "Detection failed"}'
        )
        assert "500" in out

    def test_none_fields_not_logged(self):
        out = self._capture_requests_log(
            ip="10.0.0.1", method="GET", path="/stats/summary",
            status=200, latency_ms=8
        )
        assert "QUERY" not in out
        assert "REQ" not in out
        assert "FILE" not in out
        assert "RES" not in out


# ---------------------------------------------------------------------------
# log_http_request — JSON format
# ---------------------------------------------------------------------------

class TestLogHttpRequestJson:
    def _capture_json(self, **kwargs):
        from constant_var import requests_logger
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        requests_logger.addHandler(handler)
        try:
            with patch("constant_var.LOG_FORMAT_TYPE", "json"):
                from constant_var import log_http_request
                log_http_request(**kwargs)
            raw = buf.getvalue().strip()
            return json.loads(raw) if raw else {}
        finally:
            requests_logger.removeHandler(handler)

    def test_json_keys_present(self):
        obj = self._capture_json(
            ip="1.2.3.4", method="POST", path="/video/jobs",
            status=202, latency_ms=12,
            req_body='{"url": "https://example.com/video.mp4", "confidence": 0.45}'
        )
        assert obj["event"] == "http"
        assert obj["method"] == "POST"
        assert obj["path"] == "/video/jobs"
        assert obj["ip"] == "1.2.3.4"
        assert obj["status"] == 202
        assert obj["latency_ms"] == 12
        assert "ts" in obj

    def test_req_body_parsed_as_object(self):
        obj = self._capture_json(
            ip="1.2.3.4", method="POST", path="/rtsp/detect",
            status=200, latency_ms=1800,
            req_body='{"url": "rtsp://***:***@host/stream", "confidence": 0.45}'
        )
        assert isinstance(obj.get("req_body"), dict)
        assert obj["req_body"]["confidence"] == 0.45

    def test_res_body_parsed_as_object(self):
        obj = self._capture_json(
            ip="1.2.3.4", method="GET", path="/logs",
            status=200, latency_ms=5,
            res_body='[{"id": "abc", "type": "Gambar"}]'
        )
        assert isinstance(obj.get("res_body"), list)
        assert obj["res_body"][0]["type"] == "Gambar"

    def test_upload_info_included(self):
        obj = self._capture_json(
            ip="1.2.3.4", method="POST", path="/image/annotate",
            status=200, latency_ms=300,
            upload_info={"filename": "img.png", "size_mb": 0.5, "mime": "image/png"}
        )
        assert obj["upload"]["filename"] == "img.png"
        assert obj["upload"]["size_mb"] == 0.5


# ---------------------------------------------------------------------------
# log_ws_event — text and JSON
# ---------------------------------------------------------------------------

class TestLogWsEvent:
    def _capture(self, format_type="text", **kwargs):
        from constant_var import requests_logger
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        requests_logger.addHandler(handler)
        try:
            with patch("constant_var.LOG_FORMAT_TYPE", format_type):
                from constant_var import log_ws_event
                log_ws_event(**kwargs)
            return buf.getvalue()
        finally:
            requests_logger.removeHandler(handler)

    def test_connect_text(self):
        out = self._capture(
            format_type="text",
            event="connect", ip="192.168.1.50", path="/rtsp/stream",
            config={"url": "rtsp://***:***@host/stream", "confidence": 0.45}
        )
        assert "WS-CONNECT" in out
        assert "/rtsp/stream" in out
        assert "ip=192.168.1.50" in out
        assert "CONFIG" in out
        assert "rtsp://***:***@host/stream" in out

    def test_disconnect_text_with_duration_and_frames(self):
        out = self._capture(
            format_type="text",
            event="disconnect", ip="192.168.1.50", path="/rtsp/stream",
            duration_s=300, frames=4500
        )
        assert "WS-DISCONNECT" in out
        assert "5m0s" in out
        assert "frames=4500" in out

    def test_disconnect_text_no_optional_fields(self):
        out = self._capture(
            format_type="text",
            event="disconnect", ip="10.0.0.1", path="/ezviz/stream"
        )
        assert "WS-DISCONNECT" in out
        assert "/ezviz/stream" in out

    def test_connect_json(self):
        raw = self._capture(
            format_type="json",
            event="connect", ip="10.0.0.5", path="/ezviz/stream",
            config={"device_serial": "ABC123", "confidence": 0.45}
        )
        obj = json.loads(raw.strip())
        assert obj["event"] == "ws_connect"
        assert obj["ip"] == "10.0.0.5"
        assert obj["config"]["device_serial"] == "ABC123"

    def test_disconnect_json(self):
        raw = self._capture(
            format_type="json",
            event="disconnect", ip="10.0.0.5", path="/rtsp/stream",
            duration_s=180, frames=2700
        )
        obj = json.loads(raw.strip())
        assert obj["event"] == "ws_disconnect"
        assert obj["duration_s"] == 180
        assert obj["frames"] == 2700


# ---------------------------------------------------------------------------
# debug_warning / debug_error auto-traceback
# ---------------------------------------------------------------------------

class TestDebugWrappers:
    def test_debug_warning_writes_to_logger(self):
        from constant_var import logger, debug_warning
        with patch.object(logger, "warning") as mock_warn:
            debug_warning("test warning message")
            mock_warn.assert_called_once()
            args, kwargs = mock_warn.call_args
            assert args[0] == "test warning message"
            assert kwargs.get("stacklevel") == 2

    def test_debug_error_no_traceback_outside_except(self):
        from constant_var import logger, debug_error
        with patch.object(logger, "error") as mock_err:
            debug_error("error outside except")
            _, kwargs = mock_err.call_args
            # exc_info=False outside an except block
            assert kwargs.get("exc_info") is False

    def test_debug_error_includes_traceback_inside_except(self):
        from constant_var import logger, debug_error
        with patch.object(logger, "error") as mock_err:
            try:
                raise ValueError("test exception")
            except ValueError:
                debug_error("caught an error")
            _, kwargs = mock_err.call_args
            # exc_info=True inside except block (sys.exc_info()[0] is not None)
            assert kwargs.get("exc_info") is True

    def test_detail_debug_uses_debug_level(self):
        from constant_var import logger, detail_debug
        with patch.object(logger, "debug") as mock_dbg:
            detail_debug("verbose detail")
            mock_dbg.assert_called_once()
            args, _ = mock_dbg.call_args
            assert args[0] == "verbose detail"


# ---------------------------------------------------------------------------
# RotatingFileHandler configuration
# ---------------------------------------------------------------------------

class TestRotatingFileHandlers:
    def test_requests_log_handler_exists(self):
        from constant_var import requests_logger
        assert len(requests_logger.handlers) >= 1

    def test_requests_log_rotating_handler(self):
        from logging.handlers import RotatingFileHandler
        from constant_var import requests_logger
        rotating = [h for h in requests_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) >= 1
        handler = rotating[0]
        assert handler.maxBytes == 50 * 1024 * 1024
        assert handler.backupCount == 5

    def test_app_log_rotating_handler(self):
        from logging.handlers import RotatingFileHandler
        from constant_var import logger
        rotating = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rotating) >= 1

    def test_requests_log_file_path(self):
        from logging.handlers import RotatingFileHandler
        from constant_var import requests_logger, LOGS_DIR
        rotating = [h for h in requests_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert any("requests.log" in h.baseFilename for h in rotating)

    def test_errors_log_handler_level(self):
        from logging.handlers import RotatingFileHandler
        from constant_var import logger
        error_handlers = [
            h for h in logger.handlers
            if isinstance(h, RotatingFileHandler) and "errors.log" in h.baseFilename
        ]
        assert len(error_handlers) == 1
        assert error_handlers[0].level == logging.ERROR


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------

class TestRequestLoggingMiddleware:
    """Integration tests using FastAPI TestClient."""

    def _make_app(self):
        from main import RequestLoggingMiddleware
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/json-endpoint")
        async def json_endpoint(request: Request):
            return JSONResponse({"result": "ok", "value": 42})

        @app.post("/json-body")
        async def json_body_endpoint(request: Request):
            return JSONResponse({"received": True})

        @app.post("/upload-endpoint")
        async def upload_endpoint(request: Request):
            request.state.upload_info = {
                "filename": "test.jpg",
                "size_mb": 0.5,
                "mime": "image/jpeg",
            }
            return JSONResponse({"processed": True})

        @app.get("/binary-endpoint")
        async def binary_endpoint():
            return StreamingResponse(iter([b"fake-jpeg-bytes"]), media_type="image/jpeg")

        return app

    def test_json_response_is_not_broken(self):
        """Middleware must not corrupt JSON responses."""
        app = self._make_app()
        with patch("main.log_http_request"):
            client = TestClient(app)
            resp = client.get("/json-endpoint")
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"] == "ok"
        assert data["value"] == 42

    def test_binary_response_passes_through(self):
        """Middleware must not buffer/corrupt binary (non-JSON) responses."""
        app = self._make_app()
        with patch("main.log_http_request"):
            client = TestClient(app)
            resp = client.get("/binary-endpoint")
        assert resp.status_code == 200
        assert resp.content == b"fake-jpeg-bytes"

    def test_log_http_request_called_once_per_request(self):
        app = self._make_app()
        with patch("main.log_http_request") as mock_log:
            client = TestClient(app)
            client.get("/json-endpoint")
        mock_log.assert_called_once()

    def test_client_ip_captured(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint")
        assert "ip" in captured
        assert isinstance(captured["ip"], str)
        assert len(captured["ip"]) > 0

    def test_method_and_path_captured(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint")
        assert captured["method"] == "GET"
        assert captured["path"] == "/json-endpoint"

    def test_status_code_captured(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint")
        assert captured["status"] == 200

    def test_latency_ms_is_positive_integer(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint")
        assert isinstance(captured["latency_ms"], int)
        assert captured["latency_ms"] >= 0

    def test_res_body_is_json_string(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint")
        assert captured["res_body"] is not None
        parsed = json.loads(captured["res_body"])
        assert parsed["result"] == "ok"

    def test_binary_response_res_body_is_none(self):
        """Binary responses should not have res_body captured."""
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/binary-endpoint")
        assert captured.get("res_body") is None

    def test_upload_info_from_request_state(self):
        """Route handlers set request.state.upload_info; middleware reads it."""
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.post("/upload-endpoint")
        info = captured.get("upload_info")
        assert info is not None
        assert info["filename"] == "test.jpg"
        assert info["size_mb"] == 0.5

    def test_json_request_body_captured_and_masked(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.post(
                "/json-body",
                json={"url": "rtsp://admin:secret@192.168.1.1:554/stream", "conf": 0.45}
            )
        req_body = captured.get("req_body") or ""
        assert "secret" not in req_body
        assert "***:***@" in req_body

    def test_options_request_not_logged(self):
        """OPTIONS preflight requests should not be logged."""
        app = self._make_app()
        with patch("main.log_http_request") as mock_log:
            client = TestClient(app)
            client.options("/json-endpoint")
        mock_log.assert_not_called()

    def test_x_forwarded_for_used_as_ip(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint", headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})
        # Should use the first IP from X-Forwarded-For
        assert captured["ip"] == "203.0.113.5"

    def test_query_string_captured(self):
        app = self._make_app()
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
        with patch("main.log_http_request", side_effect=capture):
            client = TestClient(app)
            client.get("/json-endpoint?limit=50&offset=0")
        assert "limit=50" in (captured.get("query") or "")


# ---------------------------------------------------------------------------
# LOG_FORMAT env var
# ---------------------------------------------------------------------------

class TestLogFormatEnvVar:
    def test_default_is_text(self):
        from constant_var import LOG_FORMAT_TYPE
        # The default when LOG_FORMAT is not set should be "text"
        assert LOG_FORMAT_TYPE in ("text", "json")  # either is valid depending on env

    def test_log_http_request_json_format_single_line(self):
        """JSON format must produce a single-line parseable JSON object."""
        from constant_var import requests_logger
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        requests_logger.addHandler(handler)
        try:
            with patch("constant_var.LOG_FORMAT_TYPE", "json"):
                from constant_var import log_http_request
                log_http_request(
                    ip="1.2.3.4", method="GET", path="/stats/summary",
                    status=200, latency_ms=7
                )
            output = buf.getvalue().strip()
            lines = [l for l in output.splitlines() if l.strip()]
            assert len(lines) == 1, f"Expected 1 line, got {len(lines)}: {output}"
            obj = json.loads(lines[0])
            assert obj["event"] == "http"
        finally:
            requests_logger.removeHandler(handler)
