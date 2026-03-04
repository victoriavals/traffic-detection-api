"""
EZVIZ Cloud API Service.

Handles authentication, capture, and stream URL retrieval from EZVIZ Open Platform.
Uses appKey/appSecret credentials from .env file.

Primary approach: Capture-based (uses /api/lapp/device/capture for reliable image retrieval).
Fallback: HLS stream URL (may fail with error 9053 on some camera models like H8c).
"""

import time
import httpx
import numpy as np

from constant_var import (
    EZVIZ_APP_KEY,
    EZVIZ_APP_SECRET,
    EZVIZ_TOKEN_URL,
    debug_info,
    debug_error,
)


class EzvizService:
    """Singleton service for EZVIZ Cloud API interactions.

    Manages access token lifecycle and provides methods to
    retrieve live stream URLs and capture images from EZVIZ cloud.
    Uses a persistent httpx client for connection pooling (faster requests).
    """

    _instance: "EzvizService | None" = None
    _access_token: str = ""
    _token_expire_time: float = 0.0
    _area_domain: str = ""
    _client: httpx.AsyncClient | None = None

    def __new__(cls) -> "EzvizService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent httpx client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                verify=False,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if EZVIZ credentials are configured in .env."""
        return bool(EZVIZ_APP_KEY and EZVIZ_APP_SECRET)

    async def get_access_token(self) -> str:
        """Get a valid access token, refreshing if expired.

        Returns:
            Valid access token string.

        Raises:
            RuntimeError: If credentials are missing or API call fails.
        """
        if not self.is_configured():
            raise RuntimeError(
                "EZVIZ credentials not configured. "
                "Set EZVIZ_APP_KEY and EZVIZ_APP_SECRET in .env file."
            )

        # Return cached token if still valid (with 5 min buffer)
        if self._access_token and time.time() * 1000 < self._token_expire_time - 300_000:
            return self._access_token

        debug_info("[EZVIZ] Requesting new access token...")

        client = await self._get_client()
        resp = await client.post(
            EZVIZ_TOKEN_URL,
            data={
                "appKey": EZVIZ_APP_KEY,
                "appSecret": EZVIZ_APP_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != "200":
            msg = result.get("msg", "Unknown error")
            debug_error(f"[EZVIZ] Token request failed: {msg}")
            raise RuntimeError(f"EZVIZ token request failed: {msg}")

        data = result["data"]
        self._access_token = data["accessToken"]
        self._token_expire_time = float(data["expireTime"])
        self._area_domain = data.get("areaDomain", "")

        debug_info(f"[EZVIZ] Token acquired, area domain: {self._area_domain}")
        return self._access_token

    def _get_api_base(self) -> str:
        """Get the correct API base URL (from areaDomain or fallback)."""
        return self._area_domain or "https://isgpopen.ezvizlife.com"

    async def get_live_stream_url(
        self, device_serial: str, channel_no: int = 1, protocol: int = 2
    ) -> str:
        """Get live stream URL for a device from EZVIZ cloud.

        Args:
            device_serial: Device serial number (e.g., 'AB1234567').
            channel_no: Camera channel number (default 1).
            protocol: Stream protocol (1=ezopen, 2=HLS, 3=RTMP, 4=FLV).

        Returns:
            Live stream URL string.

        Raises:
            RuntimeError: If API call fails.
        """
        token = await self.get_access_token()
        api_base = self._get_api_base()
        url = f"{api_base}/api/lapp/v2/live/address/get"

        debug_info(f"[EZVIZ] Getting live stream URL for device: {device_serial}")

        client = await self._get_client()
        resp = await client.post(
            url,
            data={
                "accessToken": token,
                "deviceSerial": device_serial,
                "channelNo": channel_no,
                "protocol": protocol,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != "200":
            msg = result.get("msg", "Unknown error")
            debug_error(f"[EZVIZ] Get live URL failed: {msg}")
            raise RuntimeError(f"EZVIZ get live URL failed: {msg}")

        stream_url = result["data"]["url"]
        debug_info(f"[EZVIZ] Stream URL acquired: {stream_url[:80]}...")
        return stream_url

    async def get_device_list(self) -> list[dict]:
        """Get list of all devices on the account.

        Returns:
            List of device info dictionaries.

        Raises:
            RuntimeError: If API call fails.
        """
        token = await self.get_access_token()
        api_base = self._get_api_base()
        url = f"{api_base}/api/lapp/device/list"

        debug_info("[EZVIZ] Fetching device list...")

        client = await self._get_client()
        resp = await client.post(
            url,
            data={
                "accessToken": token,
                "pageStart": 0,
                "pageSize": 50,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != "200":
            msg = result.get("msg", "Unknown error")
            debug_error(f"[EZVIZ] Device list failed: {msg}")
            raise RuntimeError(f"EZVIZ device list failed: {msg}")

        devices = result.get("data", [])
        debug_info(f"[EZVIZ] Found {len(devices)} device(s)")
        return devices

    async def capture_image(self, device_serial: str, channel_no: int = 1) -> np.ndarray:
        """Capture a snapshot from the camera and return as OpenCV frame.

        Uses the EZVIZ capture API which is more reliable than HLS streaming
        for cameras like H8c that may return error 9053 on live streams.

        Args:
            device_serial: Device serial number.
            channel_no: Camera channel number (default 1).

        Returns:
            OpenCV BGR image (numpy array).

        Raises:
            RuntimeError: If capture or download fails.
        """
        import asyncio
        import cv2

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                token = await self.get_access_token()
                api_base = self._get_api_base()
                client = await self._get_client()

                # 1. Request camera to capture a snapshot
                resp = await client.post(
                    f"{api_base}/api/lapp/device/capture",
                    data={
                        "accessToken": token,
                        "deviceSerial": device_serial,
                        "channelNo": channel_no,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                result = resp.json()

                if result.get("code") != "200":
                    msg = result.get("msg", "Unknown error")
                    # "Device response timeout" = camera busy, retry after delay
                    if "timeout" in msg.lower() or "busy" in msg.lower():
                        last_error = RuntimeError(msg)
                        debug_info(f"[EZVIZ] Camera busy (attempt {attempt+1}), waiting...")
                        await asyncio.sleep(1.5)
                        continue
                    raise RuntimeError(f"EZVIZ capture failed: {msg}")

                pic_url: str = result["data"]["picUrl"]

                # 2. Download the image
                img_resp = await client.get(pic_url)
                img_resp.raise_for_status()

                # 3. Decode to OpenCV frame
                img_array = np.frombuffer(img_resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Failed to decode captured image from EZVIZ")

                return frame

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                debug_info(f"[EZVIZ] Capture attempt {attempt+1} timeout, retrying...")
                # Reset client on connection errors
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None
                continue

        raise RuntimeError(f"EZVIZ capture failed after 3 attempts: {last_error}")

    async def get_device_info(self, device_serial: str) -> dict:
        """Get info for a specific device.

        Args:
            device_serial: Device serial number.

        Returns:
            Device info dictionary.

        Raises:
            RuntimeError: If API call fails.
        """
        token = await self.get_access_token()
        api_base = self._get_api_base()
        url = f"{api_base}/api/lapp/device/info"

        client = await self._get_client()
        resp = await client.post(
            url,
            data={
                "accessToken": token,
                "deviceSerial": device_serial,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != "200":
            msg = result.get("msg", "Unknown error")
            raise RuntimeError(f"EZVIZ device info failed: {msg}")

        return result.get("data", {})
