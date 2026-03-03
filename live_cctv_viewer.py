"""
Live CCTV Viewer — RTSP Stream Viewer.

Simple script to view a CCTV camera live via RTSP link using OpenCV.

Controls:
    - Press 'q' or ESC to quit
    - Press 'f' to toggle fullscreen
    - Press 's' to save a screenshot

Author: Naufal Firdaus
"""

import sys
import time
from typing import Optional
from pathlib import Path

import cv2


# =============================================
# CONFIGURATION
# =============================================

RTSP_URL: str = "rtsp://admin:1Triliun@192.168.1.140:554/H.264"
WINDOW_NAME: str = "Live CCTV Viewer"
SCREENSHOT_DIR: Path = Path(__file__).resolve().parent / "screenshots"

# Reconnect settings
MAX_RECONNECT_ATTEMPTS: int = 5
RECONNECT_DELAY_SECONDS: int = 3


# =============================================
# HELPER FUNCTIONS
# =============================================

def open_rtsp_stream(url: str) -> Optional[cv2.VideoCapture]:
    """Open RTSP stream with TCP transport for stability.

    Args:
        url: RTSP stream URL.

    Returns:
        VideoCapture object if successful, None if failed.
    """
    print(f"[INFO] Connecting to: {url}")

    # Use TCP transport for more stable RTSP connection
    cap: cv2.VideoCapture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency

    if not cap.isOpened():
        print(f"[ERROR] Cannot open RTSP stream: {url}")
        return None

    # Get stream info
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"[INFO] Stream opened: {width}x{height} @ {fps:.1f} FPS")
    return cap


def save_screenshot(frame, screenshot_dir: Path) -> None:
    """Save current frame as screenshot.

    Args:
        frame: Current video frame (numpy array).
        screenshot_dir: Directory to save screenshots in.
    """
    screenshot_dir.mkdir(exist_ok=True)
    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    filepath: Path = screenshot_dir / f"screenshot_{timestamp}.jpg"
    cv2.imwrite(str(filepath), frame)
    print(f"[INFO] Screenshot saved: {filepath}")


# =============================================
# MAIN APPLICATION
# =============================================

def main() -> None:
    """Main CCTV live viewer loop."""
    print("=" * 50)
    print("LIVE CCTV VIEWER")
    print("=" * 50)
    print(f"RTSP URL  : {RTSP_URL}")
    print(f"Controls  : [q/ESC] Quit | [f] Fullscreen | [s] Screenshot")
    print("=" * 50)

    cap: Optional[cv2.VideoCapture] = open_rtsp_stream(RTSP_URL)
    if cap is None:
        print("[FATAL] Failed to open stream. Exiting.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    is_fullscreen: bool = False
    reconnect_attempts: int = 0
    frame_count: int = 0
    fps_start_time: float = time.time()
    display_fps: float = 0.0

    print("[INFO] Streaming started. Press 'q' or ESC to quit.")

    while True:
        ret: bool
        frame = None
        ret, frame = cap.read()

        if not ret or frame is None:
            reconnect_attempts += 1
            print(f"[WARN] Frame read failed. Reconnecting ({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})...")

            if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                print("[FATAL] Max reconnect attempts reached. Exiting.")
                break

            cap.release()
            time.sleep(RECONNECT_DELAY_SECONDS)
            cap = open_rtsp_stream(RTSP_URL)

            if cap is None:
                continue
            reconnect_attempts = 0
            continue

        # Reset reconnect counter on successful frame
        reconnect_attempts = 0
        frame_count += 1

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed: float = time.time() - fps_start_time
            display_fps = 30.0 / elapsed if elapsed > 0 else 0.0
            fps_start_time = time.time()

        # Draw FPS overlay
        cv2.putText(
            frame, f"FPS: {display_fps:.1f}", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle keyboard input (1ms delay for responsive controls)
        key: int = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # 'q' or ESC
            print("[INFO] Quit requested by user.")
            break
        elif key == ord("f"):  # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print(f"[INFO] Fullscreen: {'ON' if is_fullscreen else 'OFF'}")
        elif key == ord("s"):  # Save screenshot
            save_screenshot(frame, SCREENSHOT_DIR)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total frames viewed: {frame_count}")
    print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()
