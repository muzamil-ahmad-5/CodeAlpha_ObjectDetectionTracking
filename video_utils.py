"""
CodeAlpha Artificial Intelligence Internship
Project: CodeAlpha_ObjectDetectionTracking
Module: video_utils.py

Description:
Helper utilities for video verification, synthetic sample video generation,
and browser-compatible H.264 video encoding.
"""

import os
import math
import tempfile
import logging
from typing import Tuple, Optional, Generator
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_available_cameras(max_to_check: int = 3) -> list:
    """Detects available camera indices on the host system."""
    available = []
    for idx in range(max_to_check):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        except Exception:
            pass
    return available if available else [0]


def open_camera_capture(camera_source, width: int = 640, height: int = 480):
    """
    Opens a camera capture stream with optimal backend settings.
    Accepts integer (device index) or string (RTSP / HTTP stream URL).
    """
    try:
        if isinstance(camera_source, int) and os.name == 'nt':
            cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened() and isinstance(camera_source, int):
            cap = cv2.VideoCapture(camera_source)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
    except Exception as e:
        logger.error(f"Error opening camera source {camera_source}: {e}")
    return None


def validate_video_file(file_path: str) -> Tuple[bool, str, int, int, float, int]:
    """
    Validates whether a given video file can be opened and parsed by OpenCV.

    Returns:
        Tuple: (is_valid, error_message, width, height, fps, total_frames)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", 0, 0, 0.0, 0

    if os.path.getsize(file_path) == 0:
        return False, "File is empty (0 bytes).", 0, 0, 0.0, 0

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, "Failed to open video file. Format may be unsupported or corrupted.", 0, 0, 0.0, 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read test frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or frame.size == 0:
        return False, "Video contains zero readable frames.", 0, 0, 0.0, 0

    return True, "Valid video file.", width, height, fps, total_frames


def generate_sample_demo_video(output_path: str, duration_sec: int = 5, fps: int = 25) -> str:
    """
    Generates a high-quality synthetic demo video with moving multi-class objects
    (simulated cars, people, and sports balls) for instant out-of-the-box testing.
    """
    width, height = 640, 360
    total_frames = duration_sec * fps
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        # Modern asphalt road background with lane dividers
        frame = np.full((height, width, 3), (35, 38, 46), dtype=np.uint8)

        # Draw road lanes
        cv2.rectangle(frame, (0, 100), (width, 280), (45, 50, 60), -1)
        dash_offset = int((frame_idx * 6) % 50)
        for x in range(-50 + dash_offset, width + 50, 50):
            cv2.line(frame, (x, 190), (x + 25, 190), (220, 220, 220), 3)

        # Sidewalk
        cv2.rectangle(frame, (0, 280), (width, height), (70, 75, 85), -1)
        cv2.rectangle(frame, (0, 0), (width, 100), (30, 32, 40), -1)

        # Simulated Object 1: Moving Car 1 (Left to Right)
        car1_x = int((frame_idx * 5) % (width + 120)) - 60
        car1_y = 130
        cv2.rectangle(frame, (car1_x, car1_y), (car1_x + 90, car1_y + 45), (180, 50, 40), -1)
        cv2.rectangle(frame, (car1_x + 15, car1_y + 8), (car1_x + 75, car1_y + 35), (230, 230, 230), -1)
        # Wheels
        cv2.circle(frame, (car1_x + 20, car1_y + 45), 8, (15, 15, 15), -1)
        cv2.circle(frame, (car1_x + 70, car1_y + 45), 8, (15, 15, 15), -1)

        # Simulated Object 2: Moving Car 2 (Right to Left)
        car2_x = width - int((frame_idx * 4) % (width + 120)) + 60
        car2_y = 210
        cv2.rectangle(frame, (car2_x, car2_y), (car2_x + 85, car2_y + 45), (40, 120, 200), -1)
        cv2.rectangle(frame, (car2_x + 15, car2_y + 8), (car2_x + 70, car2_y + 35), (230, 230, 230), -1)
        cv2.circle(frame, (car2_x + 18, car2_y + 45), 8, (15, 15, 15), -1)
        cv2.circle(frame, (car2_x + 65, car2_y + 45), 8, (15, 15, 15), -1)

        # Simulated Object 3: Pedestrian walking on sidewalk
        ped_x = 40 + int((frame_idx * 2) % (width - 80))
        ped_y = 300
        # Head
        cv2.circle(frame, (ped_x, ped_y), 7, (210, 180, 140), -1)
        # Body
        cv2.rectangle(frame, (ped_x - 6, ped_y + 7), (ped_x + 6, ped_y + 30), (50, 150, 80), -1)
        # Legs
        leg_offset = int(math.sin(t * 12) * 5)
        cv2.line(frame, (ped_x - 3, ped_y + 30), (ped_x - 3 + leg_offset, ped_y + 45), (30, 30, 30), 2)
        cv2.line(frame, (ped_x + 3, ped_y + 30), (ped_x + 3 - leg_offset, ped_y + 45), (30, 30, 30), 2)

        # Stamp watermark
        cv2.putText(
            frame,
            "CodeAlpha AI Demo Feed (Synthetic Benchmark)",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (56, 189, 248),
            1,
            lineType=cv2.LINE_AA
        )

        writer.write(frame)

    writer.release()
    logger.info(f"Sample demo video generated: {output_path}")
    return output_path


def encode_frames_to_mp4(
    frame_list: list,
    output_path: str,
    fps: float = 25.0
) -> bool:
    """
    Encodes a list of RGB/BGR frames into a standard web-playable H.264 / MP4 video
    using imageio with ffmpeg backend, with fallback to OpenCV VideoWriter.
    """
    if not frame_list:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        import imageio
        # imageio expects RGB
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            quality=8,
            macro_block_size=1
        )
        for f in frame_list:
            # Check if frame is BGR and convert to RGB
            if len(f.shape) == 3 and f.shape[2] == 3:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            else:
                rgb = f
            writer.append_data(rgb)
        writer.close()
        return True
    except Exception as e:
        logger.warning(f"imageio H.264 encode failed ({e}), falling back to cv2.VideoWriter...")

    # Fallback to cv2.VideoWriter
    try:
        h, w = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in frame_list:
            out.write(f)
        out.release()
        return True
    except Exception as e:
        logger.error(f"Failed to encode video via OpenCV: {e}")
        return False
