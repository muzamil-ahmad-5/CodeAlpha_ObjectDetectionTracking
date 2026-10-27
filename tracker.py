"""
CodeAlpha Artificial Intelligence Internship
Project: CodeAlpha_ObjectDetectionTracking
Module: tracker.py

Description:
Handles persistent multi-object tracking using state-of-the-art algorithms
(ByteTrack / BoT-SORT). Maintains track histories (trajectories), persistent
color palettes, unique object counts, and high-fidelity visual rendering.
"""

import math
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import cv2

# Curated high-contrast aesthetic colors (BGR format for OpenCV)
PALETTE_BGR = [
    (255, 107, 107),  # Vibrant Coral / Rose
    (78, 205, 196),   # Mint / Teal
    (255, 230, 109),  # Bright Sun Yellow
    (170, 111, 255),  # Violet Purple
    (77, 150, 255),   # Royal Blue
    (255, 159, 67),   # Amber Orange
    (46, 213, 115),   # Emerald Green
    (255, 75, 160),   # Neon Pink
    (0, 210, 211),    # Cyan
    (162, 155, 254),  # Soft Lavender
    (253, 121, 168),  # Rose Pink
    (0, 184, 148),    # Jade Green
    (225, 112, 85),   # Terracotta
    (9, 132, 227),    # Deep Sky
    (253, 203, 110)   # Marigold
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generates a deterministic, vibrant color for a given tracking ID."""
    if track_id < 0:
        return (180, 180, 180)  # Neutral grey for untracked
    return PALETTE_BGR[track_id % len(PALETTE_BGR)]


class ObjectTracker:
    """
    Multi-Object Tracker wrapping ByteTrack / BoT-SORT algorithms.
    Maintains persistent tracking identities, motion trajectories,
    and cumulative statistics across frames.
    """

    AVAILABLE_TRACKERS = {
        "ByteTrack (Fast, Low Latency, Ideal for Dense Crowds)": "bytetrack.yaml",
        "BoT-SORT (Camera Motion Compensation, Re-ID)": "botsort.yaml"
    }

    def __init__(
        self,
        tracker_type: str = "bytetrack.yaml",
        trail_length: int = 30
    ):
        """
        Initialize the tracker manager.

        Args:
            tracker_type: Tracking configuration ('bytetrack.yaml' or 'botsort.yaml').
            trail_length: Max number of historical centroid points to draw motion trails.
        """
        self.tracker_type = tracker_type
        self.trail_length = trail_length
        # Dictionary mapping track_id -> deque of (x_center, y_center)
        self.trajectories: Dict[int, deque] = {}
        # Set of unique IDs ever encountered
        self.unique_tracked_ids: Set[int] = set()
        # Mapping of track_id to its most frequently observed class name
        self.id_to_class: Dict[int, str] = {}
        # Frame counter
        self.frame_count = 0

    def reset(self):
        """Reset all tracking states and trajectories (e.g., when a new video is loaded)."""
        self.trajectories.clear()
        self.unique_tracked_ids.clear()
        self.id_to_class.clear()
        self.frame_count = 0

    def update_with_yolo(
        self,
        yolo_model: Any,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        selected_classes: Optional[List[str]] = None,
        device: str = "cpu"
    ) -> List[Dict[str, Any]]:
        """
        Execute YOLO tracking on a video frame using Ultralytics tracking engine.

        Args:
            yolo_model: Ultralytics YOLO model instance.
            frame: Video frame (BGR format).
            conf_threshold: Confidence threshold.
            iou_threshold: IoU threshold for NMS.
            selected_classes: Optional list of class names to track.
            device: 'cuda' or 'cpu'.

        Returns:
            List of track dicts:
            [{
                "track_id": int,
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "centroid": (int, int)
            }, ...]
        """
        self.frame_count += 1
        active_tracks: List[Dict[str, Any]] = []

        if frame is None or frame.size == 0 or yolo_model is None:
            return active_tracks

        # Convert class names to IDs if filtering is applied
        target_class_indices = None
        if selected_classes is not None and len(selected_classes) > 0:
            name_to_id = {v.lower(): k for k, v in yolo_model.names.items()}
            target_class_indices = [
                name_to_id[c.lower()] for c in selected_classes if c.lower() in name_to_id
            ]
            if not target_class_indices:
                return active_tracks

        try:
            # Execute tracking step with persistent state
            results = yolo_model.track(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=target_class_indices,
                tracker=self.tracker_type,
                persist=True,
                device=device,
                verbose=False
            )

            if not results or len(results) == 0:
                return active_tracks

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return active_tracks

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Check if tracker IDs are available (might be None on initial unconfirmed frame)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            for box, conf, cls_id, t_id in zip(boxes, confs, cls_ids, track_ids):
                class_name = yolo_model.names.get(cls_id, f"class_{cls_id}")
                t_id_int = int(t_id)

                x1, y1, x2, y2 = [float(b) for b in box]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if t_id_int != -1:
                    self.unique_tracked_ids.add(t_id_int)
                    self.id_to_class[t_id_int] = class_name

                    # Record trajectory trail
                    if t_id_int not in self.trajectories:
                        self.trajectories[t_id_int] = deque(maxlen=self.trail_length)
                    self.trajectories[t_id_int].append((cx, cy))

                active_tracks.append({
                    "track_id": t_id_int,
                    "box": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": str(class_name),
                    "centroid": (cx, cy)
                })

            return active_tracks
        except Exception as e:
            # Graceful fallback on track failure
            return []

    def get_frame_class_counts(self, active_tracks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculates object count for each class in the current frame."""
        counts: Dict[str, int] = {}
        for track in active_tracks:
            c = track["class_name"]
            counts[c] = counts.get(c, 0) + 1
        return counts

    def get_cumulative_class_counts(self) -> Dict[str, int]:
        """Calculates total unique object counts by class across the entire video."""
        counts: Dict[str, int] = {}
        for t_id, c in self.id_to_class.items():
            counts[c] = counts.get(c, 0) + 1
        return counts

    def draw_annotations(
        self,
        frame: np.ndarray,
        active_tracks: List[Dict[str, Any]],
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_tracks: bool = True,
        show_trails: bool = True,
        show_counts_hud: bool = True,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Renders aesthetic computer-vision overlays onto the frame:
        - Sleek corner-accentuated or clean bounding boxes
        - Motion trajectory trails
        - Glowing badge labels with persistent colors
        - Current frame counts & statistics overlay
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Trajectory Trails first (so boxes appear on top)
        if show_trails:
            for track in active_tracks:
                t_id = track["track_id"]
                if t_id != -1 and t_id in self.trajectories:
                    pts = list(self.trajectories[t_id])
                    color = get_color_for_id(t_id)
                    num_pts = len(pts)
                    for i in range(1, num_pts):
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        # Gradually increase thickness and opacity towards current head
                        thickness = max(1, int(math.sqrt(float(i + 1) / num_pts) * 3.5))
                        cv2.line(annotated, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)

        # 2. Draw Bounding Boxes and Label Badges
        for track in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            t_id = track["track_id"]
            class_name = track["class_name"]
            conf = track["confidence"]
            color = get_color_for_id(t_id)

            if show_boxes:
                # Modern Bounding Box with subtle corner brackets
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                
                # Corner accents for sleek tech aesthetic
                corner_len = max(8, min(24, int(min(x2 - x1, y2 - y1) * 0.2)))
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 4, lineType=cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 4, lineType=cv2.LINE_AA)

            # Build label text
            label_parts = []
            if show_tracks and t_id != -1:
                label_parts.append(f"#{t_id}")
            if show_labels:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")

            if label_parts:
                label_text = " | ".join(label_parts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                font_thick = 1

                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)
                # Compute background rectangle coordinates
                badge_y1 = max(0, y1 - text_h - 10)
                badge_y2 = y1
                badge_x1 = x1
                badge_x2 = min(w - 1, x1 + text_w + 14)

                # Draw filled badge background
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
                # Determine text contrast (black for bright colors)
                brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
                text_color = (15, 15, 15) if brightness > 140 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label_text,
                    (badge_x1 + 6, badge_y2 - 6),
                    font,
                    font_scale,
                    text_color,
                    font_thick,
                    lineType=cv2.LINE_AA
                )

        # 3. Draw HUD Statistics Overlay if requested
        if show_counts_hud:
            frame_counts = self.get_frame_class_counts(active_tracks)
            total_active = len(active_tracks)
            total_unique = len(self.unique_tracked_ids)

            # Semi-transparent dark overlay panel at top-left
            hud_lines = [
                f"Active Objects: {total_active} | Unique Total: {total_unique}"
            ]
            if fps is not None and fps > 0:
                hud_lines.insert(0, f"Processing Speed: {fps:.1f} FPS")

            if frame_counts:
                counts_str = ", ".join([f"{k}: {v}" for k, v in sorted(frame_counts.items())])
                hud_lines.append(f"Classes: {counts_str}")

            # Draw sleek HUD card
            card_h = len(hud_lines) * 24 + 18
            card_w = min(w - 20, 480)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (20, 24, 33), -1)
            cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (56, 189, 248), 1)
            cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

            # Draw HUD text
            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx, line in enumerate(hud_lines):
                line_y = 30 + idx * 24
                cv2.putText(
                    annotated,
                    line,
                    (20, line_y),
                    font,
                    0.50,
                    (240, 246, 252),
                    1,
                    lineType=cv2.LINE_AA
                )

        return annotated
