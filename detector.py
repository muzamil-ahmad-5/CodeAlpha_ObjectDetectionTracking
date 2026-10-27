"""
CodeAlpha Artificial Intelligence Internship
Project: CodeAlpha_ObjectDetectionTracking
Module: detector.py

Description:
Handles YOLO model loading, inference device selection (CUDA/CPU),
COCO class mappings, and object detection with confidence and IoU thresholds.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# COCO 80 Class Names mapping for standard YOLO models
COCO_CLASSES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Predefined category presets for intuitive UI filtering
CLASS_PRESETS: Dict[str, List[str]] = {
    "All Classes": list(COCO_CLASSES.values()),
    "Vehicles & Traffic": ["car", "motorcycle", "bus", "truck", "bicycle", "traffic light", "stop sign"],
    "Pedestrians & People": ["person", "backpack", "umbrella", "handbag"],
    "Animals & Pets": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Electronics & Office": ["laptop", "mouse", "keyboard", "cell phone", "tv", "book", "clock"],
    "Sports & Outdoor": ["sports ball", "skateboard", "surfboard", "tennis racket", "skis", "snowboard"]
}


class YOLODetector:
    """
    Modular wrapper for Ultralytics YOLO Object Detection models.
    Supports dynamic class filtering, hardware acceleration detection,
    and structured output extraction.
    """

    SUPPORTED_MODELS = {
        "YOLOv8 Nano (Fastest, Low Memory)": "yolov8n.pt",
        "YOLOv8 Small (Balanced Accuracy)": "yolov8s.pt",
        "YOLOv8 Medium (High Accuracy)": "yolov8m.pt"
    }

    def __init__(self, model_name_or_path: str = "yolov8n.pt", device: Optional[str] = None):
        """
        Initialize the YOLO detector.

        Args:
            model_name_or_path: Name of pretrained weights or path to custom .pt file.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or self._detect_best_device()
        self.model = None
        self._load_model()

    def _detect_best_device(self) -> str:
        """Determines whether CUDA GPU acceleration is available."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA acceleration detected: {device_name}")
                return "cuda"
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {e}")
        logger.info("Using CPU for inference.")
        return "cpu"

    def _load_model(self):
        """Loads or downloads the specified YOLO model weights with error handling."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {self.model_name_or_path} on {self.device}")
            self.model = YOLO(self.model_name_or_path)
            # Ensure model classes are mapped
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = COCO_CLASSES
            logger.info(f"YOLO model loaded successfully. Classes count: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name_or_path}': {e}")
            raise RuntimeError(
                f"Error initializing YOLO model '{self.model_name_or_path}'. "
                f"Ensure internet connectivity for initial download or verify local weight path. Details: {e}"
            )

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        selected_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection on a single image frame (BGR format from OpenCV).

        Args:
            frame: Numpy ndarray representing the image/frame (BGR).
            conf_threshold: Confidence score cutoff (0.0 to 1.0).
            iou_threshold: Non-Maximum Suppression (NMS) IoU cutoff.
            selected_classes: Optional list of class names to filter by.

        Returns:
            List of detected objects dictionaries:
            [{
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str
            }, ...]
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        if frame is None or frame.size == 0:
            return []

        # Map class names to class IDs for YOLO filtering if specified
        target_class_indices = None
        if selected_classes is not None and len(selected_classes) > 0:
            name_to_id = {v.lower(): k for k, v in self.class_names.items()}
            target_class_indices = [
                name_to_id[c.lower()] for c in selected_classes if c.lower() in name_to_id
            ]
            if not target_class_indices:
                # User selected classes that don't match model vocabulary
                return []

        try:
            # Perform inference
            results = self.model(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=target_class_indices,
                device=self.device,
                verbose=False
            )

            detections: List[Dict[str, Any]] = []
            if not results or len(results) == 0:
                return detections

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()  # confidence scores
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "box": [float(b) for b in box],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": str(class_name)
                })

            return detections
        except Exception as e:
            logger.error(f"Inference error during detection: {e}")
            return []

    def get_supported_classes(self) -> List[str]:
        """Returns the list of all classes supported by the loaded model."""
        if hasattr(self, "class_names") and self.class_names:
            return list(self.class_names.values())
        return list(COCO_CLASSES.values())
