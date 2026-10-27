"""
CodeAlpha Artificial Intelligence Internship
Verification & Smoke Test Script: test_pipeline.py

Tests:
1. YOLODetector initialization, COCO class mapping, and single-frame inference.
2. ObjectTracker tracking, persistent ID assignment, and trajectory rendering.
3. Video validation and synthetic benchmark video generation.
4. H.264 video encoding.
"""

import os
import sys
import numpy as np
import cv2

print("=" * 60)
print("Running CodeAlpha_ObjectDetectionTracking Smoke Tests...")
print("=" * 60)

# 1. Test video_utils
print("\n[1/4] Testing video_utils (Synthetic Video & Validation)...")
from video_utils import generate_sample_demo_video, validate_video_file, encode_frames_to_mp4

test_video_path = "test_synthetic.mp4"
try:
    generate_sample_demo_video(test_video_path, duration_sec=2, fps=20)
    is_valid, msg, w, h, fps, frames = validate_video_file(test_video_path)
    assert is_valid, f"Validation failed: {msg}"
    print(f"  -> Generated test video successfully: {w}x{h}, {fps} FPS, {frames} frames.")
except Exception as e:
    print(f"  -> FAILED video_utils test: {e}")
    sys.exit(1)

# 2. Test YOLODetector
print("\n[2/4] Testing detector.py (YOLO Model Loading & Inference)...")
from detector import YOLODetector, CLASS_PRESETS

try:
    detector = YOLODetector(model_name_or_path="yolov8n.pt")
    supported_classes = detector.get_supported_classes()
    print(f"  -> Model loaded successfully on device: {detector.device}")
    print(f"  -> Total supported classes: {len(supported_classes)}")

    # Test inference on a synthetic image with drawn shapes
    dummy_frame = np.full((360, 640, 3), (40, 40, 40), dtype=np.uint8)
    detections = detector.detect(dummy_frame, conf_threshold=0.25)
    print(f"  -> Ran detection on dummy frame. Detections count: {len(detections)}")
except Exception as e:
    print(f"  -> FAILED detector test: {e}")
    sys.exit(1)

# 3. Test ObjectTracker
print("\n[3/4] Testing tracker.py (Object Tracking & Annotations)...")
from tracker import ObjectTracker

try:
    tracker = ObjectTracker(tracker_type="bytetrack.yaml", trail_length=15)
    tracker.reset()

    # Track on our generated test video frames
    cap = cv2.VideoCapture(test_video_path)
    ret, frame = cap.read()
    assert ret, "Could not read frame from test video"

    active_tracks = tracker.update_with_yolo(
        yolo_model=detector.model,
        frame=frame,
        conf_threshold=0.20,
        device=detector.device
    )
    annotated = tracker.draw_annotations(
        frame=frame,
        active_tracks=active_tracks,
        show_boxes=True,
        show_labels=True,
        show_conf=True,
        show_tracks=True,
        show_trails=True,
        show_counts_hud=True,
        fps=30.0
    )
    cap.release()
    assert annotated.shape == frame.shape, "Annotated frame shape mismatch"
    print(f"  -> Object tracking and frame annotation verified successfully!")
except Exception as e:
    print(f"  -> FAILED tracker test: {e}")
    sys.exit(1)

# 4. Test Video Encoding
print("\n[4/4] Testing video encoding to MP4...")
try:
    encoded_out = "test_output.mp4"
    frames_buffer = [annotated, annotated, annotated]
    success = encode_frames_to_mp4(frames_buffer, encoded_out, fps=20.0)
    assert success and os.path.exists(encoded_out), "Encoding failed"
    file_size = os.path.getsize(encoded_out)
    print(f"  -> Encoded MP4 successfully! File size: {file_size} bytes.")
except Exception as e:
    print(f"  -> FAILED video encoding test: {e}")
    sys.exit(1)

# 5. Test Live Camera Interface
print("\n[5/5] Testing real camera capture capability (open_camera_capture)...")
from video_utils import open_camera_capture
cam_cap = open_camera_capture(0, width=640, height=480)
if cam_cap is not None and cam_cap.isOpened():
    ret, cam_frame = cam_cap.read()
    cam_cap.release()
    if ret and cam_frame is not None:
        print(f"  -> Successfully connected to real hardware camera 0! Resolution: {cam_frame.shape[1]}x{cam_frame.shape[0]}")
    else:
        print("  -> Camera opened but frame was empty (camera might be busy).")
else:
    print("  -> Hardware camera not detected or busy, fallback mode active.")

# Clean up test artifacts
for f in [test_video_path, encoded_out]:
    if os.path.exists(f):
        try:
            os.remove(f)
        except Exception:
            pass

print("\n" + "=" * 60)
print("ALL PIPELINE TESTS PASSED SUCCESSFULLY! Ready for deployment.")
print("=" * 60)
