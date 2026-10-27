# CodeAlpha Artificial Intelligence Internship
## Project: CodeAlpha_ObjectDetectionTracking

![Python](https://img.shields.io/badge/Python-3.10%20--%203.14-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics%20v8-00FFFF.svg)
![Tracking](https://img.shields.io/badge/Tracking-ByteTrack%20%7C%20BoT--SORT-purple.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg)

A production-grade Computer Vision application that detects multiple object classes in video streams and tracks them persistently across consecutive frames using **Ultralytics YOLO** and **ByteTrack / BoT-SORT** algorithms, wrapped in a responsive, modern **Streamlit** user interface.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Object Detection vs. Object Tracking](#-object-detection-vs-object-tracking)
3. [Architecture & Features](#-key-features)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#-installation--setup)
6. [Model Weights & Download Guide](#-model-weights--download-guide)
7. [Hardware & Memory Requirements](#-hardware--memory-requirements)
8. [Usage & Live Demo](#-usage--live-demo)
9. [Honest Limitations](#-honest-limitations)
10. [Author & Acknowledgments](#-author--acknowledgments)

---

## 🎯 Project Overview
This repository contains the solution for the **CodeAlpha Artificial Intelligence Internship** project **CodeAlpha_ObjectDetectionTracking**. 

The core objective is to analyze uploaded videos or synthetic benchmark video streams, perform real-time deep learning object detection, assign persistent tracking identities to detected objects, draw motion trajectory trails, report frame-by-frame and cumulative object counts, and allow users to export the processed video in web-compatible H.264 MP4 format.

---

## 🧠 Object Detection vs. Object Tracking

Understanding the mathematical and functional distinction between these two computer vision tasks is fundamental:

| Dimension | Object Detection | Object Tracking |
| :--- | :--- | :--- |
| **Question Answered** | *"What objects are present in this single frame and where are they located?"* | *"Where did this specific object move over time across consecutive frames?"* |
| **Input Domain** | Single isolated static image or independent frame $I_t$. | Ordered temporal sequence of frames $\{I_1, I_2, \dots, I_T\}$. |
| **Temporal Context** | **Memoryless (Markovian)**. Frame $t$ is evaluated with zero knowledge of frame $t-1$. | **High temporal awareness**. Maintains motion vectors, velocity, and visual trajectories. |
| **Output** | Spatial bounding box $[x_1, y_1, x_2, y_2]$, class label, and confidence score. | **Persistent ID** (`#1`, `#2`), trajectory history, smoothed bounding box. |
| **Computational Cost** | High (full deep convolutional/transformer forward pass). | Low to moderate (Kalman Filter state estimation + IoU association). |
| **Occlusion Handling** | Fails when an object is partially or completely occluded. | Recovers track identity using motion estimation and Re-ID feature matching. |
| **Algorithms** | YOLOv8, YOLOv11, Faster R-CNN, RT-DETR. | ByteTrack, BoT-SORT, DeepSORT, SORT. |

### The Tracking-by-Detection Paradigm
Our application employs the modern **Tracking-by-Detection** paradigm:
1. **Detection:** Pretrained YOLO extracts candidate bounding boxes for frame $t$.
2. **Motion Estimation:** A Kalman Filter predicts the anticipated position of all previously active tracks.
3. **Data Association:** **ByteTrack** associates both high-confidence and low-confidence candidate detections with existing tracks using Hungarian algorithm matching over IoU cost matrices.
4. **Identity Persistence:** When an object momentarily passes behind a street lamp, tree, or other vehicle, its velocity state maintains the track, preventing ID fragmentation.

---

## 🚀 Key Features

- **Multi-Format Video Input:** Supports `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`.
- **📹 Live Webcam & Real Camera Support:** Direct real-time streaming from built-in webcams (Index 0), external USB webcams (Index 1+), or network RTSP/HTTP IP camera feeds with optional session recording to MP4.
- **Pretrained YOLO Models:** Instant support for `yolov8n.pt` (Nano), `yolov8s.pt` (Small), and `yolov8m.pt` (Medium).
- **Compatible Multi-Object Trackers:** Seamless integration of both **ByteTrack** and **BoT-SORT**.
- **Dynamic Class Filtering:** Filter by convenient category presets (*Vehicles*, *Pedestrians*, *Animals*, *Electronics*) or choose custom COCO classes.
- **Visual Overlays:**
  - Corner-bracket aesthetic bounding boxes.
  - Glowing motion trajectory trails with configurable history length.
  - High-contrast colored pill badges with Track ID, Class Name, and Confidence Score.
  - Real-time semi-transparent HUD statistics panel.
- **Built-in Benchmark Video:** Instant synthetic benchmark generator to test detection and tracking out of the box without uploading external files.
- **Live Performance Metrics:** Frame counter, processing speed (FPS), elapsed time, ETA countdown, and active object metrics.
- **Browser-Ready Video Export:** Encodes annotated frames to standard H.264 MP4 with instant download.
- **Robust Error Handling:** Validates video format, handles missing files, zero-frame clips, and model network exceptions gracefully.

---

## 📁 Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
│
├── app.py              # Streamlit web application & user interface
├── detector.py         # YOLO model loading, class presets, inference engine
├── tracker.py          # ByteTrack/BoT-SORT tracking, trajectory trails, overlays
├── video_utils.py      # Video validation, synthetic demo generator, H.264 MP4 encoder
├── requirements.txt    # Pinned Python package dependencies
├── .gitignore          # Excludes model weights, videos, and virtual environments
└── README.md           # Internship documentation & user guide
```

---

## 💻 Hardware & Memory Requirements

### Recommended Hardware
| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i3 / AMD Ryzen 3 (4 cores) | Intel Core i7 / AMD Ryzen 7 (8+ cores) |
| **Memory (RAM)** | 4 GB RAM | 8 GB – 16 GB RAM |
| **GPU Acceleration** | None (CPU mode supported) | NVIDIA GTX 1660 / RTX 3060+ (CUDA 11.8+) |
| **Storage** | 500 MB free disk space | 2 GB free disk space |

### Processing Speed Benchmarks (YOLOv8 Nano, 720p Video)
- **Modern CPU (e.g. Intel 13th Gen / AMD 7000):** ~20 – 35 FPS
- **NVIDIA RTX GPU:** ~85 – 140 FPS
- *Tip: If running on modest CPU hardware, set "Process Every Nth Frame" to 2 or 3 in the sidebar for accelerated processing.*

---

## 📥 Model Weights & Download Guide

To prevent committing large binary files to version control, model weights (`*.pt`) are excluded in `.gitignore`.

### Automatic Download (Default)
When you select a model in the application sidebar (e.g., `yolov8n.pt`), the `YOLODetector` automatically downloads the official release weights from Ultralytics directly to your project cache on first run.

### Manual Download (Offline Environments)
If working behind a restricted firewall or offline:
1. Download `yolov8n.pt` directly from GitHub:  
   `https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`
2. Place the downloaded `.pt` file directly into the project root directory:
   ```text
   CodeAlpha_ObjectDetectionTracking/yolov8n.pt
   ```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

---

## 🧪 Testing the Smallest Working Version

To verify that your installation, OpenCV, and YOLO tracker components function correctly:

```bash
# Run a quick automated smoke test
python -c "from detector import YOLODetector; from tracker import ObjectTracker; print('Components verified successfully!')"
```

---

## ⚠️ Honest Limitations

1. **COCO Class Boundaries:** Pretrained YOLO models recognize 80 standardized COCO categories. Custom or niche objects (e.g., specific industrial components, micro-organisms) require fine-tuning.
2. **Prolonged Occlusion:** If an object is completely hidden for more than the tracker's memory horizon (typically 30 frames), it may receive a newly incremented tracking ID upon re-entry.
3. **Camera Jitter:** In handheld videos with abrupt, violent camera shaking, standard ByteTrack might experience ID switches; BoT-SORT is recommended for moving camera scenarios as it includes Camera Motion Compensation (CMC).
4. **Browser Video Codec Constraints:** Standard web browsers strictly support H.264 encoded MP4s. If using external video tools, ensure H.264 encoding rather than generic raw AVI codecs.

---

## 🎓 Internship Information
- **Internship Organization:** CodeAlpha
- **Track:** Artificial Intelligence
- **Task:** Object Detection & Tracking in Videos
- **Project Name:** `CodeAlpha_ObjectDetectionTracking`
