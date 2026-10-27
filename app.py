"""
CodeAlpha Artificial Intelligence Internship
Project: CodeAlpha_ObjectDetectionTracking
Main Application: app.py

A modern, high-performance web interface for Object Detection and Multi-Object
Tracking in video streams using Ultralytics YOLO, ByteTrack/BoT-SORT, and Streamlit.
"""

import os
import time
import tempfile
import psutil
import threading
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# WebRTC streaming components
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode
    )
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# Local modules
from detector import YOLODetector, CLASS_PRESETS, COCO_CLASSES
from tracker import ObjectTracker
from video_utils import (
    validate_video_file,
    generate_sample_demo_video,
    encode_frames_to_mp4,
    open_camera_capture,
    detect_available_cameras
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="CodeAlpha | Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern Styling with CSS Animations
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Glassmorphism Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }

    /* Animated Badge */
    .badge-internship {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(90deg, #0284c7, #2563eb);
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 14px;
        border-radius: 9999px;
        box-shadow: 0 0 14px rgba(14, 165, 233, 0.4);
        margin-bottom: 12px;
    }

    /* Glowing Pulse Dot */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    /* Metric Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-value {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .stat-accent {
        color: #38bdf8;
    }

    /* Custom Pill Tag for Active Classes */
    .class-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
    }

    /* Comparison Table Styling */
    .theory-table th {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #38bdf8 !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_cached_detector(model_choice: str) -> YOLODetector:
    """Caches YOLO model in memory to avoid reloading weights repeatedly."""
    return YOLODetector(model_name_or_path=model_choice)


def render_hero_banner():
    """Renders the top application hero banner with dynamic internship badge."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="badge-internship">
                <span class="pulse-dot"></span> CodeAlpha AI Internship Project
            </div>
            <h1 style="margin: 0; font-size: 2.3rem; font-weight: 700; background: linear-gradient(90deg, #ffffff, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Video Object Detection & Multi-Object Tracking
            </h1>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem; max-width: 820px;">
                High-accuracy spatial object localization combined with temporal identity association.
                Powered by <strong>Ultralytics YOLO</strong> and <strong>ByteTrack / BoT-SORT</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def sidebar_configuration():
    """Renders all controls and parameter adjustments in the Streamlit sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")

        # 1. Model Selector
        model_options = {
            "YOLOv8 Nano (Fastest, ~6 MB)": "yolov8n.pt",
            "YOLOv8 Small (Balanced, ~22 MB)": "yolov8s.pt",
            "YOLOv8 Medium (Accurate, ~50 MB)": "yolov8m.pt"
        }
        selected_model_label = st.selectbox(
            "YOLO Detection Model",
            options=list(model_options.keys()),
            index=0,
            help="Pretrained COCO models from Ultralytics. 'Nano' is recommended for smooth real-time CPU processing."
        )
        model_weights = model_options[selected_model_label]

        # 2. Tracking Algorithm Selector
        tracker_options = {
            "ByteTrack (Low Latency, High Association)": "bytetrack.yaml",
            "BoT-SORT (Camera Motion Compensation + Re-ID)": "botsort.yaml"
        }
        selected_tracker_label = st.selectbox(
            "Tracking Algorithm",
            options=list(tracker_options.keys()),
            index=0,
            help="ByteTrack utilizes both high and low confidence detection boxes to maintain tracks through occlusions. BoT-SORT adds camera motion compensation."
        )
        tracker_file = tracker_options[selected_tracker_label]

        st.markdown("---")
        st.markdown("### 🎯 Detection Thresholds")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.35,
            step=0.05,
            help="Filters out bounding boxes below this detection certainty score."
        )

        iou_threshold = st.slider(
            "IoU / NMS Threshold",
            min_value=0.20,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Non-Maximum Suppression threshold to merge overlapping candidate boxes."
        )

        st.markdown("---")
        st.markdown("### 🏷️ Target Classes")

        preset_choice = st.selectbox(
            "Filter Category Preset",
            options=list(CLASS_PRESETS.keys()) + ["Custom Class Selection"],
            index=0
        )

        all_coco_list = list(COCO_CLASSES.values())
        if preset_choice == "Custom Class Selection":
            selected_classes = st.multiselect(
                "Choose Specific COCO Classes",
                options=all_coco_list,
                default=["person", "car", "motorcycle", "bus", "bicycle"],
                help="Only detected objects matching these classes will be shown and tracked."
            )
        else:
            selected_classes = CLASS_PRESETS[preset_choice]
            st.caption(f"Active preset classes: **{len(selected_classes)}**")

        st.markdown("---")
        st.markdown("### 🎨 Visual & Overlay Options")

        show_boxes = st.checkbox("Draw Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Class Labels", value=True)
        show_conf = st.checkbox("Show Confidence Scores", value=True)
        show_tracks = st.checkbox("Show Persistent Track IDs (#)", value=True)
        show_trails = st.checkbox("Draw Motion Trajectory Trails", value=True)
        trail_length = st.slider("Motion Trail History (frames)", 5, 60, 25) if show_trails else 0
        show_counts_hud = st.checkbox("Show Real-time HUD Panel", value=True)

        st.markdown("---")
        st.markdown("### ⚡ Frame Skip / Processing Speed")
        frame_skip = st.slider(
            "Process Every Nth Frame",
            min_value=1,
            max_value=5,
            value=1,
            help="Increase to speed up video processing on low-spec hardware (1 = process all frames)."
        )

    return {
        "model_weights": model_weights,
        "tracker_file": tracker_file,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "selected_classes": selected_classes,
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "show_tracks": show_tracks,
        "show_trails": show_trails,
        "trail_length": trail_length,
        "show_counts_hud": show_counts_hud,
        "frame_skip": frame_skip
    }


if HAS_WEBRTC:
    class YOLOVideoProcessor(VideoProcessorBase):
        """WebRTC Video Processor for real-time continuous YOLO detection & tracking frames."""
        def __init__(self):
            self.lock = threading.Lock()
            self.detector = None
            self.tracker = None
            self.config = {}

        def update_params(self, config: dict, detector):
            with self.lock:
                self.config = config
                self.detector = detector
                if self.tracker is None:
                    self.tracker = ObjectTracker(
                        tracker_type=config.get("tracker_file", "bytetrack.yaml"),
                        trail_length=config.get("trail_length", 15)
                    )

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            if self.detector is None or self.detector.model is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            with self.lock:
                try:
                    active_tracks = self.tracker.update_with_yolo(
                        yolo_model=self.detector.model,
                        frame=img,
                        conf_threshold=self.config.get("conf_threshold", 0.35),
                        iou_threshold=self.config.get("iou_threshold", 0.50),
                        selected_classes=self.config.get("selected_classes"),
                        device=self.detector.device
                    )

                    annotated = self.tracker.draw_annotations(
                        frame=img,
                        active_tracks=active_tracks,
                        show_boxes=self.config.get("show_boxes", True),
                        show_labels=self.config.get("show_labels", True),
                        show_conf=self.config.get("show_conf", True),
                        show_tracks=self.config.get("show_tracks", True),
                        show_trails=self.config.get("show_trails", True),
                        show_counts_hud=self.config.get("show_counts_hud", True)
                    )
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")


def run_live_webcam_stream(config: dict):
    """Executes real-time object detection and tracking directly from browser webcam, hardware USB webcam, or RTSP stream."""
    st.markdown("#### 📹 Real Camera / Live Webcam Stream")

    cam_interface = st.radio(
        "Choose Camera Interface Mode:",
        [
            "🔴 Continuous Live WebRTC Stream (Real-Time 30 FPS - Browser/Cloud)",
            "📸 Single-Frame Browser Snapshot",
            "🖥️ Direct Local USB / RTSP Stream (Localhost PC Only)"
        ],
        horizontal=True,
        help="Use 'Continuous Live WebRTC' for real-time video streaming directly from your browser on Streamlit Cloud."
    )

    if cam_interface == "🔴 Continuous Live WebRTC Stream (Real-Time 30 FPS - Browser/Cloud)":
        if not HAS_WEBRTC:
            st.error("⚠️ `streamlit-webrtc` is not installed. Please run `pip install streamlit-webrtc av`.")
            return

        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 14px; border-radius: 8px; margin-bottom: 15px;">
                <h5 style="margin: 0 0 6px 0; color: #38bdf8;">🔴 Live WebRTC Video Streaming</h5>
                <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem;">
                    Click <strong>START</strong> below and allow browser camera permissions. Video is streamed in real time to YOLO with bounding boxes and trajectories drawn continuously.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            with st.spinner(f"Preparing {config['model_weights']}..."):
                detector = load_cached_detector(config["model_weights"])
        except Exception as e:
            st.error(f"❌ Failed to load YOLO Model: {e}")
            return

        # Multi-redundant public STUN servers for robust NAT / firewall traversal
        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    {"urls": ["stun:stun.cloudflare.com:3478"]},
                    {"urls": ["stun:stun.services.mozilla.com:3478"]}
                ]
            }
        )

        webrtc_ctx = webrtc_streamer(
            key="live-yolo-webrtc-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480, "max": 640},
                    "height": {"ideal": 360, "max": 480},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "style": {"width": "100%", "borderRadius": "8px"},
                "muted": True
            },
            async_processing=True
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.update_params(config, detector)

        st.caption(
            "⚠️ **Troubleshooting 'Network Problem'?** WebRTC requires open UDP ports. If your local Wi-Fi, cellular provider, or VPN blocks WebRTC UDP connections, "
            "select **📸 Single-Frame Browser Snapshot** above, or run `streamlit run app.py` on your computer for zero-lag hardware streaming."
        )
        return

    if cam_interface == "📸 Single-Frame Browser Snapshot":
        st.info("💡 **Browser Client Camera**: Directly captures a high-resolution photo from your device camera and analyzes all objects with YOLO.")
        camera_img = st.camera_input("📸 Take a camera snapshot to analyze & track:")
        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            try:
                with st.spinner(f"Analyzing snapshot with {config['model_weights']}..."):
                    detector = load_cached_detector(config["model_weights"])
                    tracker = ObjectTracker(
                        tracker_type=config["tracker_file"],
                        trail_length=config["trail_length"]
                    )
                    t0 = time.time()
                    active_tracks = tracker.update_with_yolo(
                        yolo_model=detector.model,
                        frame=cv2_img,
                        conf_threshold=config["conf_threshold"],
                        iou_threshold=config["iou_threshold"],
                        selected_classes=config["selected_classes"],
                        device=detector.device
                    )
                    dt = time.time() - t0
                    fps = 1.0 / dt if dt > 0 else 30.0

                    annotated = tracker.draw_annotations(
                        frame=cv2_img,
                        active_tracks=active_tracks,
                        show_boxes=config["show_boxes"],
                        show_labels=config["show_labels"],
                        show_conf=config["show_conf"],
                        show_tracks=config["show_tracks"],
                        show_trails=config["show_trails"],
                        show_counts_hud=config["show_counts_hud"],
                        fps=fps
                    )

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="🎯 YOLO Detection & Tracking Result", use_container_width=True)
                with col2:
                    st.markdown("#### 📊 Detections Summary")
                    counts = tracker.get_frame_class_counts(active_tracks)
                    if counts:
                        pills_html = "".join([
                            f"<span class='class-pill'>{cls_name}: <strong>{cnt}</strong></span>"
                            for cls_name, cnt in sorted(counts.items())
                        ])
                        st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No target objects detected in this snapshot. Try adjusting confidence threshold or classes in the sidebar.")
                    st.markdown("---")
                    st.markdown(
                        f"""
                        - **Inference Time:** `{dt*1000:.1f} ms` ({fps:.1f} FPS)
                        - **Objects Detected:** `{len(active_tracks)}`
                        - **Hardware Device:** `{detector.device.upper()}`
                        - **Model Weights:** `{config['model_weights']}`
                        """
                    )
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
        return

    # Direct Local USB / RTSP Stream
    st.caption("Direct low-latency OpenCV video capture from a physical hardware webcam plugged into the machine running Python, or a network RTSP stream.")

    col_cam1, col_cam2, col_cam3 = st.columns([2, 2, 2])
    with col_cam1:
        cam_source_type = st.selectbox(
            "Camera Device Source",
            ["Built-in / USB Webcam 0", "External USB Webcam 1", "RTSP / HTTP IP Camera URL"],
            index=0
        )
    with col_cam2:
        resolution_choice = st.selectbox(
            "Camera Resolution",
            ["640x480 (Recommended - High Speed 30+ FPS)", "1280x720 (HD 720p)", "320x240 (Ultra Fast CPU)"],
            index=0
        )
    with col_cam3:
        record_live = st.checkbox(
            "💾 Record Session to MP4",
            value=False,
            help="Buffers annotated frames so you can download the video after stopping."
        )

    res_w, res_h = 640, 480
    if "1280" in resolution_choice:
        res_w, res_h = 1280, 720
    elif "320" in resolution_choice:
        res_w, res_h = 320, 240

    camera_source = 0
    if cam_source_type == "External USB Webcam 1":
        camera_source = 1
    elif cam_source_type == "RTSP / HTTP IP Camera URL":
        camera_source = st.text_input("Enter Camera Stream URL:", "rtsp://192.168.1.100:554/live")

    # Action controls
    col_c1, col_c2, _ = st.columns([2, 2, 3])
    with col_c1:
        start_camera = st.button("🔴 Start Live Camera", type="primary", use_container_width=True)
    with col_c2:
        stop_camera = st.button("⏹️ Stop Camera", type="secondary", use_container_width=True)

    if stop_camera:
        st.session_state["camera_streaming"] = False

    if start_camera:
        st.session_state["camera_streaming"] = True
        st.session_state["live_recording_path"] = None

    if st.session_state.get("camera_streaming", False):
        try:
            with st.spinner(f"Loading {config['model_weights']}..."):
                detector = load_cached_detector(config["model_weights"])
        except Exception as e:
            st.error(f"❌ Failed to load YOLO Model: {e}")
            st.session_state["camera_streaming"] = False
            return

        tracker = ObjectTracker(
            tracker_type=config["tracker_file"],
            trail_length=config["trail_length"]
        )
        tracker.reset()

        cap = open_camera_capture(camera_source, width=res_w, height=res_h)
        if cap is None or not cap.isOpened():
            st.error("❌ Could not connect to a local hardware camera device.")
            st.warning(
                "🌐 **Are you accessing this app via Streamlit Cloud?**\n\n"
                "Cloud servers in remote data centers do not have physical USB webcams attached.\n\n"
                "👉 **To use your device's camera online**, switch to the **'🌐 Browser Webcam (Cloud & Mobile Compatible)'** mode above!\n\n"
                "👉 Or if testing on your local machine, run `streamlit run app.py` on your computer for direct hardware camera access."
            )
            st.session_state["camera_streaming"] = False
            return

        col_display1, col_display2 = st.columns([3, 2])
        with col_display1:
            st.markdown("#### 🔴 Live Tracked Camera Feed")
            live_cam_placeholder = st.empty()
        with col_display2:
            st.markdown("#### 📊 Active Objects Detected")
            live_counts_placeholder = st.empty()
            st.markdown("#### ⚡ Real-Time Diagnostics")
            live_stats_placeholder = st.empty()

        stop_btn_placeholder = st.empty()
        stop_live_btn = stop_btn_placeholder.button("⏹️ Stop Camera Feed Now", key="stop_live_active_btn", use_container_width=True)

        recorded_frames = []
        fps_start = time.time()
        frame_counter = 0

        try:
            while cap.isOpened() and st.session_state.get("camera_streaming", False):
                if stop_live_btn:
                    st.session_state["camera_streaming"] = False
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("⚠️ Camera stream ended or disconnected.")
                    break

                frame_counter += 1
                t0 = time.time()
                active_tracks = tracker.update_with_yolo(
                    yolo_model=detector.model,
                    frame=frame,
                    conf_threshold=config["conf_threshold"],
                    iou_threshold=config["iou_threshold"],
                    selected_classes=config["selected_classes"],
                    device=detector.device
                )
                dt = time.time() - t0
                current_fps = 1.0 / dt if dt > 0 else 30.0

                annotated_frame = tracker.draw_annotations(
                    frame=frame,
                    active_tracks=active_tracks,
                    show_boxes=config["show_boxes"],
                    show_labels=config["show_labels"],
                    show_conf=config["show_conf"],
                    show_tracks=config["show_tracks"],
                    show_trails=config["show_trails"],
                    show_counts_hud=config["show_counts_hud"],
                    fps=current_fps
                )

                if record_live:
                    recorded_frames.append(annotated_frame)

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                curr_counts = tracker.get_frame_class_counts(active_tracks)
                if curr_counts:
                    pills_html = "".join([
                        f"<span class='class-pill'>{cls_name}: <strong>{cnt}</strong></span>"
                        for cls_name, cnt in sorted(curr_counts.items())
                    ])
                    live_counts_placeholder.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
                else:
                    live_counts_placeholder.markdown("<em style='color:#94a3b8;'>Point camera at people, cell phones, cups, laptops...</em>", unsafe_allow_html=True)

                elapsed_tot = time.time() - fps_start
                avg_fps = frame_counter / elapsed_tot if elapsed_tot > 0 else 0.0

                live_stats_placeholder.markdown(
                    f"""
                    - **Stream Speed:** `{avg_fps:.1f} FPS` (instant: `{current_fps:.1f}`)
                    - **Active Tracks (Current Frame):** `{len(active_tracks)}`
                    - **Cumulative Unique Track IDs:** `{len(tracker.unique_tracked_ids)}`
                    - **Resolution:** `{frame.shape[1]} x {frame.shape[0]}`
                    - **Inference Hardware:** `{detector.device.upper()}`
                    - **Tracker Algorithm:** `{tracker.tracker_type}`
                    """
                )
        finally:
            cap.release()
            st.session_state["camera_streaming"] = False

        if record_live and len(recorded_frames) > 0:
            rec_path = os.path.join(tempfile.gettempdir(), f"webcam_recording_{int(time.time())}.mp4")
            with st.spinner("Saving recorded camera session to MP4..."):
                encode_frames_to_mp4(recorded_frames, rec_path, fps=max(10.0, avg_fps))
            if os.path.exists(rec_path):
                st.session_state["live_recording_path"] = rec_path
                st.success(f"🎬 Recorded {len(recorded_frames)} frames from live camera!")

    if st.session_state.get("live_recording_path") and os.path.exists(st.session_state["live_recording_path"]):
        rec_file = st.session_state["live_recording_path"]
        st.markdown("---")
        st.markdown("#### 🎬 Playback & Download Recorded Live Camera Session")
        st.video(rec_file)
        with open(rec_file, "rb") as f:
            rec_bytes = f.read()
        st.download_button(
            label="📥 Download Live Camera Video (MP4)",
            data=rec_bytes,
            file_name="camera_tracking_session.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )


def tab_live_processing(config: dict):
    """Handles video input selection, validation, real-time tracking pipeline, and output download."""
    st.markdown("### 📹 Video Source Selection")

    source_mode = st.radio(
        "Choose Video Input Source:",
        [
            "Upload Your Own Video File",
            "📹 Live Webcam / Real Camera",
            "Generate & Test Built-in Benchmark Video"
        ],
        horizontal=True
    )

    if source_mode == "📹 Live Webcam / Real Camera":
        run_live_webcam_stream(config)
        return

    temp_video_path = None
    original_video_name = "processed_video.mp4"

    if source_mode == "Upload Your Own Video File":
        uploaded_file = st.file_uploader(
            "Upload Video (Supported: MP4, AVI, MOV, MKV, WEBM)",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video containing moving objects, traffic, or pedestrians."
        )
        if uploaded_file is not None:
            original_video_name = uploaded_file.name
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.read())
            tfile.close()
            temp_video_path = tfile.name
    else:
        st.info("💡 **Benchmark Mode**: Generates an automated synthetic traffic & pedestrian simulation video to immediately test tracking with persistent IDs without needing to upload anything.")
        demo_btn = st.button("🚀 Generate & Load Benchmark Video", type="secondary")
        demo_file_path = os.path.join(tempfile.gettempdir(), "codealpha_benchmark.mp4")
        if demo_btn or os.path.exists(demo_file_path):
            if not os.path.exists(demo_file_path) or demo_btn:
                with st.spinner("Rendering synthetic benchmark video..."):
                    generate_sample_demo_video(demo_file_path, duration_sec=5, fps=25)
            temp_video_path = demo_file_path
            original_video_name = "benchmark_traffic.mp4"
            st.success("✅ Benchmark video ready for processing!")

    if not temp_video_path:
        st.warning("👈 Please upload a video file or generate the benchmark video to begin.")
        return

    # Validate video
    is_valid, msg, width, height, fps, total_frames = validate_video_file(temp_video_path)
    if not is_valid:
        st.error(f"❌ Video Validation Error: {msg}")
        return

    # Display Video Metadata Cards
    duration_sec = total_frames / fps if fps > 0 else 0
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 18px 0 24px 0;">
            <div class="stat-card">
                <div class="stat-label">Resolution</div>
                <div class="stat-value"><span class="stat-accent">{width}</span> x {height}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Frames</div>
                <div class="stat-value">{total_frames}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Native FPS</div>
                <div class="stat-value">{fps:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Video Length</div>
                <div class="stat-value">{duration_sec:.1f}s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Action buttons
    col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 4])
    with col_btn1:
        start_btn = st.button("▶️ Start Detection & Tracking", type="primary", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹️ Stop / Reset", type="secondary", use_container_width=True)

    # Initialize / Reset processing state
    if stop_btn:
        st.session_state["stop_processing"] = True
        st.session_state["processed_video_path"] = None
        st.rerun()

    if start_btn:
        st.session_state["stop_processing"] = False
        st.session_state["processed_video_path"] = None

        # Load Detector and Tracker
        try:
            with st.spinner(f"Loading {config['model_weights']}..."):
                detector = load_cached_detector(config["model_weights"])
        except Exception as e:
            st.error(f"❌ Failed to load YOLO Model: {e}")
            return

        tracker = ObjectTracker(
            tracker_type=config["tracker_file"],
            trail_length=config["trail_length"]
        )
        tracker.reset()

        # UI elements for live feedback
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        col_display1, col_display2 = st.columns([3, 2])
        with col_display1:
            st.markdown("#### 🔴 Live Tracked Frame")
            live_frame_placeholder = st.empty()
        with col_display2:
            st.markdown("#### 📊 Live Object Counts (Current Frame)")
            live_counts_placeholder = st.empty()
            st.markdown("#### 📈 Processing Metrics")
            live_stats_placeholder = st.empty()

        cap = cv2.VideoCapture(temp_video_path)
        frame_idx = 0
        processed_frames_list = []
        fps_meter_start = time.time()
        fps_current = 0.0

        target_classes = config["selected_classes"]
        device = detector.device

        # Process video frame by frame
        while cap.isOpened():
            if st.session_state.get("stop_processing", False):
                st.warning("⚠️ Processing stopped by user.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1

            # Handle frame skipping
            if frame_idx % config["frame_skip"] != 0:
                continue

            # Run tracking
            t0 = time.time()
            active_tracks = tracker.update_with_yolo(
                yolo_model=detector.model,
                frame=frame,
                conf_threshold=config["conf_threshold"],
                iou_threshold=config["iou_threshold"],
                selected_classes=target_classes,
                device=device
            )
            t_cost = time.time() - t0
            fps_current = 1.0 / t_cost if t_cost > 0 else 30.0

            # Render annotations
            annotated_frame = tracker.draw_annotations(
                frame=frame,
                active_tracks=active_tracks,
                show_boxes=config["show_boxes"],
                show_labels=config["show_labels"],
                show_conf=config["show_conf"],
                show_tracks=config["show_tracks"],
                show_trails=config["show_trails"],
                show_counts_hud=config["show_counts_hud"],
                fps=fps_current
            )

            processed_frames_list.append(annotated_frame)

            # Update live preview every 2 frames for smooth UI performance
            if len(processed_frames_list) % 2 == 0 or frame_idx == total_frames:
                # Convert BGR to RGB for Streamlit image rendering
                rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_frame_placeholder.image(rgb_preview, channels="RGB", use_container_width=True)

                # Update live counts
                curr_counts = tracker.get_frame_class_counts(active_tracks)
                if curr_counts:
                    pills_html = "".join([
                        f"<span class='class-pill'>{cls_name}: <strong>{count}</strong></span>"
                        for cls_name, count in sorted(curr_counts.items())
                    ])
                    live_counts_placeholder.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
                else:
                    live_counts_placeholder.markdown("<em style='color:#94a3b8;'>No target objects in current frame</em>", unsafe_allow_html=True)

                # Update stats
                progress = min(1.0, frame_idx / total_frames) if total_frames > 0 else 0.0
                progress_bar.progress(progress)
                elapsed = time.time() - fps_meter_start
                avg_fps = len(processed_frames_list) / elapsed if elapsed > 0 else 0.0
                rem_frames = total_frames - frame_idx
                eta_sec = rem_frames / (avg_fps * config["frame_skip"]) if avg_fps > 0 else 0.0

                status_text.markdown(
                    f"**Frame {frame_idx}/{total_frames}** ({progress*100:.1f}%) | "
                    f"**Speed:** {avg_fps:.1f} FPS | "
                    f"**Est. Time Remaining:** {eta_sec:.1f}s"
                )

                live_stats_placeholder.markdown(
                    f"""
                    - **Active Tracks (Current Frame):** `{len(active_tracks)}`
                    - **Cumulative Unique Track IDs:** `{len(tracker.unique_tracked_ids)}`
                    - **Inference Hardware:** `{device.upper()}`
                    - **Tracker Engine:** `{tracker.tracker_type}`
                    """
                )

        cap.release()
        status_text.markdown("✅ **Processing completed! Finalizing video encoding for download...**")

        # Encode processed frames to H.264 MP4
        output_save_path = os.path.join(tempfile.gettempdir(), f"tracked_{int(time.time())}.mp4")
        effective_fps = (fps / config["frame_skip"]) if config["frame_skip"] > 0 else fps
        with st.spinner("Encoding output video (H.264 / MP4)..."):
            success = encode_frames_to_mp4(processed_frames_list, output_save_path, fps=effective_fps)

        if success and os.path.exists(output_save_path):
            st.session_state["processed_video_path"] = output_save_path
            st.session_state["cumulative_counts"] = tracker.get_cumulative_class_counts()
            st.session_state["unique_ids_count"] = len(tracker.unique_tracked_ids)
            st.session_state["total_processed_frames"] = len(processed_frames_list)
            st.session_state["id_to_class"] = tracker.id_to_class
        else:
            st.error("Failed to encode video output.")

    # Render results section if video was processed
    if st.session_state.get("processed_video_path") and os.path.exists(st.session_state["processed_video_path"]):
        st.markdown("---")
        st.markdown("### 🎉 Processing Results & Video Download")

        res_path = st.session_state["processed_video_path"]
        cum_counts = st.session_state.get("cumulative_counts", {})
        unique_ids = st.session_state.get("unique_ids_count", 0)
        tot_frames = st.session_state.get("total_processed_frames", 0)

        # Summary KPIs
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 20px;">
                <div class="stat-card">
                    <div class="stat-label">Total Unique Objects Tracked</div>
                    <div class="stat-value" style="color: #38bdf8;">{unique_ids}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Processed Output Frames</div>
                    <div class="stat-value">{tot_frames}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Distinct Detected Classes</div>
                    <div class="stat-value">{len(cum_counts)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_v1, col_v2 = st.columns([3, 2])
        with col_v1:
            st.markdown("#### 🎬 Playback Processed Video")
            try:
                st.video(res_path)
            except Exception as e:
                st.info(f"Video ready. Download below to play in your local player. ({e})")

            # Direct Download Button
            with open(res_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="📥 Download Processed Video (MP4)",
                data=video_bytes,
                file_name=f"tracked_{original_video_name}",
                mime="video/mp4",
                type="primary",
                use_container_width=True
            )

        with col_v2:
            st.markdown("#### 📊 Cumulative Objects by Class")
            if cum_counts:
                df = pd.DataFrame(
                    list(cum_counts.items()),
                    columns=["Class", "Unique Count"]
                ).sort_values(by="Unique Count", ascending=False)
                st.bar_chart(df.set_index("Class"), color="#38bdf8")

                with st.expander("🔍 View Tracked Object Registry"):
                    id_dict = st.session_state.get("id_to_class", {})
                    if id_dict:
                        reg_df = pd.DataFrame([
                            {"Tracking ID": f"#{k}", "Class Name": v}
                            for k, v in sorted(id_dict.items())
                        ])
                        st.dataframe(reg_df, use_container_width=True, height=220)
            else:
                st.write("No objects were tracked.")


def tab_educational_guide():
    """Provides a rigorous, clear breakdown of Object Detection vs. Object Tracking."""
    st.markdown("## 🧠 Computer Vision Foundations: Detection vs. Tracking")

    st.markdown(
        """
        In computer vision and intelligent surveillance systems, **Object Detection** and
        **Object Tracking** are two complementary but fundamentally distinct tasks.
        Understanding their mathematical and architectural differences is key to building
        production-grade AI pipelines.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            r"""
            ### 🎯 1. Object Detection
            **"What is in this specific frame and where is it located?"**

            * **Scope:** Operates strictly on a **single static image** (or one isolated frame at a time).
            * **Temporal Memory:** **Zero temporal memory**. Frame $t$ is evaluated with zero knowledge of frame $t-1$ or frame $t+1$.
            * **Output:** Bounding box coordinates $[x_1, y_1, x_2, y_2]$, a classification label (e.g. `car`), and a confidence probability score $P \in [0, 1]$.
            * **Identity:** Lacks concept of persistent identity. If a car is detected in frame 10 and frame 11, the detector has no way of knowing whether it is the same car or a different one.
            * **Cost:** Computationally expensive (requires a full forward pass through deep convolutional or transformer backbones like YOLO, Faster R-CNN, or RT-DETR).
            """
        )

    with col2:
        st.markdown(
            """
            ### 🛰️ 2. Object Tracking
            **"Where did this specific object move across consecutive frames?"**

            * **Scope:** Operates over a **temporal sequence of frames** (video stream).
            * **Temporal Memory:** **High temporal awareness**. Maintains motion vectors, spatial trajectories, and visual appearance features.
            * **Output:** A **persistent tracking ID** (e.g. `ID: #4`), motion trajectory trails, velocity/direction, and smoothed bounding box coordinates.
            * **Identity:** Guarantees temporal consistency. An object retains its assigned ID across frames even during temporary occlusions or camera jitter.
            * **Cost:** Lightweight and fast when using motion models (e.g. Kalman Filter in ByteTrack), running at 100+ FPS after detections are supplied.
            """
        )

    st.markdown("---")
    st.markdown("### ⚖️ Side-by-Side Comparison")

    comparison_data = {
        "Attribute": [
            "Input Domain",
            "Temporal Awareness",
            "Output Identity",
            "Computational Overhead",
            "Handling Occlusions",
            "Primary Algorithms",
            "Use Case Goal"
        ],
        "Object Detection": [
            "Single static image / independent frame",
            "None (Markovian assumption, memoryless)",
            "Class label + Confidence score only",
            "High (Deep Neural Network forward pass)",
            "Fails if object is partially hidden",
            "YOLOv8, YOLOv11, Faster R-CNN, SSD",
            "Identify presence and location of objects"
        ],
        "Object Tracking": [
            "Continuous video stream (ordered frames)",
            "Active state estimation over time",
            "Persistent Unique ID (#1, #2, #3)",
            "Low to Moderate (Linear algebra / motion models)",
            "Recovers identity via Kalman Filter & Re-ID",
            "ByteTrack, BoT-SORT, DeepSORT, SORT",
            "Track paths, count unique items, measure speed"
        ]
    }

    st.table(pd.DataFrame(comparison_data))

    st.markdown("---")
    st.markdown("### 🔄 The Tracking-by-Detection Paradigm")
    st.markdown(
        """
        Modern state-of-the-art vision pipelines (including this application) use the **Tracking-by-Detection** paradigm:
        1. **Step 1 (Detector):** YOLO scans the current frame and outputs candidate bounding boxes with confidence scores.
        2. **Step 2 (Motion Prediction):** A Kalman Filter predicts where each existing tracked object should be in the current frame based on past velocity and position.
        3. **Step 3 (Data Association):** The algorithm calculates the **Intersection over Union (IoU)** and visual similarity between predicted track locations and new YOLO detections.
        4. **Step 4 (ByteTrack Innovation):** Unlike older trackers that discard low-confidence detections, **ByteTrack** retains low-score boxes to maintain tracks during heavy occlusions, motion blur, or scale changes, achieving unprecedented tracking stability!
        """
    )


def tab_hardware_and_diagnostics():
    """Displays hardware configuration, memory consumption, and execution environment."""
    st.markdown("## 💻 System Diagnostics & Performance Profiler")

    col_h1, col_h2, col_h3 = st.columns(3)

    # PyTorch and GPU Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_name = torch.cuda.get_device_name(0) if cuda_avail else "No CUDA GPU (CPU Mode)"
        torch_ver = torch.__version__
    except Exception as e:
        cuda_avail = False
        cuda_name = f"Unavailable: {e}"
        torch_ver = "Unknown"

    cpu_cores = psutil.cpu_count(logical=True)
    ram_info = psutil.virtual_memory()
    ram_total_gb = ram_info.total / (1024 ** 3)
    ram_used_gb = ram_info.used / (1024 ** 3)

    with col_h1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Inference Engine / Acceleration</div>
                <div class="stat-value" style="color: {'#10b981' if cuda_avail else '#f59e0b'};">
                    {'CUDA GPU' if cuda_avail else 'CPU'}
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">{cuda_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_h2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Host CPU Cores</div>
                <div class="stat-value">{cpu_cores} Cores</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">PyTorch {torch_ver}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_h3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">System Memory (RAM)</div>
                <div class="stat-value">{ram_used_gb:.1f} / {ram_total_gb:.1f} GB</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">Utilization: {ram_info.percent}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📋 Hardware & Memory Guidelines")
    st.markdown(
        """
        - **CPU Requirements:** Any modern multi-core x86/ARM processor (Intel i5/i7/i9, AMD Ryzen, or Apple Silicon). YOLOv8 Nano processes frames at 15–35 FPS on standard CPUs.
        - **GPU Acceleration (Optional):** NVIDIA GPU with CUDA 11.8+ or 12.x provides real-time processing at 60–120+ FPS.
        - **Memory (RAM):**
          - Application baseline: ~350 MB RAM
          - YOLOv8 Nano model weights: ~6 MB RAM
          - YOLOv8 Small model weights: ~22 MB RAM
          - Video frame buffer: ~200–500 MB RAM for 1080p clips
        - **Storage:** Less than 100 MB total disk space needed for application code and cached model weights.
        """
    )


def tab_internship_about():
    """Project documentation, internship requirements fulfillment, and limitations."""
    st.markdown("## 🎓 CodeAlpha AI Internship Project Overview")

    st.markdown(
        """
        ### Project Name: `CodeAlpha_ObjectDetectionTracking`
        This project was developed as part of the **CodeAlpha Artificial Intelligence Internship**.
        It implements a full-stack Computer Vision video analysis system delivering real-time
        object detection and multi-object tracking.
        """
    )

    st.markdown("### ✅ Requirements Checklist")
    checklist = [
        ("Support uploaded video files", "Supported: MP4, AVI, MOV, MKV, WEBM via file uploader and OpenCV stream decoding."),
        ("Pretrained YOLO model", "Integrated Ultralytics YOLOv8 (Nano, Small, Medium) with automatic weight caching."),
        ("Bounding boxes, labels, confidence scores", "Aesthetic corner-bracket bounding boxes, pill label badges, and confidence metrics."),
        ("Persistent tracking IDs with compatible tracker", "Integrated ByteTrack and BoT-SORT with persistent IDs and trajectory history."),
        ("Option to choose supported detection classes", "Interactive selector with category presets (Vehicles, People, Animals, etc.) or custom COCO classes."),
        ("Current-frame object counts by class", "Real-time count chips and HUD statistics overlay."),
        ("Video processing progress & FPS", "Live progress bar, elapsed time, ETA countdown, and instantaneous processing FPS."),
        ("Download processed video", "Full H.264 MP4 encoding with a direct download button for local playback."),
        ("Error handling for unsupported formats & model errors", "Validated video stream checks, empty file detection, and model loading fallbacks."),
        ("Explain difference between detection and tracking", "Comprehensive educational guide, comparison matrix, and architectural breakdown.")
    ]

    for req, details in checklist:
        st.markdown(f"- **{req}**: <span style='color:#38bdf8;'>{details}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ Known Limitations & Assumptions")
    st.markdown(
        """
        1. **Model Scope:** Pretrained models are trained on the 80 COCO classes. Objects outside the COCO taxonomy will not be detected unless fine-tuned.
        2. **Severe Occlusion:** If an object remains completely hidden for longer than the tracker's memory buffer (typically 30 frames), it may be assigned a new tracking ID upon reappearing.
        3. **Low-spec Hardware:** For high-resolution (4K) videos on CPU-only machines, use the **Frame Skip** option (e.g. process every 2nd or 3rd frame) to maintain responsive processing speeds.
        """
    )


def main():
    """Main application entrypoint."""
    render_hero_banner()
    config = sidebar_configuration()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Studio & Live Tracking",
        "📚 CV Theory: Detection vs Tracking",
        "💻 Hardware & Diagnostics",
        "ℹ️ About & Documentation"
    ])

    with tab1:
        tab_live_processing(config)

    with tab2:
        tab_educational_guide()

    with tab3:
        tab_hardware_and_diagnostics()

    with tab4:
        tab_internship_about()


if __name__ == "__main__":
    main()
