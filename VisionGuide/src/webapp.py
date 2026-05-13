import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

# Ensure local imports (matches your app.py relative imports)
from core.object_detector import ObjectDetector
from core.depth_estimator import DepthEstimator
from core.scene_analyzer import SceneAnalyzer, format_for_audio
from core.scene_tracker import SceneTracker
from core.audio_processor import AudioProcessor, AudioPriority
from core.easy_calibrator import EasyCalibrator
from utils.config import config

# ----------------------------
# UI Helpers
# ----------------------------
def draw_detections(frame: np.ndarray, detected_objects, step_size: float) -> np.ndarray:
    display = frame.copy()
    for obj in detected_objects:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{obj.class_name} {obj.confidence:.2f}"
        if obj.distance:
            # Select color by distance
            if obj.distance < 1.5:
                color = (0, 0, 255)
                dist_text = "very close"
            elif obj.distance < 3.0:
                color = (0, 165, 255)
                dist_text = f"{obj.distance:.1f} steps"
            else:
                color = (0, 255, 0)
                dist_text = f"{obj.distance:.0f} steps"
            label += f" ({dist_text})"
        else:
            color = (0, 255, 0)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(display, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # direction hint
        if obj.direction and obj.direction != "center":
            cv2.putText(display, f"{obj.direction}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return display

def run_inference_on_frame(frame: np.ndarray,
                           detector: ObjectDetector,
                           depth_estimator: DepthEstimator,
                           analyzer: SceneAnalyzer,
                           tracker: SceneTracker,
                           step_size: float,
                           conf_thr: float,
                           iou_thr: float):
    # Horizontal flip to match your app default
    frame = cv2.flip(frame, 1)
    detected = detector.detect_objects(frame, confidence_threshold=conf_thr, iou_threshold=iou_thr)
    depth_map = depth_estimator.estimate_depth(frame)
    for obj in detected:
        obj.distance = depth_estimator.get_object_distance(depth_map, obj.bbox, step_size)
    changes = tracker.update_scene(detected)
    context = analyzer.analyze_scene(detected, depth_map)
    audio_text = format_for_audio(context, detected)
    return detected, depth_map, changes, context, audio_text, frame

# ----------------------------
# App Setup
# ----------------------------
st.set_page_config(page_title="VisionGuide AI - Web", page_icon="👁️", layout="wide")

# Persistent state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.last_announcement = ""
    st.session_state.auto_announce = True
    st.session_state.last_summary_time = 0.0
    st.session_state.logs = []
    st.session_state.voice_enabled = False
    st.session_state.calibration_status = None

# Sidebar controls
st.sidebar.title("VisionGuide AI")
mode = st.sidebar.radio("Mode", ["Live Camera", "Upload"])

# Use your Pydantic config properly
conf_thr = st.sidebar.slider("Confidence threshold", 0.1, 0.9, config.model.confidence_threshold, 0.05)
iou_thr = st.sidebar.slider("IoU threshold", 0.2, 0.9, config.model.iou_threshold, 0.05)
step_size = st.sidebar.slider("Step size (meters)", 0.25, 1.5, config.navigation.step_size, 0.05)
auto_announce = st.sidebar.toggle("Auto announce changes", True, help="Speak only when meaningful changes happen")
st.session_state.auto_announce = auto_announce

col_a, col_b = st.sidebar.columns(2)
with col_a:
    voice_toggle = st.toggle("Voice output", False, help="Use system TTS; may not work on some servers")
with col_b:
    do_summary = st.button("Speak summary")

st.sidebar.divider()
st.sidebar.subheader("Calibration")
cal_show = st.sidebar.button("Show status")
cal_reset = st.sidebar.button("Reset calibration")

st.sidebar.divider()
st.sidebar.caption("Tip: Run on a machine with a camera and CUDA for best performance.")

# Initialize core components once
if not st.session_state.initialized:
    try:
        detector = ObjectDetector(config.model.yolo_model_path)
        depth_estimator = DepthEstimator(model_type=config.model.depth_model_type)
        analyzer = SceneAnalyzer()
        tracker = SceneTracker()
        audio = AudioProcessor()
        easy_cal = EasyCalibrator(depth_estimator)

        # Start audio (might fail on some systems, handle gracefully)
        try:
            audio.start()
            audio.test_audio()
            st.session_state.logs.append("Audio system initialized successfully.")
        except Exception as audio_err:
            st.session_state.logs.append(f"Audio initialization failed (non-critical): {audio_err}")

        st.session_state.detector = detector
        st.session_state.depth_estimator = depth_estimator
        st.session_state.analyzer = analyzer
        st.session_state.tracker = tracker
        st.session_state.audio = audio
        st.session_state.easy_cal = easy_cal

        st.session_state.initialized = True
        st.session_state.logs.append("Initialized VisionGuide AI core modules.")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.write("**Debug info:**")
        st.write(f"- Config camera index: {config.camera_index}")
        st.write(f"- Config YOLO model: {config.model.yolo_model_path}")
        st.write(f"- Config depth model: {config.model.depth_model_type}")
        st.stop()

# Shortcuts into session state
detector = st.session_state.detector
depth_estimator = st.session_state.depth_estimator
analyzer = st.session_state.analyzer
tracker = st.session_state.tracker
audio = st.session_state.audio
easy_cal = st.session_state.easy_cal

# Top layout
header_left, header_right = st.columns([3, 2])
with header_left:
    st.title("VisionGuide AI — Web Interface")
    st.caption("Real-time detection, distance estimation, and scene awareness.")
with header_right:
    st.metric("Auto announce", "ON" if st.session_state.auto_announce else "OFF")
    st.metric("Voice", "ON" if voice_toggle else "OFF")
    if st.session_state.last_announcement:
        st.info(f"Last announcement: {st.session_state.last_announcement}")

tab1, tab2, tab3 = st.tabs(["Run", "Calibration", "Logs"])

# ----------------------------
# Run tab
# ----------------------------
with tab1:
    if mode == "Upload":
        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
        with up_col2:
            run_button = st.button("Process")

        if run_button and file is not None:
            filename = file.name.lower()
            temp_path = Path("uploaded") / f"{int(time.time())}_{file.name}"
            temp_path.parent.mkdir(exist_ok=True, parents=True)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                detected, depth_map, changes, context, audio_text, frame = run_inference_on_frame(
                    img, detector, depth_estimator, analyzer, tracker, step_size, conf_thr, iou_thr
                )
                vis = draw_detections(frame, detected, step_size)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)
                
                # Results display
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Scene:** {context.environment_type}")
                    st.markdown(f"**Objects detected:** {len(detected)}")
                with col2:
                    if detected:
                        closest = min(detected, key=lambda x: x.distance if x.distance else float('inf'))
                        st.markdown(f"**Closest object:** {closest.class_name}")
                        if closest.distance:
                            st.markdown(f"**Distance:** {closest.distance:.1f} steps")
                
                if audio_text and audio_text.strip() != ".":
                    st.success(f"🔊 **Scene Description:** {audio_text}")
                    if voice_toggle:
                        try:
                            audio.speak_async(audio_text, AudioPriority.HIGH)
                            st.session_state.last_announcement = audio_text
                        except Exception as e:
                            st.session_state.logs.append(f"TTS failed: {e}")
                
                if changes:
                    st.write("**Detected changes:**")
                    for _, msg in list(changes.items())[:4]:
                        st.write(f"- {msg}")

    else:
        # Live camera
        cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=config.camera_index, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            start_live = st.button("🎥 Start Live Session", type="primary")
        with col2:
            stop_voice = st.button("🔇 Stop Voice")

        if stop_voice:
            try:
                audio.stop_speaking()
                st.success("Voice stopped")
            except Exception:
                st.warning("Could not stop voice")

        if start_live:
            cap = cv2.VideoCapture(int(cam_index))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
            cap.set(cv2.CAP_PROP_FPS, config.fps)

            if not cap.isOpened():
                st.error("❌ Unable to open camera. Please check:")
                st.write("- Camera is connected")
                st.write("- No other applications are using the camera")
                st.write("- Try a different camera index")
            else:
                st.success("✅ Camera opened successfully")
                
                # Create placeholders
                frame_ph = st.empty()
                info_ph = st.empty()
                stats_ph = st.empty()
                
                # Control buttons
                stop_button = st.button("⏹️ Stop Live Session")
                
                frame_count = 0
                start_time = time.time()
                
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera read failed")
                        break

                    frame_count += 1
                    
                    try:
                        detected, depth_map, changes, context, audio_text, processed_frame = run_inference_on_frame(
                            frame, detector, depth_estimator, analyzer, tracker, step_size, conf_thr, iou_thr
                        )
                        vis = draw_detections(processed_frame, detected, step_size)
                        frame_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Info display
                        object_names = [o.class_name for o in detected[:6]]
                        info_ph.write(f"**Scene:** {context.environment_type} | **Objects:** {', '.join(object_names) if object_names else 'None detected'}")
                        
                        # Stats
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        stats_ph.write(f"**FPS:** {fps:.1f} | **Frames:** {frame_count} | **Objects:** {len(detected)}")

                        # Announcements
                        if changes and st.session_state.auto_announce and voice_toggle and tracker.should_announce():
                            msgs = [m for _, m in changes.items() if m and len(m) > 3][:2]
                            if msgs:
                                announcement = ". ".join(msgs)
                                try:
                                    audio.speak_async(announcement, AudioPriority.HIGH)
                                    st.session_state.last_announcement = announcement
                                    st.session_state.logs.append(f"Announced: {announcement}")
                                except Exception as e:
                                    st.session_state.logs.append(f"TTS failed: {e}")

                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        st.session_state.logs.append(f"Frame processing error: {e}")
                    
                    time.sleep(0.05)  # ~20 FPS max

                cap.release()
                st.info("🎥 Live session stopped")

# ----------------------------
# Calibration tab
# ----------------------------
with tab2:
    st.subheader("📏 Manual Calibration")
    st.caption("Upload an image with objects at known distances to improve depth estimation accuracy.")
    
    calib_img = st.file_uploader("Upload calibration image", type=["jpg", "jpeg", "png"])
    
    if calib_img:
        colL, colR = st.columns([2, 1])
        
        with colL:
            img = cv2.imdecode(np.frombuffer(calib_img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            detected, depth_map, _, _, _, frame = run_inference_on_frame(
                img, detector, depth_estimator, analyzer, tracker, step_size, conf_thr, iou_thr
            )
            vis = draw_detections(frame, detected, step_size)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with colR:
            if detected:
                st.write("**Select object to calibrate:**")
                options = [f"{i}: {o.class_name} ({o.distance:.1f} steps)" if o.distance else f"{i}: {o.class_name}" for i, o in enumerate(detected)]
                selected = st.selectbox("Object", options)
                choice = int(selected.split(":")[0])
                
                st.write("**Actual distance:**")
                meters = st.select_slider("Distance (meters)", options=[0.5,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0], value=2.0)
                
                confidence = st.slider("Measurement confidence", 0.5, 1.0, 0.9, 0.05)
                
                if st.button("✅ Add Calibration Point", type="primary"):
                    obj = detected[choice]
                    success = depth_estimator.add_calibration_point(img, obj.bbox, meters, confidence=confidence)
                    if success:
                        st.success(f"✅ Calibration point added: {obj.class_name} at {meters}m")
                        st.session_state.logs.append(f"Calibration: {obj.class_name} at {meters}m")
                    else:
                        st.error("❌ Failed to add calibration point")
            else:
                st.info("No objects detected in this image")

    st.divider()
    
    # Calibration status
    col1, col2 = st.columns(2)
    with col1:
        if cal_show or st.button("📊 Show Calibration Status"):
            status = easy_cal.get_calibration_status()
            st.session_state.calibration_status = status
    
    with col2:
        if cal_reset or st.button("🗑️ Reset Calibration"):
            depth_estimator.calibrator.reset_calibration()
            st.success("✅ Calibration data reset")
            st.session_state.calibration_status = None
    
    if st.session_state.calibration_status:
        s = st.session_state.calibration_status
        
        # Status display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Calibrated", "✅ Yes" if s['is_calibrated'] else "❌ No")
        with col2:
            st.metric("Quality Score", f"{s['quality_score']:.2f}")
        with col3:
            st.metric("Sample Count", s['sample_count'])
        
        st.write(f"**Model Type:** {s['model_type']}")
        
        if s.get("recommendations"):
            st.info("**Recommendations:**\n" + "\n".join([f"• {rec}" for rec in s["recommendations"]]))

# ----------------------------
# Logs tab
# ----------------------------
with tab3:
    st.subheader("📋 System Logs")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Recent system events and announcements")
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.logs = []
            st.success("Logs cleared")
    
    if st.session_state.logs:
        # Show logs in reverse order (newest first)
        for i, entry in enumerate(reversed(st.session_state.logs[-50:])):
            timestamp = time.strftime("%H:%M:%S")
            st.write(f"**{timestamp}** - {entry}")
    else:
        st.info("No logs available yet. Start using the system to see activity logs.")

# System info in sidebar
st.sidebar.divider()
st.sidebar.subheader("System Info")
st.sidebar.write(f"Config loaded: ✅")
st.sidebar.write(f"Camera index: {config.camera_index}")
st.sidebar.write(f"YOLO model: {config.model.yolo_model_path}")
st.sidebar.write(f"Depth model: {config.model.depth_model_type}")
