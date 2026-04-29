"""Ground Target Tracker — Streamlit UI for vision-based UAV ground target detection & auto-tracking."""

import sys
import time
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

from yolo_detector import YoloDetector, COCO_NAMES, DEFAULT_GROUND_CLASSES
from targeting import ScreenCenterTargeter, targeter_from_config
from overlay import draw_detections, draw_trajectories, draw_targeting_reticle, draw_hud

CONFIG_PATH = ROOT / "config" / "pipeline.yaml"
YOLO_DEFAULT = ROOT / "models" / "yolov8n.pt"


def _load_config() -> dict:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


# ── Visual fallback detector (contrast-based, no model required) ───────────────

def detect_visual(frame: np.ndarray, conf_thresh: float = 0.1) -> list:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.absdiff(bg, cv2.GaussianBlur(gray, (5, 5), 0))
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame.shape[:2]
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w / max(h, 1) > 5 or w / max(h, 1) < 0.2:
            continue
        pad = int(max(w, h) * 0.3)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
        roi_mean = gray[y:y+h, x:x+w].mean()
        conf = min(1.0, abs(bg[y:y+h, x:x+w].mean() - roi_mean) / 80.0)
        if conf >= conf_thresh:
            dets.append([float(x1), float(y1), float(x2), float(y2), conf, 0])
    return dets


def _iou_track(detections: list, state: dict) -> dict:
    """Minimal stateful IoU tracker for the visual fallback mode."""
    tracks, next_id = state["tracks"], state["next_id"]

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0.0

    if not tracks:
        for det in detections:
            tracks[next_id] = {"bbox": list(det[:4]), "conf": float(det[4]), "cls": int(det[5]), "age": 0}
            next_id += 1
        state["next_id"] = next_id
        return dict(tracks)

    matched_t, matched_d = set(), set()
    for tid in list(tracks):
        best, best_j = 0.3, -1
        for j, det in enumerate(detections):
            if j in matched_d:
                continue
            v = iou(tracks[tid]["bbox"], det[:4])
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            tracks[tid].update({"bbox": list(detections[best_j][:4]),
                                  "conf": float(detections[best_j][4]),
                                  "cls": int(detections[best_j][5]), "age": 0})
            matched_t.add(tid); matched_d.add(best_j)

    for tid in list(tracks):
        if tid not in matched_t:
            tracks[tid]["age"] += 1
            if tracks[tid]["age"] > 15:
                del tracks[tid]

    for j, det in enumerate(detections):
        if j not in matched_d:
            tracks[next_id] = {"bbox": list(det[:4]), "conf": float(det[4]), "cls": int(det[5]), "age": 0}
            next_id += 1

    state["next_id"] = next_id
    return dict(tracks)


st.set_page_config(page_title="Ground Target Tracker", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("Ground Target Tracker")
st.sidebar.markdown("---")

st.sidebar.subheader("Input Source")
source_type = st.sidebar.radio("Source", ["Upload Video", "Webcam", "RTSP / File Path"])

uploaded_file = None
source_path = None

if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Video file", type=["mp4", "avi", "mkv", "mov"])
elif source_type == "Webcam":
    cam_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=0)
    source_path = int(cam_index)
else:
    source_path = st.sidebar.text_input("Path or RTSP URL", placeholder="rtsp://... or /path/to/video.mp4")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection")

det_mode = st.sidebar.radio(
    "Detection mode",
    ["YOLOv8 (GPU)", "Visual (no model)"],
    help="YOLOv8 gives best performance for ground targets.",
)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
max_frames = st.sidebar.number_input("Max frames (0 = all)", min_value=0, max_value=10000, value=0)

if det_mode == "YOLOv8 (GPU)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("YOLOv8 Settings")
    yolo_model_path = st.sidebar.text_input("Model path (.pt or .engine)", value=str(YOLO_DEFAULT))
    yolo_input_size = st.sidebar.selectbox("Input size", [320, 640, 1280], index=1)
    yolo_device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)
    default_cls_names = [COCO_NAMES[i] for i in DEFAULT_GROUND_CLASSES if i < len(COCO_NAMES)]
    selected_classes = st.sidebar.multiselect(
        "Filter classes", options=COCO_NAMES, default=default_cls_names,
        help="Empty = detect all COCO classes.",
    )
    yolo_classes = [COCO_NAMES.index(c) for c in selected_classes if c in COCO_NAMES] or None
else:
    yolo_model_path = str(YOLO_DEFAULT)
    yolo_input_size = 640
    yolo_device = "cuda"
    yolo_classes = None

st.sidebar.markdown("---")
st.sidebar.subheader("Targeting")

lock_mode = st.sidebar.selectbox(
    "Target lock mode", ["nearest_center", "largest", "highest_conf"],
    help="How to pick the primary target.",
)
show_reticle = st.sidebar.checkbox("Show targeting reticle", value=True)
show_error_signal = st.sidebar.checkbox("Show error signal", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Display")
show_trajectories = st.sidebar.checkbox("Show trajectories", value=True)
show_fps = st.sidebar.checkbox("Show FPS overlay", value=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_run, tab_config, tab_about = st.tabs(["Run", "Config", "About"])

with tab_config:
    st.subheader("config/pipeline.yaml")
    try:
        raw_yaml = CONFIG_PATH.read_text()
    except FileNotFoundError:
        raw_yaml = "# config/pipeline.yaml not found"
    edited_yaml = st.text_area("Edit config", value=raw_yaml, height=500)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save"):
            try:
                yaml.safe_load(edited_yaml)
                CONFIG_PATH.write_text(edited_yaml)
                st.success("Saved.")
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
    with c2:
        if st.button("Reset"):
            st.rerun()

with tab_about:
    st.subheader("Ground Target Tracker")
    st.markdown(
        """
Vision-based UAV ground target detection and auto-tracking system.

**Pipeline:**
- **Detection:** YOLOv8 (GPU) — real-time ground target detection
- **Tracking:** ByteTrack (ultralytics) with IoU tracker fallback
- **Targeting:** Screen-center lock — computes normalized error signal (err_x, err_y)
  and estimated pan/tilt angles for gimbal control

**Default ground target classes (COCO):**
person · bicycle · car · motorcycle · bus · truck

**To use a custom model**, export your YOLOv8 checkpoint with:
```bash
python training/export_yolov8.py --model runs/.../best.pt --output models/custom.onnx
```
        """
    )
    col_a, col_b = st.columns(2)
    col_a.metric("YOLOv8 model", "Found" if Path(yolo_model_path).exists() else "Will auto-download")


# ── Cached model loader ────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo(model_path: str, input_size: int, device: str,
              conf: float, classes_key: str) -> YoloDetector | None:
    classes = [int(c) for c in classes_key.split(",") if c] if classes_key else None
    try:
        return YoloDetector(
            model_path=model_path, conf_threshold=conf,
            input_size=input_size, classes=classes, device=device,
        )
    except Exception as e:
        st.error(f"YOLOv8 load failed: {e}")
        return None


# ── Run Tab ────────────────────────────────────────────────────────────────────

with tab_run:
    video_source = None
    tmp_path = None

    if source_type == "Upload Video" and uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_path = tmp.name
        video_source = tmp_path
    elif source_type == "Webcam":
        video_source = source_path
    elif source_type == "RTSP / File Path" and source_path:
        video_source = source_path

    if video_source is None:
        st.info("Select a video source in the sidebar to get started.")
    else:
        col_run, col_dl = st.columns([3, 1])
        run_btn = col_run.button("Start Tracking", type="primary", use_container_width=True)

        if run_btn:
            # Load detector
            yolo_det: YoloDetector | None = None
            active_mode = det_mode

            if active_mode == "YOLOv8 (GPU)":
                classes_key = ",".join(str(c) for c in (yolo_classes or []))
                yolo_det = load_yolo(yolo_model_path, yolo_input_size,
                                     yolo_device, conf_thresh, classes_key)
                if yolo_det is None:
                    st.warning("YOLOv8 unavailable — falling back to Visual mode.")
                    active_mode = "Visual (no model)"

            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(f"Cannot open: `{video_source}`")
            else:
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                limit = max_frames if max_frames > 0 else (total if total > 0 else 9_999_999)

                out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                out_path = out_tmp.name
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (fw, fh)
                )

                config = _load_config()
                config.setdefault("targeting", {})["selection_mode"] = lock_mode
                targeter = targeter_from_config(config, fw, fh)

                class_names = yolo_det.class_names if yolo_det else COCO_NAMES
                track_history: dict[int, list] = {}
                _iou_state: dict = {"tracks": {}, "next_id": 1}  # for visual fallback

                frame_placeholder = st.empty()
                metrics_placeholder = st.empty()
                targeting_placeholder = st.empty()
                progress = st.progress(0.0, text="Processing…")
                stop_btn = st.button("Stop")

                frame_count = 0
                total_ms = 0.0

                while frame_count < limit:
                    if stop_btn:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    t0 = time.perf_counter()
                    if active_mode == "YOLOv8 (GPU)" and yolo_det:
                        tracks = yolo_det.detect_and_track(frame)
                    else:
                        detections = detect_visual(frame, conf_thresh)
                        tracks = _iou_track(detections, _iou_state)
                    det_ms = (time.perf_counter() - t0) * 1000
                    total_ms += det_ms

                    for tid, t in tracks.items():
                        track_history.setdefault(tid, []).append(list(t["bbox"]))
                        if len(track_history[tid]) > 40:
                            track_history[tid].pop(0)

                    target_state = targeter.update(tracks)
                    primary_id = target_state.track_id if target_state else -1
                    fps_val = 1000.0 / det_ms if det_ms > 0 else 0.0

                    annotated = frame.copy()
                    if show_trajectories:
                        draw_trajectories(annotated, track_history, primary_id)
                    draw_detections(annotated, tracks, class_names, primary_id)
                    if show_reticle and target_state:
                        draw_targeting_reticle(annotated, target_state)
                    if show_fps:
                        draw_hud(annotated, fps_val, len(tracks), target_state)

                    writer.write(annotated)
                    frame_count += 1

                    if frame_count % 5 == 0 or frame_count == 1:
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                        avg_fps = 1000.0 / (total_ms / frame_count)
                        metrics_placeholder.markdown(
                            f"**Frame** {frame_count}/{limit if limit < 9_999_999 else '∞'} &nbsp;|&nbsp;"
                            f" **FPS** {fps_val:.0f} (avg {avg_fps:.0f}) &nbsp;|&nbsp;"
                            f" **Tracks** {len(tracks)} &nbsp;|&nbsp;"
                            f" **Det** {det_ms:.1f} ms"
                        )

                        if target_state and show_error_signal:
                            q = target_state.lock_quality
                            q_color = "green" if q > 0.6 else ("orange" if q > 0.3 else "red")
                            cls_name = (class_names[target_state.class_id]
                                        if target_state.class_id < len(class_names) else "?")
                            targeting_placeholder.markdown(
                                f"**Target** T{target_state.track_id:02d} · {cls_name}"
                                f" {target_state.confidence:.0%} &nbsp;|&nbsp;"
                                f" **err_x** `{target_state.error_x:+.3f}`"
                                f" **err_y** `{target_state.error_y:+.3f}` &nbsp;|&nbsp;"
                                f" **Pan** {target_state.angle_pan:+.1f}°"
                                f" **Tilt** {target_state.angle_tilt:+.1f}° &nbsp;|&nbsp;"
                                f" **Lock** :{q_color}[{q:.0%}]"
                            )

                    if total > 0:
                        progress.progress(min(frame_count / limit, 1.0),
                                          text=f"Frame {frame_count} / {min(limit, total)}")

                cap.release()
                writer.release()
                progress.progress(1.0, text="Done!")

                avg_fps = 1000.0 / (total_ms / max(frame_count, 1))
                st.success(
                    f"Processed {frame_count} frames · avg {avg_fps:.0f} FPS · "
                    f"{len(track_history)} unique tracks"
                )

                with open(out_path, "rb") as f:
                    col_dl.download_button(
                        "Download output", data=f,
                        file_name="ground_tracker_output.mp4",
                        mime="video/mp4", use_container_width=True,
                    )

                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
