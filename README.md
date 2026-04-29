# Ground Target Tracker

Vision-based ground target detection and auto-tracking system for UAV/drone platforms. Uses YOLOv8 for real-time detection and ByteTrack for persistent multi-target tracking, with a screen-center lock that computes normalized error signals and gimbal pan/tilt angles.

---

## Features

- **YOLOv8 detection** — GPU-accelerated, supports `.pt` (PyTorch) and `.engine` (TensorRT) models
- **ByteTrack tracking** — persistent track IDs across occlusions; IoU tracker fallback if ultralytics unavailable
- **Screen-center lock** — selects primary target, outputs normalized `err_x / err_y` error signal and pan/tilt angles for gimbal control
- **Targeting reticle overlay** — crosshair, line-to-target, lock quality ring (green/yellow/red)
- **Trajectory trails** — fade-out motion history per track
- **Streamlit UI** — live video preview, metrics panel, annotated video download
- **Class filtering** — select any COCO classes; defaults to person, bicycle, car, motorcycle, bus, truck
- **Visual fallback** — contrast-based detector that works with no model at all

---

## Quick Start

```bash
# 1. Install dependencies (one time)
./run.sh install

# 2. Launch the UI
./run.sh
```

Open **http://localhost:8501** in your browser.

---

## Run Script

```
./run.sh [command]
```

| Command | Description |
|---------|-------------|
| `./run.sh` or `./run.sh ui` | Launch Streamlit UI |
| `./run.sh install` | Create venv + install all dependencies |
| `./run.sh test` | Run unit tests |
| `./run.sh export` | Export YOLOv8 `.pt` → `.onnx` |
| `./run.sh finetune -- [args]` | Fine-tune YOLOv8 on custom dataset |
| `./run.sh check` | Print Python / CUDA / package versions |
| `./run.sh help` | Show usage |

---

## Project Structure

```
ground-target-tracker/
├── app.py                      # Streamlit UI entry point
├── run.sh                      # Unified run script
├── requirements.txt            # Python dependencies
├── config/
│   └── pipeline.yaml           # Detection, tracking, targeting config
├── models/                     # Model weights (.pt, .onnx, .engine)
├── scripts/
│   ├── yolo_detector.py        # YOLOv8 detector wrapper
│   ├── bytrack.py              # ByteTrack + IoU fallback tracker
│   ├── targeting.py            # ScreenCenterTargeter — error signal + angles
│   └── overlay.py              # Drawing: boxes, reticle, HUD, trajectories
├── training/
│   ├── export_yolov8.py        # Export .pt → ONNX
│   └── finetune_yolov8.py      # Fine-tune on custom ground-target dataset
├── tests/
│   ├── test_yolo_detector.py
│   └── test_targeting.py
├── data/                       # Test videos, datasets
├── recordings/                 # Saved output videos
└── logs/
```

---

## Configuration

Edit `config/pipeline.yaml` to tune detection and targeting behaviour:

```yaml
yolo:
  model_path: "models/yolov8n.pt"   # swap to custom weights or .engine
  input_size: 640                   # 320 / 640 / 1280
  confidence_threshold: 0.30
  classes: [0, 1, 2, 3, 5, 7]      # COCO IDs — empty list = all classes

targeting:
  selection_mode: "nearest_center"  # nearest_center | largest | highest_conf
  smoothing_alpha: 0.3              # EMA weight — higher = faster response
  camera_fx: 1066.67                # Camera intrinsics for angle computation
  camera_fy: 1066.67
  camera_cx: 640.0
  camera_cy: 360.0
```

---

## Targeting Output

For each frame the targeting engine produces:

| Field | Description |
|-------|-------------|
| `err_x` | Horizontal error, −1 (far left) to +1 (far right), 0 = centered |
| `err_y` | Vertical error, −1 (top) to +1 (bottom), 0 = centered |
| `angle_pan` | Estimated gimbal pan angle in degrees |
| `angle_tilt` | Estimated gimbal tilt angle in degrees |
| `lock_quality` | 0–100 % — decays with track age, scales with confidence |

Feed `err_x` / `err_y` directly to a PID controller or gimbal rate command.

---

## Model

The default model (`yolov8n.pt`) is downloaded automatically by ultralytics on first run.

**To use a custom model**, train or fine-tune on your dataset then export:

```bash
# Fine-tune on your dataset
./run.sh finetune -- --data data/my_dataset.yaml --epochs 50 --model yolov8s.pt

# Export best checkpoint to ONNX
./run.sh export
```

Set `yolo.model_path` in `config/pipeline.yaml` to point to the new weights.

### Recommended models by hardware

| Hardware | Model | Input size | Expected FPS |
|----------|-------|-----------|-------------|
| RTX 3060+ | `yolov8m.pt` | 640 | 60+ |
| RTX 3060+ (TensorRT) | `yolov8m.engine` | 640 | 100+ |
| Jetson Orin | `yolov8s.engine` | 640 | 30–60 |
| CPU only | `yolov8n.pt` | 320 | 5–15 |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection + ByteTrack |
| `torch` / `torchvision` | GPU inference backend |
| `opencv-python` | Frame capture, drawing |
| `streamlit` | Web UI |
| `onnxruntime-gpu` | ONNX Runtime inference (C++ pipeline path) |

Install all at once: `./run.sh install`

---

## Tests

```bash
./run.sh test
```

Covers:
- COCO class name list correctness
- YOLOv8 detector output format and coordinate de-letterboxing
- ScreenCenterTargeter error signal, angle computation, all selection modes, EMA smoothing

---

## Input Sources

The UI supports three input sources selectable in the sidebar:

- **Upload Video** — MP4, AVI, MKV, MOV
- **Webcam** — any OpenCV camera index
- **RTSP / File Path** — `rtsp://...` stream or local file path
