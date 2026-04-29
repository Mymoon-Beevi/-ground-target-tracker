#!/usr/bin/env bash
# Ground Target Tracker — unified run script
# Usage:
#   ./run.sh              → launch Streamlit UI (default)
#   ./run.sh ui           → launch Streamlit UI
#   ./run.sh install      → create venv and install all dependencies
#   ./run.sh test         → run unit tests
#   ./run.sh export       → export YOLOv8 model to ONNX
#   ./run.sh finetune     → fine-tune YOLOv8 on custom dataset (pass args after --)
#                           e.g. ./run.sh finetune -- --data data/dataset.yaml --epochs 50
#   ./run.sh check        → check environment (Python, CUDA, packages)

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[GTT]${NC} $*"; }
success() { echo -e "${GREEN}[GTT]${NC} $*"; }
warn()    { echo -e "${YELLOW}[GTT]${NC} $*"; }
error()   { echo -e "${RED}[GTT]${NC} $*" >&2; exit 1; }

# ── helpers ───────────────────────────────────────────────────────────────────

ensure_venv() {
    if [ ! -f "$PYTHON" ]; then
        warn "Virtual environment not found. Run './run.sh install' first."
        error "Aborting."
    fi
}

cmd_install() {
    info "Setting up virtual environment..."

    if [ ! -d "$VENV" ]; then
        python3 -m venv "$VENV"
        success "Created venv at $VENV"
    else
        info "Venv already exists — upgrading packages."
    fi

    "$PIP" install --upgrade pip --quiet

    info "Installing dependencies from requirements.txt..."
    "$PIP" install -r "$ROOT/requirements.txt"

    # Try GPU-capable onnxruntime; fall back to CPU silently
    "$PIP" install onnxruntime-gpu 2>/dev/null || \
        "$PIP" install onnxruntime 2>/dev/null || true

    success "Installation complete."
    echo ""
    cmd_check
}

cmd_ui() {
    ensure_venv
    info "Starting Streamlit UI → http://localhost:8501"
    cd "$ROOT"
    "$VENV/bin/streamlit" run app.py \
        --server.headless false \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false \
        "$@"
}

cmd_test() {
    ensure_venv
    info "Running unit tests..."
    cd "$ROOT"
    "$PYTHON" -m pytest tests/ -v --tb=short "$@"
}

cmd_export() {
    ensure_venv
    info "Exporting YOLOv8 model to ONNX..."
    cd "$ROOT"

    # Read model path from config
    MODEL=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('config/pipeline.yaml'))
print(cfg.get('yolo', {}).get('model_path', 'models/yolov8n.pt'))
" 2>/dev/null || echo "models/yolov8n.pt")

    OUTPUT="${MODEL%.pt}_ground.onnx"
    OUTPUT="${OUTPUT/models\//models/}"

    info "Source model : $MODEL"
    info "ONNX output  : $OUTPUT"

    "$PYTHON" training/export_yolov8.py \
        --model "$MODEL" \
        --output "$OUTPUT" \
        "$@"

    success "Export done: $OUTPUT"
}

cmd_finetune() {
    ensure_venv
    info "Starting fine-tuning..."
    cd "$ROOT"
    "$PYTHON" training/finetune_yolov8.py "$@"
}

cmd_check() {
    info "Environment check"
    echo ""

    # Python version
    PY_VER=$("$PYTHON" --version 2>&1 || echo "not found")
    echo "  Python      : $PY_VER"

    # CUDA
    CUDA=$("$PYTHON" -c "import torch; print('CUDA', torch.version.cuda, '— device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>/dev/null || echo "torch not installed")
    echo "  PyTorch     : $CUDA"

    # ultralytics
    UL=$("$PYTHON" -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null || echo "not installed")
    echo "  ultralytics : $UL"

    # onnxruntime
    ORT=$("$PYTHON" -c "import onnxruntime as ort; print(ort.__version__, '—', ort.get_device())" 2>/dev/null || echo "not installed")
    echo "  onnxruntime : $ORT"

    # streamlit
    ST=$("$PYTHON" -c "import streamlit; print(streamlit.__version__)" 2>/dev/null || echo "not installed")
    echo "  streamlit   : $ST"

    # Model file
    MODEL_PATH=$(python3 -c "
import yaml
try:
    cfg = yaml.safe_load(open('$ROOT/config/pipeline.yaml'))
    print(cfg.get('yolo', {}).get('model_path', 'models/yolov8n.pt'))
except Exception:
    print('models/yolov8n.pt')
" 2>/dev/null || echo "models/yolov8n.pt")

    if [ -f "$ROOT/$MODEL_PATH" ]; then
        SIZE=$(du -h "$ROOT/$MODEL_PATH" | cut -f1)
        echo "  YOLOv8 model: $MODEL_PATH ($SIZE) ✓"
    else
        echo "  YOLOv8 model: $MODEL_PATH — NOT FOUND (will auto-download on first run)"
    fi

    echo ""
}

# ── dispatch ──────────────────────────────────────────────────────────────────

CMD="${1:-ui}"
shift 2>/dev/null || true

# Strip leading "--" separator used for passthrough args
if [ "$1" = "--" ]; then shift; fi

case "$CMD" in
    install)   cmd_install "$@" ;;
    ui|app)    cmd_ui "$@" ;;
    test)      cmd_test "$@" ;;
    export)    cmd_export "$@" ;;
    finetune)  cmd_finetune "$@" ;;
    check)     ensure_venv; cmd_check ;;
    help|--help|-h)
        echo ""
        echo "  Ground Target Tracker — run script"
        echo ""
        echo "  Usage: ./run.sh [command] [args]"
        echo ""
        echo "  Commands:"
        echo "    install              Create venv + install all dependencies"
        echo "    ui  (default)        Launch Streamlit UI"
        echo "    test                 Run unit tests"
        echo "    export               Export YOLOv8 .pt → .onnx"
        echo "    finetune -- [args]   Fine-tune YOLOv8 on custom dataset"
        echo "    check                Print environment info"
        echo "    help                 Show this message"
        echo ""
        ;;
    *)
        error "Unknown command: '$CMD'. Run './run.sh help' for usage."
        ;;
esac
