"""Unit tests for YoloDetector coordinate de-letterboxing and interface contract."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.yolo_detector import YoloDetector, COCO_NAMES, DEFAULT_GROUND_CLASSES


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_mock_result(xyxy, confs, cls_ids):
    """Build a minimal fake ultralytics result object."""
    import types
    import torch

    boxes = types.SimpleNamespace(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        conf=torch.tensor(confs, dtype=torch.float32),
        cls=torch.tensor(cls_ids, dtype=torch.float32),
    )
    result = types.SimpleNamespace(boxes=boxes)
    return [result]


# ── tests ──────────────────────────────────────────────────────────────────────

class TestCocoNames:
    def test_length(self):
        assert len(COCO_NAMES) == 80

    def test_known_entries(self):
        assert COCO_NAMES[0] == "person"
        assert COCO_NAMES[2] == "car"
        assert COCO_NAMES[7] == "truck"


class TestDefaultClasses:
    def test_contains_ground_targets(self):
        # person, bicycle, car, motorcycle, bus, truck
        for cls in [0, 1, 2, 3, 5, 7]:
            assert cls in DEFAULT_GROUND_CLASSES


class TestYoloDetectorOutputFormat:
    """Test that detect() returns the correct [x1,y1,x2,y2,score,class_id] format
    by monkey-patching the model.predict call."""

    def test_single_detection(self, monkeypatch):
        import torch

        xyxy = [[100.0, 50.0, 300.0, 200.0]]
        confs = [0.85]
        cls_ids = [2]  # car

        det = YoloDetector.__new__(YoloDetector)
        det.conf = 0.3
        det.iou = 0.45
        det.input_size = 640
        det.classes = None
        det.device = "cpu"

        mock_result = _make_mock_result(xyxy, confs, cls_ids)
        det.model = type("M", (), {"predict": staticmethod(lambda *a, **kw: mock_result),
                                    "names": {i: n for i, n in enumerate(COCO_NAMES)}})()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = det.detect(frame)

        assert len(dets) == 1
        d = dets[0]
        assert d[0] == pytest.approx(100.0)
        assert d[1] == pytest.approx(50.0)
        assert d[2] == pytest.approx(300.0)
        assert d[3] == pytest.approx(200.0)
        assert d[4] == pytest.approx(0.85)
        assert d[5] == 2

    def test_tiny_box_filtered(self, monkeypatch):
        xyxy = [[10.0, 10.0, 12.0, 12.0]]  # 2x2 px — should be filtered
        confs = [0.95]
        cls_ids = [0]

        det = YoloDetector.__new__(YoloDetector)
        det.conf = 0.3
        det.iou = 0.45
        det.input_size = 640
        det.classes = None
        det.device = "cpu"

        mock_result = _make_mock_result(xyxy, confs, cls_ids)
        det.model = type("M", (), {"predict": staticmethod(lambda *a, **kw: mock_result),
                                    "names": {i: n for i, n in enumerate(COCO_NAMES)}})()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = det.detect(frame)
        assert len(dets) == 0

    def test_empty_frame(self, monkeypatch):
        import types
        det = YoloDetector.__new__(YoloDetector)
        det.conf = 0.3
        det.iou = 0.45
        det.input_size = 640
        det.classes = None
        det.device = "cpu"

        empty_result = [types.SimpleNamespace(boxes=types.SimpleNamespace(
            xyxy=__import__("torch").empty((0, 4)),
            conf=__import__("torch").empty(0),
            cls=__import__("torch").empty(0),
        ))]
        det.model = type("M", (), {"predict": staticmethod(lambda *a, **kw: empty_result),
                                    "names": {i: n for i, n in enumerate(COCO_NAMES)}})()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = det.detect(frame)
        assert dets == []
