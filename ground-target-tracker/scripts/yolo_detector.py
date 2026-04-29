"""YOLOv8 detector + tracker — uses model.track() (stable public ultralytics API)."""

from __future__ import annotations

import numpy as np

# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Default ground-target classes: person, bicycle, car, motorcycle, bus, truck
DEFAULT_GROUND_CLASSES = [0, 1, 2, 3, 5, 7]


class YoloDetector:
    """
    Wraps ultralytics YOLO with two modes:

    - detect(frame)            → list of [x1,y1,x2,y2,score,class_id]
    - detect_and_track(frame)  → dict {track_id: {"bbox","conf","cls","age"}}

    detect_and_track() uses model.track() — the stable public ultralytics API
    that handles ByteTrack internally, avoiding direct BYTETracker instantiation
    whose internal API changes between releases.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.30,
        nms_iou_threshold: float = 0.45,
        input_size: int = 640,
        classes: list[int] | None = None,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = nms_iou_threshold
        self.input_size = input_size
        self.classes = classes if classes is not None else DEFAULT_GROUND_CLASSES
        self.device = device

        # Warm up
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            classes=self.classes or None,
            device=self.device,
            verbose=False,
        )

    # ── detection only ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """Returns list of [x1, y1, x2, y2, score, class_id]."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            classes=self.classes or None,
            device=self.device,
            verbose=False,
        )
        return self._parse_boxes(results)

    # ── detection + tracking ──────────────────────────────────────────────────

    def detect_and_track(self, frame: np.ndarray) -> dict:
        """
        Run detection + ByteTrack via model.track() (stable ultralytics API).
        Returns {track_id: {"bbox": [x1,y1,x2,y2], "conf": float, "cls": int, "age": int}}.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            classes=self.classes or None,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,         # keeps track state across frames
            verbose=False,
        )

        tracks: dict = {}
        if not results or results[0].boxes is None:
            return tracks

        boxes = results[0].boxes
        if boxes.id is None:        # no tracks assigned yet
            return tracks

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            tracks[int(track_ids[i])] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(confs[i]),
                "cls": int(cls_ids[i]),
                "age": 0,
            }

        return tracks

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_boxes(self, results) -> list[list[float]]:
        detections: list[list[float]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            detections.append([float(x1), float(y1), float(x2), float(y2),
                                float(confs[i]), int(cls_ids[i])])
        return detections

    @property
    def class_names(self) -> list[str]:
        names = self.model.names
        if isinstance(names, dict):
            return [names.get(i, str(i)) for i in range(max(names.keys()) + 1)]
        return list(names)


def load_detector_from_config(config: dict) -> YoloDetector:
    yolo_cfg = config.get("yolo", {})
    return YoloDetector(
        model_path=yolo_cfg.get("model_path", "models/yolov8n.pt"),
        conf_threshold=yolo_cfg.get("confidence_threshold", 0.30),
        nms_iou_threshold=yolo_cfg.get("nms_iou_threshold", 0.45),
        input_size=yolo_cfg.get("input_size", 640),
        classes=yolo_cfg.get("classes") or DEFAULT_GROUND_CLASSES,
        device=yolo_cfg.get("device", "cuda"),
    )
