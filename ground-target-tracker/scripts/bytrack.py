"""ByteTrack wrapper — unified tracker interface over ultralytics BYTETracker with IoU fallback."""

from __future__ import annotations

import numpy as np


class _SimpleTracker:
    """Minimal IoU tracker used as fallback when ultralytics is unavailable."""

    def __init__(self):
        self.tracks: dict = {}
        self.next_id = 1

    def update(self, detections: list) -> dict:
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    "bbox": list(det[:4]), "conf": float(det[4]),
                    "cls": int(det[5]), "age": 0,
                }
                self.next_id += 1
            return self.tracks

        track_ids = list(self.tracks.keys())
        matched_t, matched_d = set(), set()

        for tid in track_ids:
            best_iou, best_j = 0.3, -1
            for j, det in enumerate(detections):
                if j in matched_d:
                    continue
                iou = self._iou(self.tracks[tid]["bbox"], det[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                self.tracks[tid].update({"bbox": list(detections[best_j][:4]),
                                          "conf": float(detections[best_j][4]),
                                          "cls": int(detections[best_j][5]), "age": 0})
                matched_t.add(tid); matched_d.add(best_j)

        for tid in track_ids:
            if tid not in matched_t:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > 15:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_d:
                self.tracks[self.next_id] = {
                    "bbox": list(det[:4]), "conf": float(det[4]),
                    "cls": int(det[5]), "age": 0,
                }
                self.next_id += 1

        return self.tracks

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0


class _DetResults:
    """
    Shim that makes a raw [N, 6] numpy array look like a ultralytics Results.boxes
    object so BYTETracker.update() can access .conf, .xyxy, .cls attributes.
    The new ultralytics API (>=8.1) passes this object directly instead of a
    bare ndarray.
    """

    def __init__(self, dets: np.ndarray):
        # dets: [N, 6] — x1 y1 x2 y2 conf cls
        self._d = dets

    @property
    def conf(self) -> np.ndarray:
        return self._d[:, 4] if len(self._d) else np.empty(0, np.float32)

    @property
    def xyxy(self) -> np.ndarray:
        return self._d[:, :4] if len(self._d) else np.empty((0, 4), np.float32)

    @property
    def cls(self) -> np.ndarray:
        return self._d[:, 5] if len(self._d) else np.empty(0, np.float32)

    def __getitem__(self, idx) -> "_DetResults":
        return _DetResults(self._d[idx])

    def __len__(self) -> int:
        return len(self._d)

    # Some ultralytics versions access .boxes on the results object
    @property
    def boxes(self):
        return self


class ByTracker:
    """
    Wraps ultralytics BYTETracker with the same dict-based output interface
    as SimpleTracker: {track_id: {"bbox": [x1,y1,x2,y2], "conf": float, "cls": int, "age": int}}

    Falls back to SimpleTracker if ultralytics is unavailable.
    """

    def __init__(self, fps: float = 30.0, track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1, new_track_thresh: float = 0.5,
                 track_buffer: int = 30, match_thresh: float = 0.8):
        self._fps = fps
        self._tracker = None
        self._cfg = dict(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self._fallback = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Cfg:
                pass

            cfg = _Cfg()
            for k, v in self._cfg.items():
                setattr(cfg, k, v)
            self._tracker = BYTETracker(cfg, frame_rate=int(self._fps))
        except Exception:
            self._fallback = _SimpleTracker()

    def update(self, detections: list[list[float]], frame_shape: tuple[int, int]) -> dict:
        """
        detections: list of [x1, y1, x2, y2, score, class_id]
        frame_shape: (height, width)
        Returns: {track_id: {"bbox", "conf", "cls", "age"}}
        """
        if self._fallback is not None:
            return self._fallback.update(detections)

        if not detections:
            tracks = self._tracker.update(_DetResults(np.empty((0, 6), np.float32)), None)
            return {}

        dets = np.array(detections, dtype=np.float32)  # [N, 6]
        tracks = self._tracker.update(_DetResults(dets), None)

        result: dict = {}
        for t in tracks:
            tid = int(t.track_id)
            x1, y1, x2, y2 = t.tlbr
            result[tid] = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(t.score),
                "cls": int(t.cls) if hasattr(t, "cls") else 0,
                "age": int(t.time_since_update) if hasattr(t, "time_since_update") else 0,
            }

        return result

    def reset(self) -> None:
        self._tracker = None
        self._fallback = None
        self._init_tracker()
