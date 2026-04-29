"""Screen-center targeting engine — selects primary track and computes pan/tilt error."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TargetingState:
    track_id: int
    bbox: list[float]          # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    # Normalized error: -1..+1, 0 = perfectly centered
    error_x: float = 0.0
    error_y: float = 0.0
    # Gimbal angles in degrees (relative to bore-sight)
    angle_pan: float = 0.0
    angle_tilt: float = 0.0
    # Lock quality: 0.0 (lost) .. 1.0 (fresh, high-conf)
    lock_quality: float = 0.0


@dataclass
class ScreenCenterTargeter:
    """
    Selects the primary ground target and computes the normalized screen-center
    error plus estimated gimbal pan/tilt angles.

    selection_mode:
      "nearest_center" — track whose centroid is closest to frame center
      "largest"        — track with the largest bounding-box area
      "highest_conf"   — track with the highest detection confidence
      "manual"         — lock on a specific track_id (set via lock_on())
    """

    frame_w: int = 1280
    frame_h: int = 720
    selection_mode: str = "nearest_center"
    smoothing_alpha: float = 0.3       # EMA weight for new measurement
    camera_fx: float = 1066.67
    camera_fy: float = 1066.67
    camera_cx: float = 640.0
    camera_cy: float = 360.0

    # Internal state
    _manual_id: int = field(default=-1, init=False, repr=False)
    _smoothed_cx: float = field(default=0.0, init=False, repr=False)
    _smoothed_cy: float = field(default=0.0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def lock_on(self, track_id: int) -> None:
        self._manual_id = track_id
        self.selection_mode = "manual"

    def release_lock(self) -> None:
        self._manual_id = -1
        self.selection_mode = "nearest_center"

    def update(self, tracks: dict) -> TargetingState | None:
        """
        tracks: dict of track_id -> {"bbox": [x1,y1,x2,y2], "conf": float, "cls": int, "age": int}
        Returns a TargetingState for the selected primary target, or None if no tracks.
        """
        if not tracks:
            self._initialized = False
            return None

        primary_id = self._select(tracks)
        if primary_id is None:
            return None

        t = tracks[primary_id]
        x1, y1, x2, y2 = t["bbox"]
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0

        # EMA smoothing
        if not self._initialized:
            self._smoothed_cx = raw_cx
            self._smoothed_cy = raw_cy
            self._initialized = True
        else:
            a = self.smoothing_alpha
            self._smoothed_cx = a * raw_cx + (1.0 - a) * self._smoothed_cx
            self._smoothed_cy = a * raw_cy + (1.0 - a) * self._smoothed_cy

        cx, cy = self._smoothed_cx, self._smoothed_cy

        # Normalized error: 0 at center, ±1 at edges
        err_x = (cx - self.frame_w / 2.0) / (self.frame_w / 2.0)
        err_y = (cy - self.frame_h / 2.0) / (self.frame_h / 2.0)

        # Pixel → angle using pinhole model
        angle_pan = math.degrees(math.atan2(cx - self.camera_cx, self.camera_fx))
        angle_tilt = math.degrees(math.atan2(cy - self.camera_cy, self.camera_fy))

        # Lock quality: decays with track age, scales with confidence
        age = t.get("age", 0)
        age_factor = max(0.0, 1.0 - age / 10.0)
        lock_quality = float(t.get("conf", 0.5)) * age_factor

        return TargetingState(
            track_id=primary_id,
            bbox=list(t["bbox"]),
            class_id=int(t.get("cls", 0)),
            confidence=float(t.get("conf", 0.0)),
            error_x=err_x,
            error_y=err_y,
            angle_pan=angle_pan,
            angle_tilt=angle_tilt,
            lock_quality=lock_quality,
        )

    def _select(self, tracks: dict) -> int | None:
        if not tracks:
            return None

        if self.selection_mode == "manual":
            if self._manual_id in tracks:
                return self._manual_id
            # Manual target lost — fall through to nearest_center
            self._manual_id = -1

        half_w = self.frame_w / 2.0
        half_h = self.frame_h / 2.0

        def cx(t):
            return (t["bbox"][0] + t["bbox"][2]) / 2.0

        def cy(t):
            return (t["bbox"][1] + t["bbox"][3]) / 2.0

        if self.selection_mode == "nearest_center":
            return min(
                tracks,
                key=lambda tid: (cx(tracks[tid]) - half_w) ** 2
                + (cy(tracks[tid]) - half_h) ** 2,
            )
        elif self.selection_mode == "largest":
            def area(t):
                b = t["bbox"]
                return (b[2] - b[0]) * (b[3] - b[1])
            return max(tracks, key=lambda tid: area(tracks[tid]))
        elif self.selection_mode == "highest_conf":
            return max(tracks, key=lambda tid: tracks[tid].get("conf", 0.0))
        else:
            return next(iter(tracks))


def targeter_from_config(config: dict, frame_w: int, frame_h: int) -> ScreenCenterTargeter:
    tgt = config.get("targeting", {})
    return ScreenCenterTargeter(
        frame_w=frame_w,
        frame_h=frame_h,
        selection_mode=tgt.get("selection_mode", "nearest_center"),
        smoothing_alpha=tgt.get("smoothing_alpha", 0.3),
        camera_fx=tgt.get("camera_fx", 1066.67),
        camera_fy=tgt.get("camera_fy", 1066.67),
        camera_cx=tgt.get("camera_cx", frame_w / 2.0),
        camera_cy=tgt.get("camera_cy", frame_h / 2.0),
    )
