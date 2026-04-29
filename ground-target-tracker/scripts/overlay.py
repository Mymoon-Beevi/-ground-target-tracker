"""Consolidated drawing functions for detection boxes, trajectories, HUD, and targeting reticle."""

from __future__ import annotations

import cv2
import numpy as np

from targeting import TargetingState

# Per-class colors (BGR). Cycles if more classes than entries.
_CLASS_COLORS = [
    (0, 255, 0),      # person — green
    (0, 200, 255),    # bicycle — yellow
    (0, 100, 255),    # car — orange
    (255, 100, 0),    # motorcycle — blue-ish
    (200, 0, 200),    # bus — magenta
    (100, 100, 255),  # truck — pink
]

_RETICLE_COLOR = (0, 255, 255)   # cyan
_PRIMARY_COLOR = (0, 255, 255)   # cyan
_HUD_BG = (0, 0, 0)


def _class_color(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLORS[class_id % len(_CLASS_COLORS)]


def draw_detections(
    frame: np.ndarray,
    tracks: dict,
    class_names: list[str] | None = None,
    primary_id: int = -1,
) -> np.ndarray:
    """Draw bounding boxes, class labels, confidence, and track IDs."""
    for tid, t in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        cls = int(t.get("cls", 0))
        conf = float(t.get("conf", 0.0))

        color = _PRIMARY_COLOR if tid == primary_id else _class_color(cls)
        thickness = 2 if tid == primary_id else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        cls_name = (class_names[cls] if class_names and cls < len(class_names)
                    else str(cls))
        label = f"T{tid:02d} {cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), _HUD_BG, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def draw_trajectories(
    frame: np.ndarray,
    track_history: dict[int, list],
    primary_id: int = -1,
) -> np.ndarray:
    """Draw fade-out trajectory lines for each track."""
    for tid, history in track_history.items():
        if len(history) < 2:
            continue
        pts = [(int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)) for b in history]
        is_primary = tid == primary_id
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            if is_primary:
                color = (0, int(255 * alpha), int(255 * alpha))
            else:
                color = (int(200 * alpha), int(200 * alpha), 50)
            cv2.line(frame, pts[i - 1], pts[i], color, 2 if is_primary else 1)

    return frame


def draw_targeting_reticle(
    frame: np.ndarray,
    state: TargetingState,
) -> np.ndarray:
    """Draw center crosshair, line to primary target, and error vector."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    arm = 20

    # Center crosshair
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), _RETICLE_COLOR, 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), _RETICLE_COLOR, 1)
    cv2.circle(frame, (cx, cy), arm + 4, _RETICLE_COLOR, 1)

    # Target centroid
    bx = int((state.bbox[0] + state.bbox[2]) / 2)
    by = int((state.bbox[1] + state.bbox[3]) / 2)

    # Line from center to target
    cv2.line(frame, (cx, cy), (bx, by), _RETICLE_COLOR, 1, cv2.LINE_AA)
    cv2.circle(frame, (bx, by), 5, _RETICLE_COLOR, -1)

    # Lock quality ring: green → yellow → red
    q = state.lock_quality
    ring_color = (
        int((1.0 - q) * 255),   # B
        int(q * 255),            # G
        0,
    )
    cv2.circle(frame, (bx, by), 18, ring_color, 2)

    # Error text near target
    err_label = f"e({state.error_x:+.2f}, {state.error_y:+.2f})"
    cv2.putText(frame, err_label, (bx + 22, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _RETICLE_COLOR, 1)

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    track_count: int,
    state: TargetingState | None = None,
) -> np.ndarray:
    """Draw top-left HUD bar with FPS, track count, and targeting info."""
    bar_h = 28
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), _HUD_BG, -1)

    parts = [f"FPS {fps:.0f}", f"Tracks {track_count}"]
    if state is not None:
        parts += [
            f"Lock T{state.track_id:02d}",
            f"Pan {state.angle_pan:+.1f}°",
            f"Tilt {state.angle_tilt:+.1f}°",
            f"Q {state.lock_quality:.0%}",
        ]

    text = "  |  ".join(parts)
    cv2.putText(frame, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame
