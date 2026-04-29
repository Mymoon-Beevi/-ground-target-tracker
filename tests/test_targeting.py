"""Unit tests for ScreenCenterTargeter — error signal and angle computation."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.targeting import ScreenCenterTargeter, targeter_from_config


# ── helpers ────────────────────────────────────────────────────────────────────

def _track(x1, y1, x2, y2, conf=0.8, cls=2, age=0):
    return {"bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(conf), "cls": int(cls), "age": int(age)}


def _centered_targeter(w=1280, h=720):
    return ScreenCenterTargeter(
        frame_w=w, frame_h=h,
        smoothing_alpha=1.0,  # no smoothing — raw values
        camera_fx=w * (1066.67 / 1280),
        camera_fy=h * (1066.67 / 720),
        camera_cx=w / 2.0,
        camera_cy=h / 2.0,
    )


# ── tests ──────────────────────────────────────────────────────────────────────

class TestErrorSignal:
    def test_centered_target_zero_error(self):
        t = _centered_targeter(1280, 720)
        # bbox exactly at center
        cx, cy = 640, 360
        tracks = {1: _track(cx - 50, cy - 50, cx + 50, cy + 50)}
        state = t.update(tracks)
        assert state is not None
        assert state.error_x == pytest.approx(0.0, abs=1e-3)
        assert state.error_y == pytest.approx(0.0, abs=1e-3)

    def test_top_left_target_negative_error(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(0, 0, 100, 100)}
        state = t.update(tracks)
        assert state.error_x < 0
        assert state.error_y < 0

    def test_bottom_right_target_positive_error(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(1180, 620, 1280, 720)}
        state = t.update(tracks)
        assert state.error_x > 0
        assert state.error_y > 0

    def test_error_range(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(0, 0, 1280, 720)}  # full-frame bbox
        state = t.update(tracks)
        assert -1.0 <= state.error_x <= 1.0
        assert -1.0 <= state.error_y <= 1.0


class TestAngleComputation:
    def test_center_is_zero_angle(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(590, 310, 690, 410)}
        state = t.update(tracks)
        assert state.angle_pan == pytest.approx(0.0, abs=0.5)
        assert state.angle_tilt == pytest.approx(0.0, abs=0.5)

    def test_pan_positive_for_right_of_center(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(900, 310, 1000, 410)}
        state = t.update(tracks)
        assert state.angle_pan > 0

    def test_tilt_positive_for_below_center(self):
        t = _centered_targeter(1280, 720)
        tracks = {1: _track(590, 500, 690, 600)}
        state = t.update(tracks)
        assert state.angle_tilt > 0


class TestSelectionMode:
    def test_nearest_center(self):
        t = _centered_targeter()
        tracks = {
            1: _track(0, 0, 100, 100),      # far from center
            2: _track(600, 330, 680, 390),  # near center
        }
        state = t.update(tracks)
        assert state.track_id == 2

    def test_largest(self):
        t = _centered_targeter()
        t.selection_mode = "largest"
        tracks = {
            1: _track(0, 0, 50, 50),       # small
            2: _track(0, 0, 400, 300),     # large
        }
        state = t.update(tracks)
        assert state.track_id == 2

    def test_highest_conf(self):
        t = _centered_targeter()
        t.selection_mode = "highest_conf"
        tracks = {
            1: _track(0, 0, 100, 100, conf=0.4),
            2: _track(0, 0, 100, 100, conf=0.9),
        }
        state = t.update(tracks)
        assert state.track_id == 2

    def test_manual_lock(self):
        t = _centered_targeter()
        t.lock_on(3)
        tracks = {
            1: _track(600, 330, 680, 390, conf=0.99),  # highest conf, nearest center
            3: _track(0, 0, 100, 100, conf=0.5),
        }
        state = t.update(tracks)
        assert state.track_id == 3

    def test_manual_fallback_on_lost(self):
        t = _centered_targeter()
        t.lock_on(99)  # non-existent track
        tracks = {2: _track(600, 330, 680, 390)}
        state = t.update(tracks)
        # Should fall back and return some target
        assert state is not None

    def test_empty_tracks_returns_none(self):
        t = _centered_targeter()
        assert t.update({}) is None


class TestSmoothing:
    def test_ema_with_alpha_1_is_raw(self):
        t = _centered_targeter()
        t.smoothing_alpha = 1.0
        tracks = {1: _track(200, 100, 300, 200)}
        state = t.update(tracks)
        expected_cx = 250.0
        expected_err = (expected_cx - 640) / 640
        assert state.error_x == pytest.approx(expected_err, abs=1e-4)

    def test_ema_smoothing_reduces_jump(self):
        t = _centered_targeter()
        t.smoothing_alpha = 0.1
        tracks = {1: _track(590, 310, 690, 410)}
        t.update(tracks)
        # Big jump
        tracks = {1: _track(1180, 620, 1280, 720)}
        state = t.update(tracks)
        # With alpha=0.1, smoothed value should be much less than raw
        raw_err_x = (1230 - 640) / 640
        assert abs(state.error_x) < abs(raw_err_x)


class TestConfigFactory:
    def test_from_config(self):
        config = {
            "targeting": {
                "selection_mode": "largest",
                "smoothing_alpha": 0.5,
                "camera_fx": 500.0,
                "camera_fy": 500.0,
                "camera_cx": 320.0,
                "camera_cy": 240.0,
            }
        }
        t = targeter_from_config(config, 640, 480)
        assert t.selection_mode == "largest"
        assert t.smoothing_alpha == pytest.approx(0.5)
        assert t.frame_w == 640
        assert t.frame_h == 480
