import cv2
import numpy as np

from chess_labeler import circle_detect
from tests.conftest import CIRCLE_TEST_CENTERS_RADII


def _read_bgr(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _closest_match(cx, cy, circles) -> float:
    return min(((c.cx - cx) ** 2 + (c.cy - cy) ** 2) ** 0.5 for c in circles)


def test_auto_scan_finds_all_circles(circle_test_image):
    img = _read_bgr(circle_test_image)
    circles = circle_detect.auto_scan(img, min_fraction=0.02, max_fraction=0.06)
    assert len(circles) >= len(CIRCLE_TEST_CENTERS_RADII)
    for cx, cy, r in CIRCLE_TEST_CENTERS_RADII:
        assert _closest_match(cx, cy, circles) < 10.0


def test_radius_guided_finds_circles_within_tolerance(circle_test_image):
    img = _read_bgr(circle_test_image)
    circles = circle_detect.radius_guided(img, reference_radius_px=30.0, tolerance_pct=15.0)
    assert len(circles) >= len(CIRCLE_TEST_CENTERS_RADII)
    for cx, cy, r in CIRCLE_TEST_CENTERS_RADII:
        assert _closest_match(cx, cy, circles) < 10.0


def test_circle_to_pixel_box_is_centered_square():
    circle = circle_detect.DetectedCircle(cx=100.0, cy=200.0, r=25.0)
    left, top, w, h = circle_detect.circle_to_pixel_box(circle)
    assert w == h == 50.0
    assert left == 75.0
    assert top == 175.0


def test_radius_guided_out_of_range_finds_nothing(circle_test_image):
    img = _read_bgr(circle_test_image)
    # True radius is 30px; searching around a wildly different radius with a
    # tight tolerance should not match the real circles.
    circles = circle_detect.radius_guided(img, reference_radius_px=5.0, tolerance_pct=10.0)
    for cx, cy, r in CIRCLE_TEST_CENTERS_RADII:
        matches_close = any(((c.cx - cx) ** 2 + (c.cy - cy) ** 2) ** 0.5 < 10.0 for c in circles)
        assert not matches_close
