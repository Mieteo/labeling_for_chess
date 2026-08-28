"""Shared unicode-path-safe image reading.

cv2.imread mishandles non-ASCII paths on Windows, so every place in this
tool that needs raw pixel data for classical CV or ONNX inference goes
through this instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
