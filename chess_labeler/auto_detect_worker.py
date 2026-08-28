"""Background QThread worker for running ONNX auto-detect across every
not-yet-labeled image in the open directory -- see
yeu_cau_tu_app_ky_nhan.md section 3.4, item 6: batch auto-detect must not
block the UI and must be cancellable.

Only this worker thread ever calls `AutoDetector.detect()` while a batch is
running -- main_window.py disables the single-image auto-detect button for
the duration to keep session usage single-threaded and easy to reason about.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from . import auto_detect, suggestions
from .imaging import read_image_bgr


class AutoDetectBatchWorker(QThread):
    progress = Signal(int, int, str)  # done, total, current filename
    finishedBatch = Signal(int, int)  # images_with_suggestions, images_skipped

    def __init__(
        self,
        detector: auto_detect.AutoDetector,
        image_paths: list[Path],
        conf_threshold: float,
        iou_threshold: float,
        parent=None,
    ):
        super().__init__(parent)
        self._detector = detector
        self._image_paths = list(image_paths)
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:  # noqa: N802 (Qt override)
        total = len(self._image_paths)
        processed = 0
        skipped = 0
        for i, path in enumerate(self._image_paths):
            if self._cancel_requested:
                break
            self.progress.emit(i, total, path.name)
            try:
                image_bgr = read_image_bgr(path)
                if image_bgr is None:
                    skipped += 1
                    continue
                detections = self._detector.detect(image_bgr, self._conf_threshold, self._iou_threshold)
            except Exception:
                skipped += 1
                continue
            if not detections:
                continue
            img_h, img_w = image_bgr.shape[:2]
            pending = [
                suggestions.PendingBox(
                    class_name=d.class_name,
                    xc=d.cx / img_w,
                    yc=d.cy / img_h,
                    w=d.w / img_w,
                    h=d.h / img_h,
                    score=d.score,
                )
                for d in detections
            ]
            suggestions.save_suggestions(
                path, pending, self._detector.model_path, self._conf_threshold, self._iou_threshold
            )
            processed += 1
        self.progress.emit(total, total, "")
        self.finishedBatch.emit(processed, skipped)
