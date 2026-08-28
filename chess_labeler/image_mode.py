"""Per-directory "image mode" -- physical (real photographed board) vs
digital (screenshot of a Xiangqi program) -- see
yeu_cau_tu_app_ky_nhan.md section 2.

This is a presentation-only concept: it decides whether the circle-detect
assist (Physical) or the ONNX auto-detect assist (Digital) is shown in the
UI. It has zero effect on classes.txt, YOLO box format, or the metadata
sidecar schema -- both image types share the exact same 15-class label
contract.
"""

from __future__ import annotations

from pathlib import Path

PHYSICAL = "physical"
DIGITAL = "digital"
MODES = (PHYSICAL, DIGITAL)

_DIGITAL_HINT = "digital"


def infer_mode_from_dirname(directory: Path | str) -> str:
    """Simplest possible heuristic, per spec: the directory name contains
    "digital" (case-insensitive) -> digital mode, otherwise physical.
    A user can always override this via the mode toggle in the UI."""
    name = Path(directory).name.lower()
    return DIGITAL if _DIGITAL_HINT in name else PHYSICAL
