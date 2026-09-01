"""Fixed constants shared across the labeling tool.

The class list and its order are a hard contract with the existing labelImg
data (see labeling_tool_requirements.md section 1.2) and with
tool/unify_scanner_labels.py in the main app repo. Do not reorder, rename, or
insert classes.
"""

# 15 fixed classes, order = class index used in every .txt label file.
DEFAULT_CLASSES: list[str] = [
    "red_king",
    "red_advisor",
    "red_elephant",
    "red_horse",
    "red_cannon",
    "red_rook",
    "red_pawn",
    "black_king",
    "black_advisor",
    "black_elephant",
    "black_horse",
    "black_cannon",
    "black_rook",
    "black_pawn",
    "hand",
]

HAND_CLASS_NAME = "hand"

# Digital directories get one extra 16th class appended -- a loose box around
# the whole 9x10 grid, used by the app's grid-placement fallback. See
# yeu_cau_tu_app_ky_nhan.md section 1. Physical directories keep the 15
# classes above, untouched.
DEFAULT_CLASSES_DIGITAL: list[str] = DEFAULT_CLASSES + ["board_region"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

CLASSES_FILENAME = "classes.txt"
SESSION_FILENAME = ".labeling_session.json"

# Circle detection defaults (section 4).
AUTO_SCAN_MIN_RADIUS_FRACTION = 0.015  # 1.5% of image width
AUTO_SCAN_MAX_RADIUS_FRACTION = 0.06  # 6% of image width
DEFAULT_RADIUS_TOLERANCE_PCT = 15.0  # +/-15%

# Digital ONNX auto-detect defaults -- see
# yeu_cau_tu_app_ky_nhan.md section 3.1 (exact values the source app uses).
AUTO_DETECT_INPUT_SIZE = 640
AUTO_DETECT_LETTERBOX_COLOR = 114
DEFAULT_AUTO_DETECT_CONF_THRESHOLD = 0.25
DEFAULT_AUTO_DETECT_IOU_THRESHOLD = 0.45
