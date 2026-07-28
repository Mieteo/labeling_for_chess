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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

CLASSES_FILENAME = "classes.txt"
SESSION_FILENAME = ".labeling_session.json"

# Circle detection defaults (section 4).
AUTO_SCAN_MIN_RADIUS_FRACTION = 0.015  # 1.5% of image width
AUTO_SCAN_MAX_RADIUS_FRACTION = 0.06  # 6% of image width
DEFAULT_RADIUS_TOLERANCE_PCT = 15.0  # +/-15%
