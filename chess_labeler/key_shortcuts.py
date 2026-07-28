"""Single-key shortcuts for fast class assignment.

Pure logic, no Qt dependency (keeps it independently testable). See
labeling_tool_requirements.md section 3.

Each xiangqi role is one letter, typed directly -- no Ctrl, no multi-step
chord:

    p = pawn     c = cannon    r = rook   h = horse
    e = elephant a = advisor   k = king

The letter's case selects color: lowercase = black, UPPERCASE (typed with
Caps Lock or Shift) = red. `hand` has no color and keeps its own shortcut,
Ctrl+H -- none of the 7 letters above is free for it (H is already
"horse").
"""

from __future__ import annotations

ROLE_BY_LETTER = {
    "P": "pawn",
    "C": "cannon",
    "R": "rook",
    "H": "horse",
    "E": "elephant",
    "A": "advisor",
    "K": "king",
}

HAND_CLASS = "hand"


def resolve_piece_class(letter: str, is_red: bool) -> str | None:
    """`letter`: a single role letter, either case, e.g. "p"/"P".

    Returns the target class name (e.g. "red_cannon"), or None if `letter`
    isn't one of the 7 role letters above.
    """
    role = ROLE_BY_LETTER.get(letter.upper())
    if role is None:
        return None
    return f"{'red' if is_red else 'black'}_{role}"


_LETTER_BY_ROLE = {role: letter.lower() for letter, role in ROLE_BY_LETTER.items()}


def display_label_for_class(class_name: str | None) -> str:
    """Short glyph drawn inside a box to show its assigned class: the same
    lowercase role letter used for the keyboard shortcut (color alone
    conveys red/black, so the glyph itself never needs to be uppercase),
    or the literal word "hand" for the colorless occlusion class. Empty
    string for an unassigned or unrecognized class name."""
    if not class_name:
        return ""
    if class_name == HAND_CLASS:
        return HAND_CLASS
    _, _, role = class_name.partition("_")
    return _LETTER_BY_ROLE.get(role, "")


# Display order for the "boxes in this image" panel: black pieces first (in
# this fixed role order), then red, then the colorless `hand` class, with
# unassigned boxes sorted separately (below all of these -- see panels.py).
CLASS_DISPLAY_ORDER: list[str] = [
    "black_pawn", "black_rook", "black_cannon", "black_horse",
    "black_elephant", "black_advisor", "black_king",
    "red_pawn", "red_rook", "red_cannon", "red_horse",
    "red_elephant", "red_advisor", "red_king",
    HAND_CLASS,
]
