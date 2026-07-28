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
