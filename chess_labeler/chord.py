"""Sequential 2-step Ctrl-chord state machine for fast class assignment.

Pure logic, no Qt dependency (keeps it independently testable). See
labeling_tool_requirements.md section 3. Mnemonic table:

    Ctrl,R,K -> red_king       Ctrl,B,K -> black_king
    Ctrl,R,A -> red_advisor    Ctrl,B,A -> black_advisor
    Ctrl,R,E -> red_elephant   Ctrl,B,E -> black_elephant
    Ctrl,R,H -> red_horse      Ctrl,B,H -> black_horse
    Ctrl,R,C -> red_cannon     Ctrl,B,C -> black_cannon
    Ctrl,R,R -> red_rook       Ctrl,B,R -> black_rook
    Ctrl,R,P -> red_pawn       Ctrl,B,P -> black_pawn
    Ctrl,H   -> hand (fires immediately, no step-2 wait)

Ctrl+H is ambiguous on its own: it means "hand" when idle, but means "horse"
when it's the step-2 key right after Ctrl+R / Ctrl+B. The disambiguation is
purely a function of whether a chord is currently pending -- exactly the
state this class tracks.
"""

from __future__ import annotations

import dataclasses
import time
from enum import Enum, auto

COLOR_KEYS = {"R": "red", "B": "black"}
ROLE_KEYS = {
    "K": "king",
    "A": "advisor",
    "E": "elephant",
    "H": "horse",
    "C": "cannon",
    "R": "rook",
    "P": "pawn",
}
HAND_KEY = "H"

DEFAULT_TIMEOUT_SECONDS = 2.0


class ChordOutcome(Enum):
    IGNORED = auto()  # key irrelevant to any chord; state unchanged
    PENDING = auto()  # step 1 done, waiting on step 2 -- UI should show indicator
    CANCELLED = auto()  # explicit Esc, timeout, or unrecognized step-2 key
    ASSIGNED = auto()  # class_name is ready to be applied to the selected box


@dataclasses.dataclass
class ChordEvent:
    outcome: ChordOutcome
    class_name: str | None = None
    pending_label: str | None = None  # e.g. "R…" for a UI indicator


class ChordShortcutManager:
    """Feed every Ctrl-held letter key press via `key_pressed`, Escape via
    `escape_pressed`, and poll `check_timeout` periodically (e.g. from a UI
    timer) so a pending chord doesn't hang forever if the user gets
    distracted mid-sequence.
    """

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self._timeout_seconds = timeout_seconds
        self._pending_color: str | None = None
        self._pending_since: float | None = None

    @property
    def is_pending(self) -> bool:
        return self._pending_color is not None

    @property
    def pending_indicator(self) -> str | None:
        if self._pending_color is None:
            return None
        return "R…" if self._pending_color == "red" else "B…"

    def reset(self) -> None:
        self._pending_color = None
        self._pending_since = None

    def escape_pressed(self) -> ChordEvent:
        was_pending = self.is_pending
        self.reset()
        return ChordEvent(
            ChordOutcome.CANCELLED if was_pending else ChordOutcome.IGNORED
        )

    def check_timeout(self, now: float | None = None) -> ChordEvent:
        if not self.is_pending:
            return ChordEvent(ChordOutcome.IGNORED)
        now = time.monotonic() if now is None else now
        if now - self._pending_since > self._timeout_seconds:
            self.reset()
            return ChordEvent(ChordOutcome.CANCELLED)
        return ChordEvent(ChordOutcome.IGNORED)

    def key_pressed(
        self, key: str, ctrl_held: bool, now: float | None = None
    ) -> ChordEvent:
        """`key`: a single letter, e.g. "R", "C", "H" (case-insensitive)."""
        key = key.upper()
        now = time.monotonic() if now is None else now

        if not ctrl_held:
            # Spec requires holding Ctrl throughout the whole chord.
            was_pending = self.is_pending
            self.reset()
            return ChordEvent(
                ChordOutcome.CANCELLED if was_pending else ChordOutcome.IGNORED
            )

        if self.is_pending and (now - self._pending_since > self._timeout_seconds):
            self.reset()

        if not self.is_pending:
            if key in COLOR_KEYS:
                self._pending_color = COLOR_KEYS[key]
                self._pending_since = now
                return ChordEvent(ChordOutcome.PENDING, pending_label=self.pending_indicator)
            if key == HAND_KEY:
                return ChordEvent(ChordOutcome.ASSIGNED, class_name="hand")
            return ChordEvent(ChordOutcome.IGNORED)

        # Pending: this key is step 2 (role letter), e.g. H here means horse,
        # not hand -- we're only in this branch because a chord is pending.
        if key in ROLE_KEYS:
            color = self._pending_color
            role = ROLE_KEYS[key]
            self.reset()
            return ChordEvent(ChordOutcome.ASSIGNED, class_name=f"{color}_{role}")

        self.reset()
        return ChordEvent(ChordOutcome.CANCELLED)
