"""Editable Xiangqi board and board-FEN helpers.

The image-label format keeps board metadata in a sidecar JSON file.  This
module deliberately has no dependency on that file format: it only owns the
9 x 10 position editor and the board part of a Xiangqi FEN.  Keeping that
boundary small lets the main window load/save metadata without ever mixing it
into the strict five-column YOLO ``.txt`` labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Final, TypeAlias

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


Square: TypeAlias = tuple[int, int]  # (file 0..8, rank 0..9), black at rank 0

BOARD_FILES: Final = 9
BOARD_RANKS: Final = 10
EMPTY_BOARD_FEN: Final = "9/9/9/9/9/9/9/9/9/9"
STARTING_BOARD_FEN: Final = (
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/"
    "P1P1P1P1P/1C5C1/9/RNBAKABNR"
)
VALID_PIECES: Final = frozenset("KABNRCPkabnrcp")

# Maximum counts in one legal starting set.  The validator intentionally does
# not reject a reduced position: incomplete positions are useful while an
# annotator is still reconstructing a photo.  It does flag impossible extras.
MAX_PIECE_COUNTS: Final = {
    "K": 1,
    "A": 2,
    "B": 2,
    "N": 2,
    "R": 2,
    "C": 2,
    "P": 5,
    "k": 1,
    "a": 2,
    "b": 2,
    "n": 2,
    "r": 2,
    "c": 2,
    "p": 5,
}

_PIECE_CODE_NAMES: Final = {
    "K": "red_king",
    "A": "red_advisor",
    "B": "red_elephant",
    "N": "red_horse",
    "R": "red_rook",
    "C": "red_cannon",
    "P": "red_pawn",
    "k": "black_king",
    "a": "black_advisor",
    "b": "black_elephant",
    "n": "black_horse",
    "r": "black_rook",
    "c": "black_cannon",
    "p": "black_pawn",
}

_PIECE_DISPLAY_NAMES: Final = {
    "K": "Đỏ: Tướng (K)",
    "A": "Đỏ: Sĩ (A)",
    "B": "Đỏ: Tượng (B)",
    "N": "Đỏ: Mã (N)",
    "R": "Đỏ: Xe (R)",
    "C": "Đỏ: Pháo (C)",
    "P": "Đỏ: Tốt (P)",
    "k": "Đen: Tướng (k)",
    "a": "Đen: Sĩ (a)",
    "b": "Đen: Tượng (b)",
    "n": "Đen: Mã (n)",
    "r": "Đen: Xe (r)",
    "c": "Đen: Pháo (c)",
    "p": "Đen: Tốt (p)",
}

# Chinese glyphs help an annotator compare the editor with a photographed
# board.  The ASCII FEN letter remains the source of truth and is used as a
# fallback if the system lacks a suitable CJK font.
_PIECE_GLYPHS: Final = {
    "K": "帥",
    "A": "仕",
    "B": "相",
    "N": "傌",
    "R": "俥",
    "C": "炮",
    "P": "兵",
    "k": "將",
    "a": "士",
    "b": "象",
    "n": "馬",
    "r": "車",
    "c": "砲",
    "p": "卒",
}

_PIECE_MIME_TYPE: Final = "application/x-xiangqi-piece"


class PiecePaletteButton(QPushButton):
    """A visual Xiangqi piece that can be selected or dragged onto the board."""

    def __init__(self, piece: str, parent: QWidget | None = None):
        super().__init__(_PIECE_GLYPHS[piece], parent)
        self.piece = piece
        self._press_position: QPoint | None = None
        self.setObjectName(f"piecePalette{piece}")
        self.setToolTip(f"{_PIECE_DISPLAY_NAMES[piece]} — kéo vào bàn cờ")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMinimumSize(38, 38)
        self.setStyleSheet(
            "QPushButton {"
            " border: 1px solid #4e3923; border-radius: 19px;"
            " background: #fffdf6; font-weight: bold; font-size: 20px;"
            f" color: {'#c62828' if piece.isupper() else '#202124'};"
            "}"
            "QPushButton:hover { background: #fff1b8; }"
            "QPushButton:checked { border: 3px solid #e6a500; background: #fff1b8; }"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if (
            self._press_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._press_position).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._start_drag()
            self._press_position = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._press_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _start_drag(self) -> None:
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData(_PIECE_MIME_TYPE, self.piece.encode("ascii"))
        drag.setMimeData(mime_data)
        drag.setPixmap(self._drag_pixmap())
        drag.setHotSpot(QPoint(19, 19))
        drag.exec(Qt.DropAction.CopyAction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _drag_pixmap(self) -> QPixmap:
        pixmap = QPixmap(42, 42)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#4e3923"), 2))
            painter.setBrush(QColor("#fffdf6"))
            painter.drawEllipse(QPointF(21, 21), 18, 18)
            font = QFont(self.font())
            font.setBold(True)
            font.setPointSizeF(18)
            painter.setFont(font)
            painter.setPen(QColor("#c62828") if self.piece.isupper() else QColor("#202124"))
            painter.drawText(QRectF(3, 3, 36, 36), Qt.AlignmentFlag.AlignCenter, _PIECE_GLYPHS[self.piece])
        finally:
            painter.end()
        return pixmap


def describe_validation_issue(issue: str) -> str:
    """Turn a stable validator code into a short annotator-facing message."""

    if issue == "facing_kings":
        return "Hai tướng đối mặt trên cùng một cột"
    if issue.startswith("red_king_count_"):
        return f"Đỏ phải có đúng 1 tướng (đang có {issue.rsplit('_', 1)[1]})"
    if issue.startswith("black_king_count_"):
        return f"Đen phải có đúng 1 tướng (đang có {issue.rsplit('_', 1)[1]})"
    if "_outside_palace_" in issue:
        piece = issue.rsplit("_outside_palace_", 1)[0].replace("_", " ")
        return f"{piece} nằm ngoài cung"
    if issue.startswith("red_elephant_crossed_river_"):
        return "Tượng đỏ đã qua sông"
    if issue.startswith("black_elephant_crossed_river_"):
        return "Tượng đen đã qua sông"
    if issue.startswith("too_many_"):
        return "Số lượng quân vượt quá giới hạn: " + issue.removeprefix("too_many_").replace("_", " ")
    return issue.replace("_", " ")


def is_valid_square(square: Square) -> bool:
    """Return whether ``square`` names one of the 90 intersections."""

    return (
        isinstance(square, tuple)
        and len(square) == 2
        and isinstance(square[0], int)
        and isinstance(square[1], int)
        and 0 <= square[0] < BOARD_FILES
        and 0 <= square[1] < BOARD_RANKS
    )


def _require_square(square: Square) -> None:
    if not is_valid_square(square):
        raise ValueError(f"Invalid Xiangqi square: {square!r}")


class XiangqiBoard:
    """A mutable 9 x 10 Xiangqi position encoded as board FEN.

    Ranks are stored in FEN order: rank 0 is Black's back rank at the top of
    an unflipped board, and rank 9 is Red's back rank.  This is intentionally
    a position editor, not a move-legality engine: arbitrary edit operations
    are allowed and :meth:`validation_issues` reports the useful structural
    warnings afterwards.
    """

    def __init__(self, fen: str | None = None):
        self._pieces: dict[Square, str] = {}
        self.load_fen(STARTING_BOARD_FEN if fen is None else fen)

    @classmethod
    def empty(cls) -> "XiangqiBoard":
        return cls(EMPTY_BOARD_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "XiangqiBoard":
        return cls(fen)

    def copy(self) -> "XiangqiBoard":
        copied = XiangqiBoard.empty()
        copied._pieces = dict(self._pieces)
        return copied

    def load_fen(self, fen: str) -> None:
        """Replace this position with a board-FEN string.

        ``fen`` may include a full-FEN suffix (side-to-move and move fields);
        only the first, board-position field belongs to this editor.  Invalid
        row counts, invalid piece letters and ranks that do not expand to nine
        intersections raise ``ValueError`` instead of silently corrupting an
        annotation.
        """

        if not isinstance(fen, str) or not fen.strip():
            raise ValueError("Board FEN must be a non-empty string")
        board_field = fen.strip().split()[0]
        ranks = board_field.split("/")
        if len(ranks) != BOARD_RANKS:
            raise ValueError(
                f"Board FEN must contain {BOARD_RANKS} ranks, got {len(ranks)}"
            )

        parsed: dict[Square, str] = {}
        for rank, encoded_rank in enumerate(ranks):
            file = 0
            for token in encoded_rank:
                if token in "123456789":
                    file += int(token)
                elif token in VALID_PIECES:
                    if file >= BOARD_FILES:
                        raise ValueError(f"Rank {rank + 1} is wider than 9 files")
                    parsed[(file, rank)] = token
                    file += 1
                else:
                    raise ValueError(f"Invalid FEN token {token!r} in rank {rank + 1}")
                if file > BOARD_FILES:
                    raise ValueError(f"Rank {rank + 1} is wider than 9 files")
            if file != BOARD_FILES:
                raise ValueError(
                    f"Rank {rank + 1} must expand to 9 files, got {file}"
                )

        self._pieces = parsed

    def to_fen(self) -> str:
        """Return the board-only Xiangqi FEN for the current position."""

        encoded_ranks: list[str] = []
        for rank in range(BOARD_RANKS):
            empty_count = 0
            result: list[str] = []
            for file in range(BOARD_FILES):
                piece = self._pieces.get((file, rank))
                if piece is None:
                    empty_count += 1
                    continue
                if empty_count:
                    result.append(str(empty_count))
                    empty_count = 0
                result.append(piece)
            if empty_count:
                result.append(str(empty_count))
            encoded_ranks.append("".join(result))
        return "/".join(encoded_ranks)

    @property
    def fen(self) -> str:
        return self.to_fen()

    def pieces(self) -> dict[Square, str]:
        """Return a copy of the current square-to-piece mapping."""

        return dict(self._pieces)

    def piece_at(self, square: Square) -> str | None:
        _require_square(square)
        return self._pieces.get(square)

    def set_piece(self, square: Square, piece: str | None) -> bool:
        """Place ``piece`` at a square, replacing its prior occupant.

        Passing ``None`` clears the square.  The boolean result tells callers
        whether this changed the board, which prevents spurious dirty states
        and signals from UI controls.
        """

        _require_square(square)
        if piece is not None and piece not in VALID_PIECES:
            raise ValueError(f"Unsupported Xiangqi FEN piece: {piece!r}")
        old_piece = self._pieces.get(square)
        if old_piece == piece:
            return False
        if piece is None:
            self._pieces.pop(square, None)
        else:
            self._pieces[square] = piece
        return True

    def remove(self, square: Square) -> str | None:
        """Remove and return a piece, or ``None`` for an already empty square."""

        _require_square(square)
        return self._pieces.pop(square, None)

    def move(self, source: Square, target: Square) -> bool:
        """Move a piece, capturing any target occupant.  No move rules apply."""

        _require_square(source)
        _require_square(target)
        if source == target:
            return False
        piece = self._pieces.get(source)
        if piece is None:
            return False
        self._pieces[target] = piece
        del self._pieces[source]
        return True

    def reset(self) -> bool:
        """Restore the normal starting scaffold and return whether it changed."""

        return self._replace_with_fen(STARTING_BOARD_FEN)

    def clear(self) -> bool:
        """Remove every piece and return whether anything was removed."""

        if not self._pieces:
            return False
        self._pieces.clear()
        return True

    def _replace_with_fen(self, fen: str) -> bool:
        before = self.to_fen()
        self.load_fen(fen)
        return before != self.to_fen()

    def validation_issues(self) -> list[str]:
        """Return stable warning codes for common impossible positions.

        The codes are deliberately terse and machine-friendly so callers can
        store/display them as desired.  They cover annotation mistakes that
        materially affect a generated FEN without pretending to be a complete
        Xiangqi legality checker.
        """

        issues: list[str] = []
        counts = Counter(self._pieces.values())

        for king, color in (("K", "red"), ("k", "black")):
            count = counts[king]
            if count != 1:
                issues.append(f"{color}_king_count_{count}")

        for piece, maximum in MAX_PIECE_COUNTS.items():
            count = counts[piece]
            if count > maximum:
                issues.append(f"too_many_{_PIECE_CODE_NAMES[piece]}_{count}")

        red_total = sum(counts[piece] for piece in "KABNRCP")
        black_total = sum(counts[piece] for piece in "kabnrcp")
        if red_total > 16:
            issues.append(f"too_many_red_pieces_{red_total}")
        if black_total > 16:
            issues.append(f"too_many_black_pieces_{black_total}")

        for square, piece in sorted(self._pieces.items(), key=lambda item: (item[0][1], item[0][0])):
            file, rank = square
            color = "red" if piece.isupper() else "black"
            if piece.upper() in {"K", "A"} and not self._in_palace(color, square):
                issues.append(f"{_PIECE_CODE_NAMES[piece]}_outside_palace_{file}_{rank}")
            if piece == "B" and rank < 5:
                issues.append(f"red_elephant_crossed_river_{file}_{rank}")
            if piece == "b" and rank > 4:
                issues.append(f"black_elephant_crossed_river_{file}_{rank}")

        red_king = next((square for square, piece in self._pieces.items() if piece == "K"), None)
        black_king = next((square for square, piece in self._pieces.items() if piece == "k"), None)
        if red_king is not None and black_king is not None and red_king[0] == black_king[0]:
            file = red_king[0]
            first_rank, last_rank = sorted((red_king[1], black_king[1]))
            if all((file, rank) not in self._pieces for rank in range(first_rank + 1, last_rank)):
                issues.append("facing_kings")

        return issues

    @staticmethod
    def _in_palace(color: str, square: Square) -> bool:
        file, rank = square
        if not 3 <= file <= 5:
            return False
        return 7 <= rank <= 9 if color == "red" else 0 <= rank <= 2


class XiangqiBoardView(QWidget):
    """Painted board surface used by :class:`XiangqiBoardEditor`.

    The view deliberately emits interaction requests rather than mutating the
    model itself.  That gives the editor one place to distinguish user edits
    from programmatic loads and ensures ``boardChanged`` never fires merely
    because metadata was loaded from disk.
    """

    moveRequested = Signal(object, object)  # source square, target square
    placeRequested = Signal(object, str)  # target square, FEN piece
    deleteRequested = Signal(object)  # square
    undoRequested = Signal()
    redoRequested = Signal()

    def __init__(self, board: XiangqiBoard, parent: QWidget | None = None):
        super().__init__(parent)
        self._board = board
        self._flipped = False
        self._placing_piece: str | None = None
        self._selected: Square | None = None
        self._press_square: Square | None = None
        self._drag_source: Square | None = None
        self._drag_position = QPoint()
        self._drag_started = False
        self._drop_square: Square | None = None
        self.setMinimumSize(260, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

    @property
    def selected_square(self) -> Square | None:
        return self._selected

    @property
    def flipped(self) -> bool:
        return self._flipped

    def set_board(self, board: XiangqiBoard) -> None:
        self._board = board
        self._selected = None
        self._cancel_drag()
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        if self._flipped != flipped:
            self._flipped = flipped
            self._selected = None
            self._cancel_drag()
            self.update()

    def set_placing_piece(self, piece: str | None) -> None:
        if piece is not None and piece not in VALID_PIECES:
            raise ValueError(f"Unsupported palette piece: {piece!r}")
        self._placing_piece = piece
        self._selected = None
        self._cancel_drag()
        self.update()

    def square_center(self, square: Square) -> QPoint:
        """Return the current viewport centre for a logical board square.

        This small public helper keeps widget-level tests and future UI
        automation independent from a magic pixel layout.
        """

        _require_square(square)
        display_file, display_rank = self._to_display_square(square)
        rect = self._board_rect()
        x_step = rect.width() / (BOARD_FILES - 1)
        y_step = rect.height() / (BOARD_RANKS - 1)
        return QPoint(
            round(rect.left() + display_file * x_step),
            round(rect.top() + display_rank * y_step),
        )

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#f7f2e8"))
            rect = self._board_rect()
            self._paint_board(painter, rect)
            self._paint_pieces(painter, rect)
        finally:
            painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        square = self._square_at(event.position())
        if square is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.setFocus()
            self.deleteRequested.emit(square)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        self.setFocus()

        if self._placing_piece is not None:
            self.placeRequested.emit(square, self._placing_piece)
            event.accept()
            return

        self._press_square = square
        self._drag_source = square if self._board.piece_at(square) is not None else None
        self._drag_position = event.position().toPoint()
        self._drag_started = False
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_source is not None and event.buttons() & Qt.MouseButton.LeftButton:
            position = event.position().toPoint()
            if self._drag_started or (position - self._drag_position).manhattanLength() >= 4:
                self._drag_started = True
                self._drag_position = position
                self.update()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._press_square is None:
            super().mouseReleaseEvent(event)
            return

        source = self._press_square
        target = self._square_at(event.position())
        drag_source = self._drag_source
        dragged = self._drag_started
        self._press_square = None
        self._cancel_drag()

        if target is not None:
            if dragged and drag_source is not None and target != drag_source:
                self.moveRequested.emit(drag_source, target)
            elif self._selected is not None:
                if target == self._selected:
                    self._selected = None
                else:
                    self.moveRequested.emit(self._selected, target)
                    self._selected = None
            elif self._board.piece_at(source) is not None:
                self._selected = source
        self.update()
        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._piece_from_mime(event.mimeData()) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._piece_from_mime(event.mimeData()) is None:
            event.ignore()
            return
        self._drop_square = self._square_at(event.position())
        self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._drop_square = None
        self.update()
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        piece = self._piece_from_mime(event.mimeData())
        square = self._square_at(event.position())
        self._drop_square = None
        self.update()
        if piece is None or square is None:
            event.ignore()
            return
        self.setFocus()
        self.placeRequested.emit(square, piece)
        event.acceptProposedAction()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected is not None:
            square = self._selected
            self._selected = None
            self.deleteRequested.emit(square)
            self.update()
            event.accept()
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.redoRequested.emit()
            else:
                self.undoRequested.emit()
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Y:
            self.redoRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _cancel_drag(self) -> None:
        self._press_square = None
        self._drag_source = None
        self._drag_started = False

    @staticmethod
    def _piece_from_mime(mime_data: QMimeData) -> str | None:
        if not mime_data.hasFormat(_PIECE_MIME_TYPE):
            return None
        try:
            piece = bytes(mime_data.data(_PIECE_MIME_TYPE)).decode("ascii")
        except UnicodeDecodeError:
            return None
        return piece if piece in VALID_PIECES else None

    def _board_rect(self) -> QRectF:
        contents = QRectF(self.contentsRect())
        margin = min(28.0, max(12.0, min(contents.width(), contents.height()) * 0.06))
        available_width = max(1.0, contents.width() - 2 * margin)
        available_height = max(1.0, contents.height() - 2 * margin)
        target_ratio = 8.0 / 9.0  # eight horizontal vs nine vertical intervals
        if available_width / available_height > target_ratio:
            height = available_height
            width = height * target_ratio
        else:
            width = available_width
            height = width / target_ratio
        return QRectF(
            contents.center().x() - width / 2,
            contents.center().y() - height / 2,
            width,
            height,
        )

    def _square_at(self, point: QPointF) -> Square | None:
        rect = self._board_rect()
        x_step = rect.width() / (BOARD_FILES - 1)
        y_step = rect.height() / (BOARD_RANKS - 1)
        display_file = round((point.x() - rect.left()) / x_step)
        display_rank = round((point.y() - rect.top()) / y_step)
        if not (0 <= display_file < BOARD_FILES and 0 <= display_rank < BOARD_RANKS):
            return None
        nearest_x = rect.left() + display_file * x_step
        nearest_y = rect.top() + display_rank * y_step
        if abs(point.x() - nearest_x) > x_step * 0.52 or abs(point.y() - nearest_y) > y_step * 0.52:
            return None
        return self._from_display_square((display_file, display_rank))

    def _to_display_square(self, square: Square) -> Square:
        file, rank = square
        return (BOARD_FILES - 1 - file, BOARD_RANKS - 1 - rank) if self._flipped else square

    def _from_display_square(self, square: Square) -> Square:
        return self._to_display_square(square)

    def _paint_board(self, painter: QPainter, rect: QRectF) -> None:
        x_step = rect.width() / (BOARD_FILES - 1)
        y_step = rect.height() / (BOARD_RANKS - 1)
        pen = QPen(QColor("#75522d"), max(1.0, min(x_step, y_step) * 0.035))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Horizontal ranks extend across the complete board.
        for rank in range(BOARD_RANKS):
            y = rect.top() + rank * y_step
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # Interior files pause at the river; the outer files remain continuous.
        for file in range(BOARD_FILES):
            x = rect.left() + file * x_step
            if file in {0, BOARD_FILES - 1}:
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            else:
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.top() + 4 * y_step))
                painter.drawLine(QPointF(x, rect.top() + 5 * y_step), QPointF(x, rect.bottom()))

        for top_rank in (0, 7):
            bottom_rank = top_rank + 2
            painter.drawLine(
                QPointF(rect.left() + 3 * x_step, rect.top() + top_rank * y_step),
                QPointF(rect.left() + 5 * x_step, rect.top() + bottom_rank * y_step),
            )
            painter.drawLine(
                QPointF(rect.left() + 5 * x_step, rect.top() + top_rank * y_step),
                QPointF(rect.left() + 3 * x_step, rect.top() + bottom_rank * y_step),
            )

        river_font = QFont(self.font())
        river_font.setBold(True)
        river_font.setPointSizeF(max(8.0, min(x_step, y_step) * 0.33))
        painter.setFont(river_font)
        painter.setPen(QColor("#8c6239"))
        painter.drawText(
            QRectF(rect.left(), rect.top() + 4 * y_step, rect.width() / 2, y_step),
            Qt.AlignmentFlag.AlignCenter,
            "楚河",
        )
        painter.drawText(
            QRectF(rect.left() + rect.width() / 2, rect.top() + 4 * y_step, rect.width() / 2, y_step),
            Qt.AlignmentFlag.AlignCenter,
            "漢界",
        )

    def _paint_pieces(self, painter: QPainter, rect: QRectF) -> None:
        x_step = rect.width() / (BOARD_FILES - 1)
        y_step = rect.height() / (BOARD_RANKS - 1)
        radius = max(9.0, min(x_step, y_step) * 0.36)
        piece_font = QFont(self.font())
        piece_font.setBold(True)
        piece_font.setPointSizeF(max(8.0, radius * 0.9))
        painter.setFont(piece_font)

        for square, piece in self._board.pieces().items():
            if self._drag_started and square == self._drag_source:
                continue
            center = self.square_center(square)
            if square == self._selected:
                painter.setPen(QPen(QColor("#e6a500"), max(2.0, radius * 0.18)))
                painter.setBrush(QColor("#fff1b8"))
            else:
                painter.setPen(QPen(QColor("#4e3923"), max(1.0, radius * 0.08)))
                painter.setBrush(QColor("#fffdf6"))
            painter.drawEllipse(QPointF(center), radius, radius)
            painter.setPen(QColor("#c62828") if piece.isupper() else QColor("#202124"))
            painter.drawText(
                QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                Qt.AlignmentFlag.AlignCenter,
                _PIECE_GLYPHS[piece],
            )

        if self._drop_square is not None:
            center = self.square_center(self._drop_square)
            painter.setPen(QPen(QColor("#e6a500"), max(2.0, radius * 0.14)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(center), radius * 1.12, radius * 1.12)

        if self._drag_started and self._drag_source is not None:
            piece = self._board.piece_at(self._drag_source)
            if piece is not None:
                center = QPointF(self._drag_position)
                painter.setPen(QPen(QColor("#4e3923"), max(1.0, radius * 0.08)))
                painter.setBrush(QColor(255, 253, 246, 210))
                painter.drawEllipse(center, radius, radius)
                painter.setPen(QColor("#c62828") if piece.isupper() else QColor("#202124"))
                painter.drawText(
                    QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    _PIECE_GLYPHS[piece],
                )


class XiangqiBoardEditor(QWidget):
    """Compact, self-contained board-FEN editor for a right-side dock.

    ``boardChanged`` is emitted only after a direct user operation (mouse
    move/place/delete, toolbar reset or clear).  Loading a sidecar metadata
    file through :meth:`set_board_fen` redraws the editor but remains clean.
    """

    boardChanged = Signal(str, list)  # board FEN, validation issue codes

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._board = XiangqiBoard()
        # The history contains board-only FEN snapshots.  Presentation state
        # (selection, palette, display flip) is intentionally excluded: Undo
        # means undo a position edit, not a view preference.
        self._history: list[str] = [self._board.to_fen()]
        self._history_index = 0
        self._clean_fen = self._board.to_fen()
        self._view = XiangqiBoardView(self._board, self)
        self._palette_buttons: dict[str, PiecePaletteButton] = {}
        self._fen_label = QLabel(self)
        self._issues_label = QLabel(self)
        self._delete_button = QPushButton("Xóa ô chọn", self)
        self._move_button = QPushButton("Di chuyển / bắt", self)
        self._undo_button = QPushButton("Hoàn tác", self)
        self._redo_button = QPushButton("Làm lại", self)
        self._copy_button = QPushButton("Sao chép FEN", self)
        self._reset_button = QPushButton("Thế cờ đầu", self)
        self._empty_button = QPushButton("Bàn trống", self)
        self._flip_button = QPushButton("Lật bàn", self)
        self._build_ui()
        self._connect_signals()
        self._refresh_display()

    @property
    def board_fen(self) -> str:
        return self._board.to_fen()

    @property
    def fen(self) -> str:
        """Alias for :attr:`board_fen` useful in a form binding."""

        return self.board_fen

    @property
    def validation_issues(self) -> list[str]:
        return self._board.validation_issues()

    @property
    def is_dirty(self) -> bool:
        """Whether the board differs from the last loaded/saved position."""

        return self.board_fen != self._clean_fen

    @property
    def can_undo(self) -> bool:
        return self._history_index > 0

    @property
    def can_redo(self) -> bool:
        return self._history_index < len(self._history) - 1

    @property
    def is_flipped(self) -> bool:
        return self._view.flipped

    def board(self) -> XiangqiBoard:
        """Return a copy so callers cannot mutate without a user signal."""

        return self._board.copy()

    def set_board_fen(self, fen: str) -> None:
        """Load a FEN programmatically without emitting ``boardChanged``."""

        self._board.load_fen(fen)
        self._reset_history_as_clean()
        self._view.set_board(self._board)
        self._refresh_display()

    def set_starting_position(self) -> None:
        """Programmatically show the initial scaffold without dirtying data."""

        self.set_board_fen(STARTING_BOARD_FEN)

    def set_empty_position(self) -> None:
        """Programmatically show an empty board without dirtying data."""

        self.set_board_fen(EMPTY_BOARD_FEN)

    def set_flipped(self, flipped: bool) -> None:
        """Change only visual orientation; FEN and dirty state stay intact."""

        self._view.set_flipped(flipped)

    def mark_clean(self) -> None:
        """Mark the current position saved after its sidecar metadata is written.

        The history remains available, so an annotator can still undo a saved
        operation if necessary; only the destructive-operation confirmation
        baseline changes.
        """

        self._clean_fen = self.board_fen

    def undo(self) -> bool:
        """Restore the prior board snapshot and emit its live FEN."""

        if not self.can_undo:
            return False
        self._history_index -= 1
        self._restore_history_position()
        return True

    def redo(self) -> bool:
        """Restore the next board snapshot and emit its live FEN."""

        if not self.can_redo:
            return False
        self._history_index += 1
        self._restore_history_position()
        return True

    def copy_board_fen(self) -> str:
        """Copy the board-only FEN and return it for simple callers/tests."""

        fen = self.board_fen
        QApplication.clipboard().setText(fen)
        self._copy_button.setToolTip("Đã sao chép FEN")
        return fen

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        """Handle the documented history keys while this editor owns focus."""

        modifiers = event.modifiers()
        has_control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if has_control and event.key() == Qt.Key.Key_Z:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
            event.accept()
            return
        if has_control and event.key() == Qt.Key.Key_Y:
            self.redo()
            event.accept()
            return
        super().keyPressEvent(event)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(5)
        board_and_palettes = QHBoxLayout()
        board_and_palettes.setContentsMargins(0, 0, 0, 0)
        board_and_palettes.setSpacing(4)
        board_and_palettes.addLayout(self._build_palette("black", "kabnrcp"))
        board_and_palettes.addWidget(self._view, 1)
        board_and_palettes.addLayout(self._build_palette("red", "KABNRCP"))
        root.addLayout(board_and_palettes, 1)

        editing_actions = QHBoxLayout()
        editing_actions.setContentsMargins(0, 0, 0, 0)
        editing_actions.addWidget(self._move_button)
        editing_actions.addWidget(self._delete_button)
        root.addLayout(editing_actions)

        history_actions = QHBoxLayout()
        history_actions.setContentsMargins(0, 0, 0, 0)
        history_actions.addWidget(self._undo_button)
        history_actions.addWidget(self._redo_button)
        history_actions.addWidget(self._copy_button)
        root.addLayout(history_actions)

        position_actions = QHBoxLayout()
        position_actions.setContentsMargins(0, 0, 0, 0)
        position_actions.addWidget(self._reset_button)
        position_actions.addWidget(self._empty_button)
        position_actions.addWidget(self._flip_button)
        root.addLayout(position_actions)

        self._fen_label.setObjectName("boardFenLabel")
        self._fen_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._fen_label.setWordWrap(True)
        self._fen_label.setStyleSheet("font-family: Consolas, monospace; font-size: 10px;")
        root.addWidget(self._fen_label)

        self._issues_label.setObjectName("boardValidationLabel")
        self._issues_label.setWordWrap(True)
        root.addWidget(self._issues_label)

    def _build_palette(self, color: str, pieces: str) -> QVBoxLayout:
        """Build one compact side tray; black stays left and red stays right."""

        palette = QVBoxLayout()
        palette.setContentsMargins(0, 0, 0, 0)
        palette.setSpacing(3)
        title = QLabel("Đen" if color == "black" else "Đỏ", self)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-weight: bold; color: {'#202124' if color == 'black' else '#c62828'};")
        palette.addWidget(title)
        for piece in pieces:
            button = PiecePaletteButton(piece, self)
            button.clicked.connect(lambda checked, selected=piece: self._select_palette_piece(selected, checked))
            self._palette_buttons[piece] = button
            palette.addWidget(button)
        palette.addStretch(1)
        return palette

    def _connect_signals(self) -> None:
        self._move_button.clicked.connect(self._select_move_mode)
        self._delete_button.clicked.connect(self._delete_selected)
        self._undo_button.clicked.connect(self.undo)
        self._redo_button.clicked.connect(self.redo)
        self._copy_button.clicked.connect(self.copy_board_fen)
        self._reset_button.clicked.connect(self._reset_user)
        self._empty_button.clicked.connect(self._clear_user)
        self._flip_button.clicked.connect(self._flip_display)
        self._view.moveRequested.connect(self._move_user)
        self._view.placeRequested.connect(self._place_user)
        self._view.deleteRequested.connect(self._delete_user)
        self._view.undoRequested.connect(self.undo)
        self._view.redoRequested.connect(self.redo)

    def _select_palette_piece(self, piece: str, checked: bool) -> None:
        """Allow click-to-place as a quick alternative to drag-and-drop."""

        for candidate, button in self._palette_buttons.items():
            if candidate != piece:
                button.setChecked(False)
        self._view.set_placing_piece(piece if checked else None)

    def _select_move_mode(self) -> None:
        for button in self._palette_buttons.values():
            button.setChecked(False)
        self._view.set_placing_piece(None)

    def _move_user(self, source: Square, target: Square) -> None:
        if self._board.move(source, target):
            self._after_user_change()

    def _place_user(self, square: Square, piece: str) -> None:
        if self._board.set_piece(square, piece):
            self._after_user_change()

    def _delete_user(self, square: Square) -> None:
        if self._board.remove(square) is not None:
            self._after_user_change()

    def _delete_selected(self) -> None:
        square = self._view.selected_square
        if square is not None:
            self._delete_user(square)

    def _reset_user(self) -> None:
        if self.board_fen == STARTING_BOARD_FEN:
            return
        if self.is_dirty and not self._confirm_destructive_change("Khôi phục thế cờ đầu"):
            return
        if self._board.reset():
            self._after_user_change()

    def _clear_user(self) -> None:
        if self.board_fen == EMPTY_BOARD_FEN:
            return
        if self.is_dirty and not self._confirm_destructive_change("Xóa toàn bộ quân trên bàn"):
            return
        if self._board.clear():
            self._after_user_change()

    def _flip_display(self) -> None:
        self._view.set_flipped(not self._view.flipped)

    def _after_user_change(self) -> None:
        self._record_history_position()
        self._view.set_board(self._board)
        self._refresh_display()
        self.boardChanged.emit(self.board_fen, self.validation_issues)

    def _record_history_position(self) -> None:
        """Append a user-edited board, discarding an abandoned redo branch."""

        fen = self.board_fen
        if self._history[self._history_index] == fen:
            return
        del self._history[self._history_index + 1 :]
        self._history.append(fen)
        self._history_index = len(self._history) - 1

    def _reset_history_as_clean(self) -> None:
        self._history = [self.board_fen]
        self._history_index = 0
        self._clean_fen = self.board_fen

    def _restore_history_position(self) -> None:
        self._board.load_fen(self._history[self._history_index])
        self._view.set_board(self._board)
        self._refresh_display()
        # Undo/redo is still an explicit user operation, so parent metadata
        # bindings must receive the new FEN.
        self.boardChanged.emit(self.board_fen, self.validation_issues)

    def _confirm_destructive_change(self, action: str) -> bool:
        answer = QMessageBox.question(
            self,
            "Xác nhận thay đổi bàn cờ",
            f"{action} sẽ thay thế các chỉnh sửa FEN chưa lưu. Bạn có muốn tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _update_history_controls(self) -> None:
        self._undo_button.setEnabled(self.can_undo)
        self._redo_button.setEnabled(self.can_redo)

    def _refresh_display(self) -> None:
        self._update_history_controls()
        self._fen_label.setText(f"FEN: {self.board_fen}")
        issues = self.validation_issues
        if issues:
            self._issues_label.setStyleSheet("color: #a15c00;")
            self._issues_label.setText("Cảnh báo: " + "; ".join(describe_validation_issue(issue) for issue in issues))
        else:
            self._issues_label.setStyleSheet("color: #2e7d32;")
            self._issues_label.setText("✓ Vị trí hợp lệ theo các kiểm tra cơ bản")
