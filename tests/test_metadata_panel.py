from PySide6.QtWidgets import QApplication, QScrollArea, QTabWidget

from chess_labeler import metadata
from chess_labeler.metadata_panel import MetadataPanel


def _ensure_qapp():
    return QApplication.instance() or QApplication([])


def test_capture_dropdowns_show_vietnamese_and_persist_stable_schema_codes():
    _ensure_qapp()
    panel = MetadataPanel()
    try:
        for field, allowed in metadata.CAPTURE_ENUMS.items():
            combo = panel._capture_controls[field]
            stored_values = {combo.itemData(index) for index in range(combo.count())}
            displayed = {combo.itemText(index) for index in range(combo.count())}
            assert stored_values == allowed
            assert combo.currentData() == "unknown"
            assert "unknown" not in displayed

        orientation = panel._orientation
        assert orientation.currentData() == "unknown"
        assert orientation.currentText() == "Chưa xác định"
        assert panel.values()["board"] == {"image_orientation": "unknown"}
        assert panel._capture_group.isEditable()
        assert panel._capture_group.currentData() == ""
        assert panel.values()["capture"]["capture_group"] is None
        assert panel._content_cohort.currentText() == "Chưa gán nhãn"
        assert panel.values()["capture"]["content_cohort"] is None
        panel._content_cohort.setCurrentIndex(panel._content_cohort.findData("screen_photo"))
        assert panel.values()["capture"]["content_cohort"] == "screen_photo"

        panel.set_recent_values([], ["session-002", "session-001"])
        assert panel._capture_group.itemData(1) == "session-002"
        panel._capture_group.setCurrentText("session-001")
        assert panel.values()["capture"]["capture_group"] == "session-001"
    finally:
        panel.close()


def test_metadata_panel_is_one_no_scroll_form_with_value_based_statuses():
    _ensure_qapp()
    panel = MetadataPanel()
    try:
        assert not panel.findChildren(QTabWidget)
        assert not panel.findChildren(QScrollArea)
        assert panel.findChild(type(panel._orientation), "capture_lighting") is panel._capture_controls["lighting"]
        assert panel._notes.placeholderText() == "Ghi chú (tùy chọn)"
        assert "1" in panel.findChild(type(panel._corner_presence), "cornerInstruction").text()

        panel.set_values(
            {
                "board": {
                    "corners_px": {
                        "top_left": {"x": 1, "y": 1},
                        "top_right": {"x": 9, "y": 1},
                        "bottom_right": {"x": 9, "y": 9},
                        "bottom_left": {"x": 1, "y": 9},
                    },
                    "board_fen": "9/9/9/9/9/9/9/9/9/9",
                }
            }
        )
        assert panel._corner_presence.text() == "Đã đánh dấu (4/4)"
        assert panel._fen_presence.text() == "Đã có FEN"

        panel.set_values({"capture": {"capture_group": "session-017"}})
        assert panel._capture_group.currentText() == "session-017"
        assert panel.values()["capture"]["capture_group"] == "session-017"
        panel.set_values({"capture": {"content_cohort": "procedural_render"}})
        assert panel._content_cohort.currentText() == "Render tạo bằng code"
    finally:
        panel.close()
