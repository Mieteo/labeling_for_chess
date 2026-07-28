import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        from pathlib import Path

        directory = Path(sys.argv[1])
        if directory.is_dir():
            window._open_directory(directory)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
