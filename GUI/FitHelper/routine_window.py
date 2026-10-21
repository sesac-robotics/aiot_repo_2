import os
from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic

class FullRoutineWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "full_routine.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, 'closeRoutineButton'):
            self.closeRoutineButton.clicked.connect(self.return_to_welcome)

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()