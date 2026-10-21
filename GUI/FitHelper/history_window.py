import os
from PyQt6.QtWidgets import QMainWindow, QDialog, QVBoxLayout, QCalendarWidget
from PyQt6 import uic

class HistoryDetailWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "history_detail.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, 'backDashboardButton'):
            self.backDashboardButton.clicked.connect(self.return_to_welcome)

        if hasattr(self, 'openCalendarButton'):
            self.openCalendarButton.clicked.connect(self.open_calendar_view)

        self.mock_workout_records = {
            "2026-03-15": "✅ Completed: Day 1 - Shoulder Mobility (Score: 92)",
            "2026-03-16": "✅ Completed: Day 2 - Core Stability (Score: 88)",
            "2026-03-20": "✅ Completed: Day 3 - Posture Release (Score: 95)"
        }

    def open_calendar_view(self):
        self.calendar_dialog = QDialog(self)
        self.calendar_dialog.setWindowTitle("Workout Calendar")
        self.calendar_dialog.resize(400, 300)
        
        layout = QVBoxLayout(self.calendar_dialog)
        calendar_widget = QCalendarWidget(self.calendar_dialog)
        calendar_widget.clicked.connect(lambda qdate: self.on_date_selected(qdate))
        
        layout.addWidget(calendar_widget)
        self.calendar_dialog.setLayout(layout)
        self.calendar_dialog.exec()

    def on_date_selected(self, qdate):
        date_str = qdate.toString("yyyy-MM-dd")
        if hasattr(self, 'dateEdit'):
            self.dateEdit.setDate(qdate)

        record_text = self.mock_workout_records.get(date_str, f"❌ No workout record for {date_str}")
        if hasattr(self, 'recordLabel'):
            self.recordLabel.setText(record_text)

        self.calendar_dialog.accept()

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()