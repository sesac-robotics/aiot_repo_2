import os
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QTimer
from PyQt6 import uic
from body_analysis_report import BodyAnalysisReportWindow

class BodyAnalysisLoadingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        
        ui_path = os.path.join(os.path.dirname(__file__), "bodyAnalysisLoading.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, 'analysisProgressBar'):
            self.analysisProgressBar.setValue(0)

        self.status_messages = {
            10: "Initializing AI posture assessment framework...",
            20: "Uploading and securing body scan images...",
            30: "Detecting key joints and skeletal alignment points...",
            40: "Analyzing spinal curvature and shoulder balance...",
            50: "Evaluating pelvic tilt and lower body symmetry...",
            60: "Calculating body fat distribution and type...",
            70: "Cross-referencing posture metrics with fitness database...",
            80: "Generating customized AI corrective prescription...",
            90: "Finalizing 7-day personalized workout master plan...",
            100: "Body analysis successfully completed! Preparing report..."
        }

        self.current_progress = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(10000)

    def update_progress(self):
        self.current_progress += 10

        if self.current_progress <= 100:
            if hasattr(self, 'analysisProgressBar'):
                self.analysisProgressBar.setValue(self.current_progress)
            
            if hasattr(self, 'analyzingStatusLabel'):
                msg = self.status_messages.get(self.current_progress, "Processing analysis...")
                self.analyzingStatusLabel.setText(msg)

        if self.current_progress >= 100:
            self.timer.stop()
            QTimer.singleShot(1000, self.finish_analysis_and_open_result)

    def finish_analysis_and_open_result(self):
        self.result_window = BodyAnalysisReportWindow(self.welcome_window)
        self.result_window.show()
        self.close()

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
