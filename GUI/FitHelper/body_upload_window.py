import os
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog
from PyQt6 import uic

class BodyAnalysisLoadingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "bodyAnalysisLoading.ui")
        uic.loadUi(ui_path, self)

        for btn in self.findChildren(QPushButton):
            text = btn.text().lower()
            if 'back' in text or 'home' in text or 'dashboard' in text:
                btn.clicked.connect(self.return_to_welcome)

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()


class BodyAnalysisReportWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "bodyAnalysisReport.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, 'viewRoutineButton'):
            self.viewRoutineButton.clicked.connect(self.go_to_routine)

    def go_to_routine(self):
        from routine_window import FullRoutineWindow
        self.routine_window = FullRoutineWindow(self.welcome_window)
        self.routine_window.show()
        self.close()


class BodyUploadWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "bodyUpload.ui")
        uic.loadUi(ui_path, self)

        self.uploaded_photos = {1: None, 2: None, 3: None, 4: None}

        if hasattr(self, 'uploadPhoto1Button'):
            self.uploadPhoto1Button.clicked.connect(lambda: self.select_photo(1, self.uploadPhoto1Button))
        if hasattr(self, 'uploadPhoto2Button'):
            self.uploadPhoto2Button.clicked.connect(lambda: self.select_photo(2, self.uploadPhoto2Button))
        if hasattr(self, 'uploadPhoto3Button'):
            self.uploadPhoto3Button.clicked.connect(lambda: self.select_photo(3, self.uploadPhoto3Button))
        if hasattr(self, 'uploadPhoto4Button'):
            self.uploadPhoto4Button.clicked.connect(lambda: self.select_photo(4, self.uploadPhoto4Button))

        if hasattr(self, 'submitAnalysisButton'):
            self.submitAnalysisButton.clicked.connect(self.submit_analysis)

        if hasattr(self, 'backDashboardButton'):
            self.backDashboardButton.clicked.connect(self.return_to_welcome)

    def select_photo(self, photo_index, button_widget):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Body Photo", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.uploaded_photos[photo_index] = file_name
            base_name = os.path.basename(file_name)
            if photo_index == 1:
                button_widget.setText(f"✔ Front\n({base_name[:10]}...)")
            elif photo_index == 2:
                button_widget.setText(f"✔ Left Side\n({base_name[:10]}...)")
            elif photo_index == 3:
                button_widget.setText(f"✔ Right Side\n({base_name[:10]}...)")
            elif photo_index == 4:
                button_widget.setText(f"✔ Back\n({base_name[:10]}...)")

    def submit_analysis(self):
        missing_photos = [idx for idx, path in self.uploaded_photos.items() if path is None]
        
        if missing_photos:
            if hasattr(self, 'errorMsgLabel'):
                self.errorMsgLabel.setText("⚠️ Please upload all 4 photos for precise AI assessment.")
            return

        if hasattr(self, 'errorMsgLabel'):
            self.errorMsgLabel.setText("")
        
        # 분석 완료 후 Report 화면으로 연결하거나 로딩 화면을 띄울 수 있습니다.
        self.report_window = BodyAnalysisReportWindow(self.welcome_window)
        self.report_window.show()
        self.close()

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()