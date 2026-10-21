import os
from PyQt6.QtWidgets import QMainWindow, QPushButton
from PyQt6.QtCore import Qt
from PyQt6 import uic
from routine_window import FullRoutineWindow

class BodyAnalysisReportWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        
        ui_path = os.path.join(os.path.dirname(__file__), "bodyAnalysisResult.ui")
        uic.loadUi(ui_path, self)

        # 1. View 7-Day Recommended Routine 버튼 클릭 시 full_routine.ui 창 오픈 연결
        if hasattr(self, 'viewRoutineButton'):
            self.viewRoutineButton.clicked.connect(self.open_full_routine)

        # 2. 대시보드로 돌아가는 버튼 연결
        for btn in self.findChildren(QPushButton):
            text = btn.text().lower()
            if 'back' in text or 'dashboard' in text or '홈' in text:
                btn.clicked.connect(self.return_to_welcome)
                break

    def open_full_routine(self):
        """7-Day Recommended Routine 버튼 클릭 시 대시보드를 다시 띄우지 않고 FullRoutineWindow만 단독 오픈"""
        self.full_routine_window = FullRoutineWindow(self.welcome_window)
        self.full_routine_window.show()
        # 대시보드가 중복으로 뜨지 않도록 현재 리포트 창만 깔끔하게 닫기
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.close()

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        event.accept()