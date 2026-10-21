import os
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic
from video_receiver import VideoReceiver

AIFIT_SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://192.168.15.118:8000").rstrip("/")
STREAM_URL = f"{AIFIT_SERVER_URL}/stream"

class WorkoutWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "workout_window.ui")
        uic.loadUi(ui_path, self)

        self.receiver = None
        self.is_paused = False

        # 버튼 시그널 연결 확인
        if hasattr(self, 'endWorkoutButton'):
            self.endWorkoutButton.clicked.connect(self.return_to_welcome)
            
        if hasattr(self, 'pauseResumeButton'):
            self.pauseResumeButton.clicked.connect(self.toggle_pause)

        self.start_video()

    def start_video(self):
        if self.receiver is not None and self.receiver.isRunning():
            return
        self.receiver = VideoReceiver(STREAM_URL, self)
        self.receiver.frame_received.connect(self.update_frame)
        self.receiver.status_changed.connect(self.update_status)
        self.receiver.start()

    def update_frame(self, image):
        if self.is_paused:
            return

        if hasattr(self, 'videoLabel'):
            pixmap = QPixmap.fromImage(image)
            self.videoLabel.setPixmap(
                pixmap.scaled(
                    self.videoLabel.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
            )

    def update_status(self, message):
        if hasattr(self, 'feedbackLabel') and not self.is_paused:
            self.feedbackLabel.setText(f"💡 AI Coach: {message}")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            if hasattr(self, 'pauseResumeButton'):
                self.pauseResumeButton.setText("▶ Resume")
            if hasattr(self, 'feedbackLabel'):
                self.feedbackLabel.setText("💡 AI Coach: 일시정지됨")
        else:
            if hasattr(self, 'pauseResumeButton'):
                self.pauseResumeButton.setText("⏸ Pause")
            if hasattr(self, 'feedbackLabel'):
                self.feedbackLabel.setText("💡 AI Coach: 운동 재개")

    def return_to_welcome(self):
        # 1. 스트리밍 스레드 중지 요청 및 안전한 대기 처리
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            # 메인 GUI가 멈추지 않도록 타임아웃을 짧게 주어 대기
            self.receiver.wait(300)
        
        # 2. 부모 창(대시보드) 표시
        if self.parent_window is not None:
            self.parent_window.show()
        
        # 3. 현재 운동 창 닫기
        self.close()

    def closeEvent(self, event):
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(300)
        if self.parent_window is not None:
            self.parent_window.show()
        event.accept()