import os
import requests
import cv2
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout


AIFIT_SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL",
    "http://192.168.15.118:8000"
).rstrip("/")
STREAM_URL = f"{AIFIT_SERVER_URL}/stream"


class VideoReceiver(QThread):
    frame_received = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._response = None

    def run(self):
        buffer = bytearray()
        try:
            with requests.get(STREAM_URL, stream=True, timeout=(5, 30)) as response:
                self._response = response
                response.raise_for_status()
                self.status_changed.emit("서버 연결 성공 · 영상 대기 중")

                for chunk in response.iter_content(chunk_size=8192):
                    if self.isInterruptionRequested():
                        break
                    if not chunk:
                        continue
                    buffer.extend(chunk)

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            if len(buffer) > 1:
                                del buffer[:-1]
                            break

                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            break

                        jpeg_bytes = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, _ = rgb.shape
                        image = QImage(
                            rgb.data, width, height, rgb.strides[0],
                            QImage.Format.Format_RGB888
                        ).copy()
                        self.frame_received.emit(image)

        except requests.RequestException as error:
            if not self.isInterruptionRequested():
                self.status_changed.emit(f"서버 연결 오류: {error}")
        finally:
            self._response = None
            self.status_changed.emit("영상 수신 종료")

    def stop(self):
        self.requestInterruption()
        response = self._response
        if response is not None:
            try:
                response.close()
            except requests.RequestException:
                pass


class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIFit 실시간 운동 영상")
        self.resize(960, 720)

        self.video_label = QLabel("영상 연결 대기 중")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.status_label = QLabel(f"연결되지 않음 · 서버: {AIFIT_SERVER_URL}")

        self.start_button = QPushButton("영상 시작")
        self.stop_button = QPushButton("영상 종료")
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.receiver = None
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)

    def start_video(self):
        if self.receiver is not None and self.receiver.isRunning():
            return
        self.receiver = VideoReceiver(self)
        self.receiver.frame_received.connect(self.update_frame)
        self.receiver.status_changed.connect(self.status_label.setText)
        self.receiver.finished.connect(self.on_receiver_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.receiver.start()

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        )

    def stop_video(self):
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(3500)
        self.on_receiver_finished()

    def on_receiver_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(3500)
        event.accept()
