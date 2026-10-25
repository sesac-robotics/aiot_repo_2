import sys
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

# 1. 영상을 백그라운드에서 읽어오는 스레드 클래스

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.url)

        if not cap.isOpened():
            print(f"카메라 연결 실패: {self.url}")
            cap.release()
            return

        print(f"카메라 연결 성공: {self.url}")

        while self.running:
            ret, frame = cap.read()

            if not ret:
                print("영상 프레임 수신 실패")
                self.msleep(100)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape

            q_img = QImage(
                rgb_image.data,
                w,
                h,
                ch * w,
                QImage.Format.Format_RGB888
            ).copy()

            q_img = q_img.scaled(
                640,
                480,
                Qt.AspectRatioMode.KeepAspectRatio
            )

            self.change_pixmap_signal.emit(q_img)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# 2. 메인 윈도우 UI 클래스
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32-CAM PyQt6 스트리밍")
        self.setGeometry(100, 100, 660, 500)

        # 영상을 띄울 QLabel 생성
        self.image_label = QLabel(self)
        self.image_label.setText("카메라 연결 중...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # TODO: 본인의 ESP32-CAM IP 주소로 수정하세요.
        # 예제 아두이노 CameraWebServer 소스는 대개 81번 포트 /stream을 사용합니다.
        self.esp32_url = "http://192.168.15.208:81/stream"

        # 비디오 스레드 시작
        self.thread = VideoThread(self.esp32_url)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # 시그널을 받아 QLabel의 이미지를 업데이트하는 슬롯 함수
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    # 프로그램 종료 시 스레드 안전하게 닫기
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
