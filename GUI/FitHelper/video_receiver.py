import requests
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class VideoReceiver(QThread):
    frame_received = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    def __init__(self, stream_url, parent=None):
        super().__init__(parent)
        self.stream_url = stream_url
        self._response = None

    def run(self):
        buffer = bytearray()
        try:
            with requests.get(self.stream_url, stream=True, timeout=(5, 30)) as response:
                self._response = response
                response.raise_for_status()
                self.status_changed.emit("서버 연결 성공 · 운동 시작")

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

        except (requests.RequestException, AttributeError, TypeError):
            pass
        finally:
            self._response = None
            self.status_changed.emit("영상 수신 종료")

    def stop(self):
        self.requestInterruption()
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass