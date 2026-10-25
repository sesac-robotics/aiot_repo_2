import requests
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class VideoReceiver(QThread):
    frame_received = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    def __init__(self, stream_url, parent=None, access_token=None):
        super().__init__(parent)
        self.stream_url = stream_url
        self.access_token = access_token
        self._response = None

    def run(self):
        if not self.access_token:
            self.status_changed.emit("로그인이 필요합니다")
            return

        buffer = bytearray()
        terminal_status = None
        try:
            with requests.get(
                self.stream_url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                stream=True,
                timeout=(5, 30),
            ) as response:
                self._response = response
                if response.status_code == 401:
                    terminal_status = "로그인 토큰이 만료되었습니다. 다시 로그인해 주세요"
                    self.status_changed.emit(terminal_status)
                    return
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

        except requests.RequestException as exc:
            if not self.isInterruptionRequested():
                terminal_status = f"서버 연결 오류: {exc}"
                self.status_changed.emit(terminal_status)
        except (AttributeError, TypeError) as exc:
            terminal_status = f"영상 처리 오류: {exc}"
            self.status_changed.emit(terminal_status)
        finally:
            self._response = None
            if terminal_status is None:
                self.status_changed.emit("영상 수신 종료")

    def stop(self):
        self.requestInterruption()
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
