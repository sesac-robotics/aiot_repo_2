"""팀원 ESP32 수신기를 복사해 명시적인 ESP32·웹캠 선택을 추가합니다.

CameraReader 출처: Server/PostureAnalysis/posture-error-training/live_app/engine.py.
원본의 재접속·HTTP capture 대체 수신을 보존합니다. 카메라 생성만으로 촬영하지 않습니다.
"""

import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

import cv2
import numpy as np


class CameraReader:
    def __init__(self, source):
        self.source = source
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.epoch = 0
        self.error = ""
        self.connected = False
        self.capture_fps = 0.0
        self.reads = deque(maxlen=40)
        self.transport = "stream"
        self.capture_url = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                raise RuntimeError(
                    "카메라 종료 대기 중입니다. 추가 카메라를 열지 않습니다."
                )

    def snapshot(self):
        with self.lock:
            return (
                self.seq,
                self.epoch,
                self.latest,
                self.connected,
                self.error,
                self.capture_fps,
            )

    def snapshots(self, url):
        self.transport = "snapshot"
        while not self.stop_event.is_set():
            begin = time.monotonic()
            with urlopen(url, timeout=3) as response:
                payload = response.read(5_000_001)
            if len(payload) > 5_000_000:
                raise ConnectionError("카메라 사진이 너무 큽니다.")
            frame = cv2.imdecode(
                np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ConnectionError("카메라 JPEG 사진을 읽을 수 없습니다.")
            now = time.monotonic()
            self.reads.append(now)
            with self.lock:
                self.latest = (frame, now)
                self.seq += 1
                self.connected = True
                self.error = ""
                self.capture_fps = (
                    (len(self.reads) - 1) / (self.reads[-1] - self.reads[0])
                    if len(self.reads) > 1
                    else 0.0
                )
            self.stop_event.wait(max(0, 1 / 6 - (time.monotonic() - begin)))

    def run(self):
        local = Path(self.source).is_file()
        camera = self.source.isdigit()
        attempt = 0
        while not self.stop_event.is_set():
            cap = None
            try:
                with self.lock:
                    if not self.error:
                        self.error = "카메라 연결 중…"
                    self.connected = False
                    self.latest = None
                    self.epoch += 1
                if self.capture_url or urlparse(self.source).path == "/capture":
                    self.snapshots(self.capture_url or self.source)
                    break
                cap = (
                    cv2.VideoCapture(int(self.source))
                    if camera
                    else cv2.VideoCapture(
                        self.source,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_N_THREADS,
                            1,
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                            4000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                            3000,
                        ],
                    )
                )
                if not cap.isOpened():
                    parsed = urlparse(self.source)
                    if (
                        parsed.scheme in ["http", "https"]
                        and parsed.port == 81
                        and parsed.path == "/stream"
                    ):
                        self.capture_url = urlunparse(
                            (
                                parsed.scheme,
                                parsed.hostname + ":80",
                                "/capture",
                                "",
                                "",
                                "",
                            )
                        )
                        raise ConnectionError(
                            "영상 스트림 응답 없음 · 사진 연속 수신으로 전환 중…"
                        )
                    raise ConnectionError(
                        "카메라에 연결할 수 없습니다. 전원·Wi-Fi·주소를 확인하세요."
                    )
                self.reads.clear()
                decoded = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                pace = 1 / max(fps, 1) if local else 0.0
                next_frame = time.monotonic()
                attempt = 0
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        if local and decoded:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            with self.lock:
                                self.epoch += 1
                                self.latest = None
                                self.connected = False
                            self.reads.clear()
                            continue
                        raise ConnectionError(
                            "영상 수신이 끊겼습니다. 재연결 중…"
                        )
                    decoded += 1
                    now = time.monotonic()
                    self.reads.append(now)
                    with self.lock:
                        self.latest = (frame, now)
                        self.seq += 1
                        self.connected = True
                        self.error = ""
                        self.capture_fps = (
                            (len(self.reads) - 1)
                            / (self.reads[-1] - self.reads[0])
                            if len(self.reads) > 1
                            else 0.0
                        )
                    if local:
                        next_frame += pace
                        self.stop_event.wait(
                            max(0, next_frame - time.monotonic())
                        )
            except Exception as e:
                with self.lock:
                    self.connected = False
                    self.latest = None
                    self.error = str(e)
                    self.epoch += 1
                attempt += 1
                if self.stop_event.wait(min(attempt, 4)):
                    break
            finally:
                if cap is not None:
                    cap.release()
        with self.lock:
            self.connected = False
            self.latest = None


class ChoiceCamera:
    """자동 모드에서 ESP32 연결이 제한 시간 안에 성립하지 않으면 웹캠으로 전환합니다."""

    def __init__(self, options, factory=CameraReader, timeout=10):
        self.options = options
        self.factory = factory
        self.timeout = timeout
        self.reader = None
        self.started = None
        self.selected = None
        self.offset = 0

    def start(self):
        self.selected = (
            "webcam" if self.options["camera"] == "webcam" else "esp32"
        )
        self._open()

    def _open(self):
        self.reader = self.factory(str(self.options[self.selected]))
        self.started = time.monotonic()
        self.reader.start()

    def snapshot(self):
        seq, epoch, latest, connected, error, fps = self.reader.snapshot()
        if connected:
            self.started = time.monotonic()
        if (
            self.options["camera"] == "auto"
            and self.selected == "esp32"
            and not connected
            and time.monotonic() - self.started >= self.timeout
        ):
            self.reader.stop()
            self.offset += epoch + 1
            self.selected = "webcam"
            self._open()
            seq, epoch, latest, connected, error, fps = self.reader.snapshot()
        return seq, self.offset + epoch, latest, connected, error, fps

    def stop(self):
        if self.reader:
            self.reader.stop()
