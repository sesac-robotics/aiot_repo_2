"""
HTTP MJPEG / RTSP 영상 연결 및 프레임 수신 관리

- 별도 스레드에서 영상 수신
- 가장 최근 프레임만 저장
- Condition을 이용한 스레드 동기화
- 영상 연결 실패 시 자동 재연결
- Event를 이용한 안전한 종료
"""

import json
import logging
import os
import threading
import time
from collections import deque

import cv2

# OpenCV의 capture options는 프로세스 환경변수입니다. 모든 CameraStream open을
# 직렬화하고 RTSP open 직후 복원하여 HTTP open에 옵션이 섞이지 않게 합니다.
_CAPTURE_OPEN_LOCK = threading.Lock()
_LOGGER = logging.getLogger("uvicorn.error")


class CameraStream:
    def __init__(
        self,
        stream_url: str,
        metrics=None,
        *,
        source_type="http",
        rtsp_transport="tcp",
    ):
        if not stream_url:
            raise ValueError("카메라 스트림 URL이 없습니다.")

        if source_type not in ("http", "rtsp"):
            raise ValueError("camera source_type은 http 또는 rtsp여야 합니다.")
        if source_type == "rtsp" and rtsp_transport not in ("tcp", "udp"):
            raise ValueError("RTSP_TRANSPORT는 tcp 또는 udp여야 합니다.")
        self.source_type = source_type
        self.rtsp_transport = rtsp_transport
        self._connected = False
        self._reconnect_count = 0
        self._last_frame_at = None
        self._frame_read_ms = None
        self._capture_times = deque()
        self._started_at = time.monotonic()
        self._last_log_at = None

        # 서비스가 소유한 계측기만 공유하며 별도 연결은 생성하지 않습니다.
        self.metrics = metrics
        self.stream_url = stream_url
        self.capture = None

        # 카메라에서 받은 최신 프레임
        self.latest_frame = None

        # 카메라에서 수신한 프레임 번호
        self.frame_id = 0

        # 추론 워커가 마지막으로 가져간 프레임 번호
        self.last_read_id = 0

        # 프레임 공유 및 수신 알림을 위한 동기화 객체
        self.condition = threading.Condition()

        # 수신 스레드 종료 신호
        self.stop_event = threading.Event()

        # 카메라 수신 스레드
        self.thread = None

        # connect()와 release()의 동시 실행 방지
        self.state_lock = threading.Lock()

    # 카메라 수신 스레드 시작
    def connect(self):
        with self.state_lock:
            # 이미 수신 스레드가 실행 중이면 중복 생성 방지
            if self.thread is not None and self.thread.is_alive():
                if self.stop_event.is_set():
                    raise RuntimeError("카메라가 종료 중입니다.")
                return

            # 새로운 연결 시작을 위한 상태 초기화
            self.stop_event.clear()
            # 이전에 수신했던 프레임 상태 정리
            with self.condition:
                self.latest_frame = None
                self.frame_id = 0
                self.last_read_id = 0
                self._connected = False
                self._reconnect_count = 0
                self._last_frame_at = None
                self._frame_read_ms = None
                self._capture_times.clear()
                self._started_at = time.monotonic()
                self._last_log_at = None

            # 별도 스레드에서 카메라 연결 및 수신 수행
            self.thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="camera-worker"
            )

            self.thread.start()

        _LOGGER.info("[Camera] source=%s worker_started", self.source_type)

    # 연결 시에만 전송 옵션을 지정하며 프레임은 기존 FFmpeg BGR 형식을 유지합니다.
    def _create_capture(self):
        with _CAPTURE_OPEN_LOCK:
            key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
            previous = os.environ.get(key)
            try:
                if self.source_type == "rtsp":
                    # 기존 비전송 옵션은 보존하고 이 연결의 transport만 교체합니다.
                    options = [
                        item
                        for item in (previous or "").split("|")
                        if item
                        and item.split(";", 1)[0]
                        not in ("rtsp_transport", "rtsp_flags")
                    ]
                    options.append("rtsp_transport;" + self.rtsp_transport)
                    os.environ[key] = "|".join(options)
                return cv2.VideoCapture(
                    self.stream_url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        5000,
                    ],
                )
            finally:
                if self.source_type == "rtsp":
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous

    def get_diagnostics(self):
        """기존 REST 응답은 유지하고 카메라 진단을 별도 로그/내부 조회로 제공합니다."""
        now = time.monotonic()
        with self.condition:
            while self._capture_times and self._capture_times[0] <= now - 5.0:
                self._capture_times.popleft()
            elapsed = min(5.0, max(0.0, now - self._started_at))
            result = {
                "camera_source_type": self.source_type,
                "camera_connected": self._connected,
                "camera_receive_fps": (
                    len(self._capture_times) / elapsed if elapsed else 0.0
                ),
                "frame_id": self.frame_id,
                "frame_read_duration_ms": self._frame_read_ms,
                "reconnect_count": self._reconnect_count,
                "last_frame_age_ms": (
                    None
                    if self._last_frame_at is None
                    else max(0.0, now - self._last_frame_at) * 1000
                ),
            }
        result["inference_fps"] = (
            self.metrics.snapshot()["pose_inference_fps"]
            if self.metrics is not None
            else None
        )
        return result

    def _log_diagnostics(self):
        # 정상 수신 중 최대 초당 한 번만 직렬화합니다. URL과 예외 원문은 기록하지 않습니다.
        now = time.monotonic()
        # if self._last_log_at is None or now - self._last_log_at >= 1.0:
        #     self._last_log_at = now
        #     _LOGGER.info("[CameraMetrics] %s", json.dumps(self.get_diagnostics()))

    # 영상 수신과 자동 재연결
    def _capture_loop(self):
        # 최초 연결은 재연결 횟수에 포함하지 않습니다.
        attempts = 0
        # 종료 요청이 들어오기 전까지 영상 수신 유지
        while not self.stop_event.is_set():
            capture = None

            try:
                with self.condition:
                    self._reconnect_count += int(attempts > 0)
                attempts += 1
                _LOGGER.info(
                    "[Camera] source=%s connecting attempt=%d",
                    self.source_type,
                    attempts,
                )

                capture = self._create_capture()

                if not capture.isOpened():
                    _LOGGER.warning(
                        "[Camera] source=%s connection_failed", self.source_type
                    )

                else:
                    with self.condition:
                        self.capture = capture
                        self._connected = True

                    _LOGGER.info(
                        "[Camera] source=%s connected", self.source_type
                    )

                    # 카메라 프레임 지속 수신
                    while not self.stop_event.is_set():
                        read_started = time.perf_counter()
                        success, frame = capture.read()
                        read_ms = (time.perf_counter() - read_started) * 1000
                        with self.condition:
                            self._frame_read_ms = read_ms

                        if not success or frame is None:
                            _LOGGER.warning(
                                "[Camera] source=%s frame_read_failed",
                                self.source_type,
                            )
                            break

                        # 가장 최근 프레임만 보관
                        with self.condition:
                            # 아직 읽지 않은 슬롯을 덮어쓸 때만 누락으로 집계합니다.
                            if self.metrics is not None:
                                self.metrics.record_event(
                                    "capture",
                                    skipped=int(
                                        self.latest_frame is not None
                                        and self.frame_id > self.last_read_id
                                    ),
                                )
                            self.latest_frame = frame
                            self.frame_id += 1
                            self._last_frame_at = time.monotonic()
                            # 수신 시각만 최근 5초 보관하며 영상 큐는 생성하지 않습니다.
                            while (
                                self._capture_times
                                and self._capture_times[0]
                                <= self._last_frame_at - 5.0
                            ):
                                self._capture_times.popleft()
                            self._capture_times.append(self._last_frame_at)

                            # 새로운 프레임 도착 알림
                            self.condition.notify_all()
                        self._log_diagnostics()

            except Exception as e:
                # OpenCV 예외에는 인증 URL이 포함될 수 있어 타입만 남깁니다.
                _LOGGER.warning(
                    "[Camera] source=%s stream_error=%s",
                    self.source_type,
                    type(e).__name__,
                )

            finally:
                # 이전 연결 정리
                with self.condition:
                    self.latest_frame = None
                    self.capture = None
                    self._connected = False
                    self.condition.notify_all()

                if capture is not None:
                    try:
                        capture.release()
                    except Exception as error:
                        _LOGGER.warning(
                            "[Camera] release_error=%s", type(error).__name__
                        )
                self._log_diagnostics()

            # 종료 요청이 없다면 1초 후 재연결 시도
            if not self.stop_event.is_set():
                _LOGGER.info(
                    "[Camera] source=%s reconnect_in_sec=1", self.source_type
                )
                self.stop_event.wait(1.0)

        with self.condition:
            self.condition.notify_all()

        _LOGGER.info("[Camera] source=%s worker_stopped", self.source_type)

    # 추론 워커에 최신 프레임 전달
    def read(self):
        # connect()가 호출되지 않은 경우
        if self.thread is None:
            raise RuntimeError("카메라 수신 스레드가 시작되지 않았습니다.")

        with self.condition:
            # 새로운 프레임이 도착할 때까지 최대 1초 대기
            ready = self.condition.wait_for(
                lambda: (
                    (
                        self.latest_frame is not None
                        and self.frame_id > self.last_read_id
                    )
                    or self.stop_event.is_set()
                ),
                timeout=1.0,
            )

            # 종료되었거나 새로운 프레임이 없는 경우
            if (
                not ready
                or self.stop_event.is_set()
                or self.latest_frame is None
            ):
                return None

            # 마지막으로 가져간 프레임 번호 갱신
            self.last_read_id = self.frame_id

            # 수신 스레드와 추론 스레드의 프레임 분리
            return self.latest_frame.copy()

    # 수신 스레드 종료
    def release(self):
        # 카메라 수신 스레드에 종료 신호 전달
        with self.state_lock:
            self.stop_event.set()
            worker = self.thread

        # 프레임을 기다리고 있는 스레드 깨우기
        with self.condition:
            self.condition.notify_all()

        # 수신 스레드가 종료될 때까지 대기
        if worker is not None:
            worker.join(timeout=7.0)

            if worker.is_alive():
                _LOGGER.warning(
                    "[Camera] source=%s worker_stop_delayed", self.source_type
                )
            else:
                with self.state_lock:
                    if self.thread is worker:
                        self.thread = None

        # VideoCapture 해제는 수신 스레드의 finally에서 수행
        _LOGGER.info("[Camera] source=%s stop_requested", self.source_type)
