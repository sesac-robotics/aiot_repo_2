"""
ESP32-CAM 영상 연결 및 프레임 수신 관리

- 별도 스레드에서 영상 수신
- 가장 최근 프레임만 저장
- Condition을 이용한 스레드 동기화
- 영상 연결 실패 시 자동 재연결
- Event를 이용한 안전한 종료
"""

import cv2
import threading

class CameraStream:
    def __init__(self, stream_url: str):
        if not stream_url:
            raise ValueError("카메라 스트림 URL이 없습니다.")

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

            # 별도 스레드에서 카메라 연결 및 수신 수행
            self.thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="camera-worker"
            )

            self.thread.start()

        print("[Camera] 수신 스레드 시작")

    # OpenCV 연결 생성
    def _create_capture(self):
        # FFmpeg 연결 및 프레임 읽기 타임아웃 설정
        return cv2.VideoCapture(
            self.stream_url,
            # OpenCV에서 FFMPEG 백엔드를 명시적으로 지정
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000
            ]
        )

    # 영상 수신과 자동 재연결
    def _capture_loop(self):
        # 종료 요청이 들어오기 전까지 영상 수신 유지
        while not self.stop_event.is_set():
            capture = None

            try:
                print(f"[Camera] 연결 시도: {self.stream_url}")

                capture = self._create_capture()

                if not capture.isOpened():
                    print("[Camera] 연결 실패")

                else:
                    with self.condition:
                        self.capture = capture

                    print(f"[Camera] 연결 성공: {self.stream_url}")

                    # 카메라 프레임 지속 수신
                    while not self.stop_event.is_set():
                        success, frame = capture.read()

                        if not success or frame is None:
                            print("[Camera] 프레임 수신 실패")
                            break

                        # 가장 최근 프레임만 보관
                        with self.condition:
                            self.latest_frame = frame
                            self.frame_id += 1

                            # 새로운 프레임 도착 알림
                            self.condition.notify_all()

            except Exception as e:
                print(f"[Camera] 스트리밍 오류: {e}")

            finally:
                # 이전 연결 정리
                with self.condition:
                    self.latest_frame = None
                    self.capture = None
                    self.condition.notify_all()

                if capture is not None:
                    capture.release()

            # 종료 요청이 없다면 1초 후 재연결 시도
            if not self.stop_event.is_set():
                print("[Camera] 1초 후 재연결 시도")
                self.stop_event.wait(1.0)

        with self.condition:
            self.condition.notify_all()

        print("[Camera] 수신 스레드 종료")

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
                timeout=1.0
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
                print("[Camera] 수신 스레드 종료 지연")
            else:
                with self.state_lock:
                    if self.thread is worker:
                        self.thread = None

        # VideoCapture 해제는 수신 스레드의 finally에서 수행
        print("[Camera] 종료 요청 완료")
