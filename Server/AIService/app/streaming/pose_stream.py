
import cv2
import time
import torch
import threading
import json
import logging
import math

from copy import deepcopy
from datetime import datetime, timezone
from collections import deque

from app.metrics import PipelineMetrics
from app.camera.stream import CameraStream
from app.streaming.output import encode_jpeg
from app.analysis.keypoints import extract_mediapipe_keypoints


class PoseStreamService:
    def __init__(self, stream_url, pose_estimator, model_name, metrics_log_interval=5.0):
        # 잘못된 간격은 워커 시작 전에 검증합니다.
        if not math.isfinite(metrics_log_interval) or metrics_log_interval <= 0:
            raise ValueError("METRICS_LOG_INTERVAL_SECONDS는 양수여야 합니다.")
        self.metrics = PipelineMetrics()
        self.metrics_log_interval = metrics_log_interval
        self.metrics_worker = None
        self.camera = CameraStream(stream_url, metrics=self.metrics)
        if model_name == "mediapipe":
            pose_estimator.metrics = self.metrics
        self.pose_estimator = pose_estimator
        self.model_name = model_name

        # 모든 클라이언트가 공유할 최신 추론 영상
        self.latest_jpeg = None

        # 추론이 완료된 프레임 번호
        self.frame_id = 0

        # 최근 추론 결과를 JSON 형태로 보관
        self.latest_pose = None

        # 클라이언트 간 최신 프레임 공유 및 동기화
        self.condition = threading.Condition()

        # 추론 워커 종료 관리
        self.stop_event = threading.Event()
        self.worker = None

    def start(self):
        # 추론 워커 중복 실행 방지
        if self.worker is not None and self.worker.is_alive():
            return

        self.stop_event.clear()

        # 이전에 처리했던 영상 결과 초기화
        with self.condition:
            self.latest_jpeg = None
            self.latest_pose = None
            self.frame_id = 0

        # ESP32-CAM 수신 스레드 실행
        self.camera.connect()

        # 단일 Pose 추론 워커 생성
        self.worker = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="pose-inference-worker"
        )

        self.worker.start()

        # 추론 정지/카메라 단절 중에도 로그를 남기는 계측 전용 워커입니다.
        self.metrics_worker = threading.Thread(
            target=self._metrics_loop, daemon=True, name="metrics-logger"
        )
        self.metrics_worker.start()

        print("[AI Service] Pose 추론 워커 시작")

    # CUDA 연산 동기화
    def _sync_inferecnce(self):
        # CUDA 비동기 연산 완료까지 측정
        if (
            self.model_name == "yolo"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()
        elif self.model_name == 'mediapipe':
            '''
            detect_for_video()는 동기식 함수
            함수가 반환되면 해당 프레임의 추론이 완료됨
            MediaPipe GPU delegate는 PyTorch CUDA와 별도이기 때문에
            torch.cuda.synchronize()를 호출하지 않음
            '''
            pass

    def _inference_loop(self):
        # 최근 30프레임 기준 성능 측정
        frame_times = deque(maxlen=31)

        while not self.stop_event.is_set():
            try:
                # 1. ESP32-CAM 최신 프레임 가져오기
                frame = self.camera.read()

                if frame is None:
                    continue

                # 2. Pose Estimation
                self._sync_inferecnce()
                infer_start = time.perf_counter()

                annotated_frame, result = (
                    self.pose_estimator.predict(frame)
                )

                # MediaPipe 추론 결과에서 33개 관절 추출
                keypoints = extract_mediapipe_keypoints(result)

                self._sync_inferecnce()
                infer_ms = (
                    time.perf_counter() - infer_start
                ) * 1000

                # 전체 추론 단계에는 전처리/그리기/관절 추출도 포함됩니다.
                self.metrics.record_duration("inference", infer_ms)
                self.metrics.record_event("inference")

                # 3. 최근 처리 FPS 계산
                frame_times.append(time.perf_counter())

                if len(frame_times) >= 2:
                    duration = (
                        frame_times[-1] - frame_times[0]
                    )
                    fps = (
                        (len(frame_times) - 1) / duration
                        if duration > 0 else 0.0
                    )
                else:
                    fps = 0.0

                # 4. 영상에 성능 정보 표시
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | "
                    f"{self.model_name}: {infer_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # 5. JPEG 인코딩
                encode_start = time.perf_counter()

                jpeg_bytes = encode_jpeg(annotated_frame)

                encode_ms = (
                    time.perf_counter() - encode_start
                ) * 1000

                # encode_jpeg의 압축 및 bytes 복사 시간을 함께 측정합니다.
                self.metrics.record_duration("jpeg_encoding", encode_ms)

                if jpeg_bytes is None:
                    continue

                # 6. 최신 추론 결과 공유
                with self.condition:
                    self.latest_jpeg = jpeg_bytes
                    self.frame_id += 1

                    # API 반환용 MediaPipe 추론 데이터
                    self.latest_pose = {
                        "frame_id": self.frame_id,
                        "processed_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "model": self.model_name,
                        "person_detected": bool(keypoints),
                        "keypoints": keypoints
                    }

                    # 대기 중인 모든 클라이언트에 알림
                    self.condition.notify_all()

                # 성공적으로 게시한 결과만 전체 처리 FPS에 반영합니다.
                self.metrics.record_event("processed")

            except Exception as e:
                print(f"[AI Service] 추론 오류: {e}")
                self.stop_event.wait(0.2)

        with self.condition:
            self.condition.notify_all()

        print("[AI Service] 추론 워커 종료")

    def get_metrics(self):
        # 조회는 공유 표본만 읽으며 카메라/모델을 실행하지 않습니다.
        return self.metrics.snapshot()

    def _metrics_loop(self):
        # Event.wait를 사용해 종료 요청에 즉시 반응합니다.
        logger = logging.getLogger("uvicorn.error")
        while not self.stop_event.wait(self.metrics_log_interval):
            logger.info("[Performance] %s", json.dumps(self.get_metrics()))

    # 여러 클라이언트에 영상 전송
    def generate_frames(self):
        # 각 클라이언트의 마지막 전송 프레임 번호
        last_sent_id = 0

        while not self.stop_event.is_set():
            with self.condition:
                # 새로운 추론 결과가 생성될 때까지 대기
                ready = self.condition.wait_for(
                    lambda: (
                        self.frame_id > last_sent_id
                        or self.stop_event.is_set()
                    ),
                    timeout=1.0
                )

                if self.stop_event.is_set():
                    break

                if not ready or self.latest_jpeg is None:
                    continue

                # 최신 JPEG를 참조하고 번호 갱신
                jpeg_bytes = self.latest_jpeg
                last_sent_id = self.frame_id

            # Lock 밖에서 클라이언트에 영상 전송
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + (
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode("ascii")
                + jpeg_bytes
                + b"\r\n"
            )

    def get_latest_pose(self):
        with self.condition:
            if self.latest_pose is None:
                return None

            return deepcopy(self.latest_pose)

    def stop(self):
        # 추론 워커 종료 요청
        self.stop_event.set()

        # 프레임을 기다리는 클라이언트 깨우기
        with self.condition:
            self.condition.notify_all()

        # 로거도 같은 종료 신호를 사용하며 모델 객체에는 접근하지 않습니다.
        if self.metrics_worker is not None:
            self.metrics_worker.join(timeout=1.0)

        # 카메라 수신 스레드 종료
        self.camera.release()

        # 추론 워커 종료 대기
        if self.worker is not None:
            self.worker.join(timeout=3.0)

            if self.worker.is_alive():
                print("[AI Service] 추론 워커 종료 지연")
            else:
                self.worker = None

        if self.worker is None or not self.worker.is_alive():
            close_fn = getattr(self.pose_estimator, 'close', None)

            if callable(close_fn):
                close_fn()

        print("[AI Service] 종료 요청 완료")
