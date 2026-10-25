"""카메라 수신, 추론 워커, 운동 분석, 자세 판정과 JPEG 출력을 조립합니다. 최신 영상 경로와 완료 운동 이벤트 경로의 수명·전달 책임을 구분합니다."""

import json
import logging
import math
import threading
import time
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

import cv2
import torch

from app.analysis.exercise import ExerciseAnalysisBridge
from app.analysis.frame_debug import FrameDebug
from app.analysis.keypoints import extract_mediapipe_keypoints
from app.analysis.registry import RegisteredExerciseAnalyzer
from app.analysis.shoulder_debug import ShoulderPressDebug, session_fields
from app.camera.stream import CameraStream
from app.metrics import PipelineMetrics
from app.streaming.output import encode_jpeg


class PoseStreamService:
    def __init__(
        self,
        stream_url,
        pose_estimator,
        model_name,
        metrics_log_interval=5.0,
        *,
        estimator_factory=None,
        session_manager=None,
        exercise_analyzer=None,
        camera_source_type="http",
        rtsp_transport="tcp",
        posture_service=None,
    ):
        # 잘못된 간격은 워커 시작 전에 검증합니다.
        if not math.isfinite(metrics_log_interval) or metrics_log_interval <= 0:
            raise ValueError("METRICS_LOG_INTERVAL_SECONDS는 양수여야 합니다.")
        # 실제 서버는 팩터리를 전달하여 모델 생명주기를 추론 워커에 귀속시킵니다.
        if (pose_estimator is None) == (estimator_factory is None):
            raise ValueError(
                "pose_estimator와 estimator_factory 중 하나만 지정해야 합니다."
            )
        self.estimator_factory = estimator_factory
        self._lifecycle_lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_error = None
        self.metrics = PipelineMetrics()
        self.metrics_log_interval = metrics_log_interval
        self.metrics_worker = None
        # 입력만 선택하며 이후 추론/분석/출력 경로는 동일합니다.
        self.camera = CameraStream(
            stream_url,
            metrics=self.metrics,
            source_type=camera_source_type,
            rtsp_transport=rtsp_transport,
        )
        if model_name == "mediapipe" and pose_estimator is not None:
            pose_estimator.metrics = self.metrics
        self.pose_estimator = pose_estimator
        self.model_name = model_name
        # FastAPI와 같은 객체를 주입받으며 별도 관리자/분석 스레드는 생성하지 않습니다.
        self.session_manager = session_manager
        # 자세 추론은 별도 단일 슬롯 워커에 맡기고 기존 Rule-based 계산을 기다리게 하지 않습니다.
        self.posture_service = posture_service
        if posture_service is not None:
            posture_service.pipeline_metrics = self.metrics
        self._shoulder_debug = ShoulderPressDebug()
        self._frame_debug = FrameDebug()
        self.exercise_analysis = (
            ExerciseAnalysisBridge(
                session_manager,
                (
                    exercise_analyzer
                    if exercise_analyzer is not None
                    else RegisteredExerciseAnalyzer()
                ),
            )
            if session_manager is not None
            else None
        )

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
        # 동시 start/stop에서도 워커를 두 번 만들지 않습니다.
        with self._lifecycle_lock:
            if self.worker is not None and self.worker.is_alive():
                if self.stop_event.is_set():
                    raise RuntimeError("추론 워커가 아직 종료 중입니다.")
                return
            if self.worker is not None and self.estimator_factory is None:
                raise RuntimeError(
                    "종료한 모델 재사용 대신 새 estimator_factory를 지정하세요."
                )
            # 종료한 이전 로거가 새 실행의 Event.clear를 보지 않도록 먼저 회수합니다.
            if self.metrics_worker is not None:
                self.metrics_worker.join(timeout=1.0)
                if self.metrics_worker.is_alive():
                    raise RuntimeError("성능 로그 워커가 아직 종료 중입니다.")
            self.stop_event.clear()
            self._ready.clear()
            self._startup_error = None
            with self.condition:
                self.latest_jpeg = None
                self.latest_pose = None
                self.frame_id = 0
            self.worker = threading.Thread(
                target=self._run_worker,
                daemon=True,
                name="pose-inference-worker",
            )
            self.worker.start()
            # GPU 초기화 실패를 ASGI startup 실패로 전달합니다.
            if not self._ready.wait(timeout=30.0):
                self.stop_event.set()
                raise TimeoutError(
                    "Pose 모델 초기화가 30초 안에 완료되지 않았습니다."
                )
            if self._startup_error is not None:
                self.worker.join(timeout=10.0)
                raise RuntimeError(
                    "Pose 추론 워커 초기화 실패"
                ) from self._startup_error
            self.metrics_worker = threading.Thread(
                target=self._metrics_loop, daemon=True, name="metrics-logger"
            )
            self.metrics_worker.start()
        print("[AI Service] Pose 추론 워커 시작")

    def _run_worker(self):
        # 생성/추론/close를 모두 같은 Python 워커에서 수행합니다.
        # MediaPipe 내부 GL 스레드의 Tensor 재사용 문제를 고치는 코드는 아닙니다.
        estimator = None
        try:
            estimator = (
                self.estimator_factory()
                if self.estimator_factory is not None
                else self.pose_estimator
            )
            self.pose_estimator = estimator
            if self.model_name == "mediapipe":
                estimator.metrics = self.metrics
            if not self.stop_event.is_set():
                self.camera.connect()
            self._ready.set()
            self._inference_loop()
        except BaseException as error:
            if not self._ready.is_set():
                self._startup_error = error
            logging.getLogger("uvicorn.error").exception("Pose 워커 실패")
        finally:
            self.stop_event.set()
            try:
                self.camera.release()
            finally:
                try:
                    close_fn = getattr(estimator, "close", None)
                    if callable(close_fn):
                        close_fn()
                finally:
                    # 초기화 실패나 close 예외가 있어도 시작/구독 대기를 해제합니다.
                    self._ready.set()
                    with self.condition:
                        self.condition.notify_all()

    # CUDA 연산 동기화
    def _sync_inferecnce(self):
        # CUDA 비동기 연산 완료까지 측정
        if self.model_name == "yolo" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.model_name == "mediapipe":
            """
            detect_for_video()는 동기식 함수
            함수가 반환되면 해당 프레임의 추론이 완료됨
            MediaPipe GPU delegate는 PyTorch CUDA와 별도이기 때문에
            torch.cuda.synchronize()를 호출하지 않음
            """
            pass

    def _debug_analysis_pipeline(self, stage):
        # 분석기 호출/프레임이 없어도 기존 워커의 대기 루프에서 1초 간격으로 상태를 읽습니다.
        self._shoulder_debug.emit(
            "pipeline",
            lambda: {
                "stage": stage,
                "pose_model": self.model_name,
                "bridge_configured": self.exercise_analysis is not None,
                "mediapipe_selected": self.model_name == "mediapipe",
                **session_fields(self.session_manager),
            },
        )

    def _inference_loop(self):
        # 최근 30프레임 기준 성능 측정
        frame_times = deque(maxlen=31)
        # 워커 재시작 시에도 이전 부분 동작을 이어서 처리하지 않습니다.
        if self.exercise_analysis is not None:
            try:
                self.exercise_analysis.reset()
            except Exception:
                logging.getLogger("uvicorn.error").exception(
                    "운동 분석기 시작 초기화 실패"
                )

        while not self.stop_event.is_set():
            try:
                self._debug_analysis_pipeline("waiting_for_frame")
                # 1. ESP32-CAM 최신 프레임 가져오기
                frame = self.camera.read()

                if frame is None:
                    self._debug_analysis_pipeline("no_camera_frame")
                    if self.exercise_analysis is not None:
                        self.exercise_analysis._squat_debug.pipeline(
                            self.session_manager, "no_camera_frame"
                        )
                    continue

                # 문맥만 잠깐 읽고 잠금을 해제한 뒤 GPU 추론을 실행합니다.
                analysis_context = (
                    self.session_manager.get_analysis_context()
                    if self.session_manager is not None
                    and self.model_name == "mediapipe"
                    else None
                )
                # 분석 입력 시각만 기록합니다. GPU 호출/전송 지연을 동작 시간으로 더하지 않습니다.
                analysis_timestamp = time.monotonic()

                # 2. Pose Estimation
                self._sync_inferecnce()
                infer_start = time.perf_counter()

                annotated_frame, result = self.pose_estimator.predict(frame)

                # MediaPipe 추론 결과에서 33개 관절 추출
                keypoints = extract_mediapipe_keypoints(result)
                # JPEG 게시 번호와 분석 번호를 동일하게 예약합니다. 인코딩 실패 시 게시/분석하지 않습니다.
                output_frame_id = self.frame_id + 1
                frame_trace = None
                if self._frame_debug.enabled and self.model_name == "mediapipe":
                    from app.analysis.exercises.ytw_raise import YTWRaiseConfig

                    analyzer = (
                        self.exercise_analysis.analyzer
                        if self.exercise_analysis
                        else None
                    )
                    analyzer = getattr(analyzer, "_analyzer", analyzer)
                    config = getattr(analyzer, "config", None)
                    config_source = (
                        "active_ytw_analyzer"
                        if isinstance(config, YTWRaiseConfig)
                        else "ytw_defaults"
                    )
                    config = (
                        config
                        if isinstance(config, YTWRaiseConfig)
                        else YTWRaiseConfig()
                    )
                    frame_trace = self._frame_debug.capture(
                        result,
                        keypoints,
                        frame_id=output_frame_id,
                        timestamp=analysis_timestamp,
                        size=(frame.shape[1], frame.shape[0]),
                        estimator=self.pose_estimator,
                        config=config,
                        config_source=config_source,
                    )

                self._sync_inferecnce()
                infer_ms = (time.perf_counter() - infer_start) * 1000

                # 전체 추론 단계에는 전처리/그리기/관절 추출도 포함됩니다.
                self.metrics.record_duration("inference", infer_ms)
                self.metrics.record_event("inference")

                # 3. 최근 처리 FPS 계산
                frame_times.append(time.perf_counter())

                if len(frame_times) >= 2:
                    duration = frame_times[-1] - frame_times[0]
                    fps = (
                        (len(frame_times) - 1) / duration
                        if duration > 0
                        else 0.0
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
                    2,
                )

                # 5. JPEG 인코딩
                self._frame_debug.overlay(annotated_frame, frame_trace)
                encode_start = time.perf_counter()

                jpeg_bytes = encode_jpeg(annotated_frame)

                encode_ms = (time.perf_counter() - encode_start) * 1000

                # encode_jpeg의 압축 및 bytes 복사 시간을 함께 측정합니다.
                self.metrics.record_duration("jpeg_encoding", encode_ms)

                if jpeg_bytes is None:
                    self._debug_analysis_pipeline("jpeg_encoding_failed")
                    continue

                # 6. 최신 추론 결과 공유
                with self.condition:
                    self.latest_jpeg = jpeg_bytes
                    self.frame_id = output_frame_id

                    # API 반환용 MediaPipe 추론 데이터
                    self.latest_pose = {
                        "frame_id": self.frame_id,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "model": self.model_name,
                        "person_detected": bool(keypoints),
                        "keypoints": keypoints,
                    }

                    # 대기 중인 모든 클라이언트에 알림
                    self.condition.notify_all()

                # 성공적으로 게시한 결과만 전체 처리 FPS에 반영합니다.
                self.metrics.record_event("processed")
                self._frame_debug.publish(frame_trace)
                # JPEG/API 게시 후 같은 워커에서 분석하며 오래된 결과는 반영하지 않습니다.
                if (
                    self.exercise_analysis is not None
                    and self.model_name == "mediapipe"
                ):
                    self.exercise_analysis.process(
                        analysis_context,
                        keypoints,
                        frame_size=(frame.shape[1], frame.shape[0]),
                        timestamp=analysis_timestamp,
                        frame_id=output_frame_id,
                        frame_trace=frame_trace,
                    )
                if (
                    self.posture_service is not None
                    and self.model_name == "mediapipe"
                ):
                    self.posture_service.submit(
                        analysis_context,
                        keypoints,
                        frame_id=output_frame_id,
                        timestamp=analysis_timestamp,
                        frame_size=(frame.shape[1], frame.shape[0]),
                        analysis_started_at=infer_start,
                        analysis_result=getattr(
                            self.exercise_analysis, "last_result", None
                        ),
                    )

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
        # while not self.stop_event.wait(self.metrics_log_interval):
        # logger.info("[Performance] %s", json.dumps(self.get_metrics()))

    # 여러 클라이언트에 영상 전송
    def generate_frames(self):
        # 각 클라이언트의 마지막 전송 프레임 번호
        last_sent_id = 0

        while not self.stop_event.is_set():
            with self.condition:
                # 새로운 추론 결과가 생성될 때까지 대기
                ready = self.condition.wait_for(
                    lambda: (
                        self.frame_id > last_sent_id or self.stop_event.is_set()
                    ),
                    timeout=1.0,
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
                + (f"Content-Length: {len(jpeg_bytes)}\r\n\r\n").encode("ascii")
                + jpeg_bytes
                + b"\r\n"
            )

    def get_latest_pose(self):
        with self.condition:
            if self.latest_pose is None:
                return None

            return deepcopy(self.latest_pose)

    def stop(self):
        with self._lifecycle_lock:
            self.stop_event.set()
            # read 대기를 깨우되 GPU 객체는 여기서 close하지 않습니다.
            self.camera.stop_event.set()
            with self.camera.condition:
                self.camera.condition.notify_all()
            with self.condition:
                self.condition.notify_all()
            if self.metrics_worker is not None:
                self.metrics_worker.join(timeout=1.0)
            if self.worker is not None:
                self.worker.join(timeout=10.0)
                if self.worker.is_alive():
                    # 늦은 추론과 close의 경합을 피하고 재시작도 차단합니다.
                    logging.getLogger("uvicorn.error").error(
                        "Pose 워커 종료 지연: GPU 정리는 소유 워커 완료 시 수행됩니다."
                    )
        print("[AI Service] 종료 요청 완료")
