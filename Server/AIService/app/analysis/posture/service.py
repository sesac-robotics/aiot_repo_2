"""단일 슬롯 워커: 느린 자세 모델이 영상/Reps/REST를 기다리게 하지 않습니다."""

import json
import logging
import os
import threading
import time
from collections import deque
from copy import deepcopy

from app.analysis.posture_state import posture_state as empty_result

LOGGER = logging.getLogger("uvicorn.error")


class PostureService:
    def __init__(
        self,
        manager,
        root,
        *,
        architecture="cnn",
        device="cpu",
        predictor=None,
        mode="model",
        camera_view="unknown",
        rule_config=None,
        camera_views=None,
    ):
        self.manager = manager
        self.mode = mode
        if predictor is not None:
            self.predictor = predictor
        elif mode == "rule_based":
            from .rule_based import RuleBasedPredictor

            self.predictor = RuleBasedPredictor(
                config=rule_config,
                camera_view=camera_view,
                camera_views=camera_views,
            )
        elif mode == "model":
            from .predictor import PosturePredictor

            self.predictor = PosturePredictor(root, architecture, device)
        else:
            raise ValueError("지원하지 않는 자세 평가 모드입니다.")
        self.manager.enable_posture_analysis()
        self.pipeline_metrics = None
        self._condition = threading.Condition()
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pending = None
        self._last_key = None
        self._times = deque(maxlen=300)
        self._events = deque()
        self._started = time.monotonic()
        self._last_log = -float("inf")

    @classmethod
    def from_env(cls, manager):
        # 명시 mode가 우선하고, 미설정 배포는 기존 ENABLED 설정을 그대로 따릅니다.
        explicit = os.getenv("POSTURE_EVALUATION_MODE")
        enabled = os.getenv("POSTURE_MODEL_ENABLED", "false").strip().lower()
        if explicit is None and enabled not in ("true", "false", "1", "0"):
            LOGGER.warning("[Posture] invalid POSTURE_MODEL_ENABLED; disabled")
            return None
        mode = (
            explicit.strip().lower()
            if explicit is not None
            else "model" if enabled in ("true", "1") else "off"
        )
        if mode == "off":
            return None
        try:
            if mode not in ("model", "rule_based"):
                raise ValueError(
                    "POSTURE_EVALUATION_MODE는 rule_based/model/off여야 합니다."
                )
            config = None
            camera_views = None
            if mode == "rule_based":
                from .rule_based import RuleConfig

                config = RuleConfig(
                    **json.loads(os.getenv("POSTURE_RULE_CONFIG", "{}"))
                )
                camera_views = json.loads(
                    os.getenv("POSTURE_CAMERA_VIEWS", "{}")
                )
            return cls(
                manager,
                os.getenv(
                    "POSTURE_MODEL_DIR",
                    "~/Desktop/posture_classification_handoff_v2_feedback",
                ),
                architecture=os.getenv("POSTURE_MODEL_ARCHITECTURE", "cnn"),
                device=os.getenv("POSTURE_MODEL_DEVICE", "cpu"),
                mode=mode,
                camera_view=os.getenv("POSTURE_CAMERA_VIEW", "unknown")
                .strip()
                .lower(),
                rule_config=config,
                camera_views=camera_views,
            )
        except Exception as error:
            manager.enable_posture_analysis(
                unavailable_reason=type(error).__name__
            )
            LOGGER.warning(
                "[Posture] configuration unavailable reason=%s",
                type(error).__name__,
            )
            return None

    def start(self):
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError("자세 워커가 종료 중입니다.")
                return
            self._stop.clear()
            self._last_key = None
            with self._condition:
                self._pending = None
            self._thread = threading.Thread(
                target=self._run, name="posture-worker", daemon=True
            )
            self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:
            self._pending = None
            self._condition.notify_all()
        with self._lifecycle:
            if self._thread is not None:
                self._thread.join(timeout=7)
                if self._thread.is_alive():
                    LOGGER.warning("[Posture] worker_stop_delayed")
                else:
                    self._thread = None

    def submit(
        self,
        context,
        keypoints,
        *,
        frame_id,
        timestamp,
        frame_size,
        analysis_started_at=None,
        analysis_result=None,
    ):
        # 작은 관절 목록만 복사하고 무제한 프레임/영상 큐는 생성하지 않습니다.
        job = (
            context,
            deepcopy(keypoints),
            frame_id,
            timestamp,
            frame_size,
            analysis_started_at,
            analysis_result,
        )
        with self._condition:
            if not self._stop.is_set():
                self._pending = job
                self._condition.notify()

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None
                        or self._stop.is_set(),
                        timeout=0.25,
                    )
                    if self._stop.is_set():
                        break
                    job, self._pending = self._pending, None
                if job is not None:
                    self.process(*job)
                else:
                    # 카메라가 없어도 Pause/Finish 시 버퍼를 지우고 운동 시작 시 모델을 준비합니다.
                    context = self.manager.get_analysis_context()
                    try:
                        self.predictor.prepare(context)
                        sequence = self.predictor.sequence
                        if (
                            context is not None
                            and sequence is not None
                            and not sequence.classifier.loaded
                        ):
                            self.manager.record_posture_analysis(
                                context,
                                empty_result(
                                    "unavailable",
                                    "model_unavailable:"
                                    + str(sequence.classifier.load_error),
                                ),
                            )
                    except Exception as error:
                        self._publish_failure(context, error)
        finally:
            self.predictor.reset()

    def _publish_failure(self, context, error):
        # 파일/추론 실패를 Reps 경로로 전파하지 않고 원인 타입을 보존합니다.
        result = empty_result("unavailable", type(error).__name__)
        self.manager.record_posture_analysis(context, result)
        now = time.monotonic()
        if now - self._last_log >= 1:
            LOGGER.warning(
                "[Posture] unavailable exercise=%s reason=%s",
                getattr(context, "exercise_name", None),
                type(error).__name__,
            )
            self._last_log = now
        self.predictor.reset()

    def process(
        self,
        context,
        keypoints,
        frame_id,
        timestamp,
        frame_size,
        analysis_started_at=None,
        analysis_result=None,
    ):
        """동기 핵심은 CPU-only 테스트에서 직접 검증하며 실제 서버에서는 자세 워커만 호출합니다."""
        if (
            self._stop.is_set()
            or context is None
            or context != self.manager.get_analysis_context()
        ):
            self.predictor.reset()
            self._last_key = None
            return False
        key = (context, frame_id)
        if (
            self._last_key is not None
            and self._last_key[0] == context
            and frame_id <= self._last_key[1]
        ):
            return False
        self._last_key = key
        # 최신 슬롯도 오래 대기한 요청이면 새 판정처럼 처리하지 않습니다.
        if time.monotonic() - timestamp > 1:
            self.predictor.reset()
            return self.manager.record_posture_analysis(
                context, empty_result("insufficient_pose", "stale_frame")
            )
        try:
            start = time.perf_counter()
            if self.mode == "rule_based":
                result = self.predictor.push(
                    context,
                    keypoints,
                    timestamp,
                    frame_size,
                    analysis_result=analysis_result,
                )
            else:
                result = self.predictor.push(
                    context, keypoints, timestamp, frame_size
                )
            elapsed = (time.perf_counter() - start) * 1000
            if result is None:
                return False
            # 추론 자체가 오래 걸린 결과도 최신 판정처럼 게시하지 않습니다.
            if time.monotonic() - timestamp > 1:
                self.predictor.reset()
                result = empty_result(
                    "insufficient_pose", "stale_frame_after_inference"
                )
            result.update(frame_id=frame_id, observed_at=timestamp)
            if self._stop.is_set():
                return False
            accepted = self.manager.record_posture_analysis(context, result)
            if accepted:
                self._log_metrics(context, result, elapsed, analysis_started_at)
            return accepted
        except Exception as error:
            self._publish_failure(context, error)
            return False

    def _log_metrics(self, context, result, elapsed, analysis_started_at):
        now = time.monotonic()
        if result["status"] == "ready":
            self._events.append(now)
            self._times.append(elapsed)
        while self._events and self._events[0] <= now - 5:
            self._events.popleft()
        if now - self._last_log < 1:
            return
        self._last_log = now
        metrics = (
            self.pipeline_metrics.snapshot()
            if self.pipeline_metrics is not None
            else {}
        )
        LOGGER.info(
            "[PostureMetrics] %s",
            json.dumps(
                dict(
                    camera_receive_fps=metrics.get("camera_capture_fps"),
                    mediapipe_inference_ms=metrics.get(
                        "mediapipe_inference_ms"
                    ),
                    posture_inference_ms=(
                        elapsed if result["status"] == "ready" else None
                    ),
                    posture_inference_mean_ms=(
                        sum(self._times) / len(self._times)
                        if self._times
                        else None
                    ),
                    posture_inference_fps=len(self._events)
                    / min(5, max(now - self._started, 0.001)),
                    end_to_end_analysis_ms=(
                        (time.perf_counter() - analysis_started_at) * 1000
                        if analysis_started_at is not None
                        else None
                    ),
                    model_name=(result.get("classification") or {}).get(
                        "model_name"
                    ),
                    exercise_name=context.exercise_name,
                    prediction_status=result["status"],
                    prediction_reason=result.get("reason"),
                )
            ),
        )
