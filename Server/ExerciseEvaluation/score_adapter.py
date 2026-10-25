"""보존된 Heavy 관절 추출·반복 분절·점수 계산을 새 실행기에 연결합니다."""

import copy

import cv2
import mediapipe as mp
import numpy as np
from movement import MovementTracker
from shoulder_candidate import ShoulderTracker

from .legacy_score import ENGINE, finish_observation_window, safe, score_samples
from .paths import FROZEN


class PoseScoreAdapter:
    """10Hz 슬롯 누락을 유지하며 완결된 반복만 기존 계산식으로 평가합니다."""

    def __init__(self, exercise, view="front"):
        self.exercise, self.view = exercise, view
        ENGINE.model(exercise)
        self.tracker = (
            ShoulderTracker(exercise, view)
            if exercise == "shoulder"
            else MovementTracker(exercise, view)
        )
        self.pending = []
        self.repetitions = []
        self.available = []
        self.last_tick = -1
        self.shape = None
        self.live_score = None
        self.live_error = None
        self.last_live_tick = -10

    def process_points(self, xy, tick, width, height):
        """저장 관절 재생과 실제 추출에 동일한 분절·복귀 관측 로직을 적용합니다."""
        if tick <= self.last_tick:
            return None
        if self.shape is not None and self.shape != (width, height):
            raise ValueError(
                "영상 크기가 변경됐습니다. 측정을 다시 시작합니다."
            )
        self.shape = (width, height)
        self.last_tick = tick
        self.available.append((tick, xy.copy()))
        self.available = self.available[-300:]
        segment = self.tracker.push(tick, xy, width=width, height=height)
        if segment:
            self.pending.append(
                (
                    segment,
                    copy.deepcopy(self.tracker.standing_reference),
                    self.tracker.count,
                )
            )
        self._finish(width, height)
        segmented = self.exercise in ("squats", "shoulder")
        current = (
            list(self.tracker.buffer)
            if segmented and self.tracker.active
            else ([] if segmented else self.available[-60:])
        )
        if not current or not (self.tracker.observation or {}).get("ready"):
            self.live_score = None
            self.live_error = None
        elif tick - self.last_live_tick >= 10 and len(current) >= 10:
            self.last_live_tick = tick
            try:
                live = score_samples(
                    current,
                    width,
                    height,
                    self.exercise,
                    self.view,
                    phase="active",
                    standing_reference=self.tracker.standing_reference,
                )
                self.live_score = live["score_100"]
                self.live_error = None
            except Exception as error:
                self.live_score = None
                self.live_error = str(error)
        return self.snapshot()

    def snapshot(self):
        """현재 완료 결과와 진행 중 잠정 결과를 JSON 가능한 값으로 반환합니다."""
        return safe(
            {
                "status": "collecting",
                "movement": self.tracker.status(),
                "repetitions": list(self.repetitions),
                "pending": len(self.pending),
                "live_score_100": self.live_score,
                "live_error": self.live_error,
                "scope": "원래 점수 계산의 실험 결과입니다. 미평가 항목을 만점으로 채우지 않습니다.",
            }
        )

    def _finish(self, width, height, flush=False):
        while self.pending:
            segment, reference, number = self.pending[0]
            if (
                not flush
                and self.exercise == "squats"
                and reference is not None
                and self.last_tick < segment[-1][0] + 3
            ):
                break
            self.pending.pop(0)
            returns = (
                finish_observation_window(segment, reference, self.available)
                if self.exercise == "squats" and reference is not None
                else None
            )
            try:
                result = score_samples(
                    segment,
                    width,
                    height,
                    self.exercise,
                    self.view,
                    standing_reference=reference,
                    return_samples=returns,
                )
                self.repetitions.append(
                    {
                        "number": number,
                        "start_tick": segment[0][0],
                        "end_tick": segment[-1][0],
                        "score_100": result["score_100"],
                        "items": result["items"],
                    }
                )
            except Exception as error:
                self.repetitions.append(
                    {"number": number, "status": "held", "error": str(error)}
                )

    def flush(self):
        """저장 구간의 종료 재생에서만 대기 중인 완료 반복을 마감합니다."""
        if self.shape:
            self._finish(*self.shape, flush=True)
        return safe(self.repetitions)

    def close(self):
        """포즈 전용 재생기는 해제할 외부 장치가 없습니다."""


class ScoreAdapter(PoseScoreAdapter):
    """기존 배포와 동일한 MediaPipe Tasks Heavy를 사용합니다."""

    def __init__(self, exercise):
        super().__init__(exercise)
        self.origin = None
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(
                    FROZEN / "mediapipe-pilot/models/heavy.task"
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process(self, frame, stamp):
        """실제 수신 시간으로 슬롯을 정하며 누락 프레임을 복제하지 않습니다."""
        if self.origin is None:
            self.origin = stamp
        tick = round((stamp - self.origin) * 10)
        if tick <= self.last_tick:
            return None
        h, w = frame.shape[:2]
        result = self.pose.detect_for_video(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ),
            ),
            tick * 100,
        )
        xy = np.full((33, 2), np.nan)
        if result.pose_landmarks:
            for j, p in enumerate(result.pose_landmarks[0]):
                if p.visibility >= 0.5 and p.presence >= 0.5:
                    xy[j] = [p.x * w, p.y * h]
        return self.process_points(xy, tick, w, h)

    def close(self):
        """Heavy 추출 자원을 해제하고 미완료 동작을 점수화하지 않습니다."""
        self.pose.close()
