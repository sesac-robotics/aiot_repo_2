"""보존된 Heavy 관절 추출·반복 분절·점수 계산을 새 실행기에 연결합니다."""

import copy

import cv2
import mediapipe as mp
import numpy as np
from movement import MovementTracker
from shoulder_candidate import ShoulderTracker

from .legacy_score import ENGINE, finish_observation_window, safe, score_samples
from .paths import FROZEN, REPO
from .preview_privacy import encode_masked_preview
from .score_observation import (
    ScoringReadyGate,
    choose_recovery,
    inspect_pose,
    retry_due,
    segment_observation,
)
from .score_recording import ScoreRecording, sha256


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
        self.live_enabled = True

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
        elif (
            self.live_enabled
            and tick - self.last_live_tick >= 10
            and len(current) >= 10
        ):
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
        detected = self.tracker.count
        complete = sum(
            rep.get("assessment_complete", False) for rep in self.repetitions
        )
        return safe(
            {
                "status": "collecting",
                "movement": self.tracker.status(),
                "repetitions": list(self.repetitions),
                "pending": len(self.pending),
                "live_score_100": self.live_score,
                "live_error": self.live_error,
                "assessment_summary": dict(
                    detected_repetitions=detected,
                    scored_repetitions=complete,
                    retry_required_repetitions=max(
                        0, (detected or 0) - complete
                    ),
                    all_detected_repetitions_scored=bool(
                        detected and complete == detected and not self.pending
                    ),
                ),
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
                        "observation": (
                            segment_observation(segment, width, height)
                            if self.exercise == "squats"
                            else None
                        ),
                        "assessment_complete": all(
                            result["score_100"].get(method, {}).get("total")
                            is not None
                            for method in ("ai", "rule")
                        ),
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

    def __init__(self, exercise, session_context=None, recording_root=None):
        super().__init__(exercise)
        self.origin = None
        self.last_preview = float("-inf")
        self.extraction_last_tick = -1
        self.retry_pose = None
        self.last_retry = float("-inf")
        self.last_retry_success = float("-inf")
        self.gate = ScoringReadyGate()
        self.readiness = None
        self.quality = None
        self.recording = None
        self.had_reference = False
        self.ready_buffer = []
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
        if session_context:
            args = {} if recording_root is None else {"root": recording_root}
            self.recording = ScoreRecording(
                session_context,
                exercise,
                provenance={
                    "heavy_model_sha256": sha256(
                        FROZEN / "mediapipe-pilot/models/heavy.task"
                    ),
                    "score_adapter_sha256": sha256(__file__),
                    "recovery_model_sha256": sha256(
                        REPO
                        / "Server/ExerciseEvaluation/models/pose_landmarker_full.task"
                    ),
                    "recovery_model": "MediaPipe Full IMAGE float16 v1",
                    "visibility_minimum": 0.5,
                    "presence_minimum": 0.5,
                    "border_fraction": 0.02,
                    "scoring_model_and_frozen_tracker_changed": False,
                },
                **args,
            )

    def snapshot(self):
        state = super().snapshot()
        state["scoring_readiness"] = self.readiness
        state["joint_quality"] = self.quality
        if self.exercise == "squats" and not self.gate.prepared:
            state["status"] = "preparing"
        if self.recording:
            state["recording"] = self.recording.snapshot()
        return state

    @staticmethod
    def raw_pose(result):
        if not result.pose_landmarks:
            return np.full((33, 5), np.nan)
        return np.asarray(
            [
                [p.x, p.y, p.z, p.visibility, p.presence]
                for p in result.pose_landmarks[0]
            ],
            float,
        )

    def process(self, frame, stamp, metadata=None):
        """실제 수신 시간으로 슬롯을 정하며 누락 프레임을 복제하지 않습니다."""
        if self.origin is None:
            self.origin = stamp
        tick = round((stamp - self.origin) * 10)
        if tick <= self.extraction_last_tick:
            return None
        self.extraction_last_tick = tick
        h, w = frame.shape[:2]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        )
        result = self.pose.detect_for_video(image, tick * 100)
        raw = self.raw_pose(result)
        xy, quality = inspect_pose(raw, w, h)
        retry = None
        extraction = "video"
        if self.exercise == "squats" and not quality["all_items_observed"]:
            # A second genuine detection may recover stale VIDEO tracking. It never
            # supplies a missing frame or mixes joints from different detections.
            if retry_due(raw, stamp, self.last_retry):
                self.last_retry = stamp
                if self.retry_pose is None:
                    options = mp.tasks.vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=str(
                                REPO
                                / "Server/ExerciseEvaluation/models/pose_landmarker_full.task"
                            )
                        ),
                        running_mode=mp.tasks.vision.RunningMode.IMAGE,
                        num_poses=1,
                    )
                    self.retry_pose = (
                        mp.tasks.vision.PoseLandmarker.create_from_options(
                            options
                        )
                    )
                retry = self.raw_pose(self.retry_pose.detect(image))
                selected, extraction, xy, quality = choose_recovery(
                    raw, retry, w, h
                )
                if extraction != "video":
                    self.last_retry_success = stamp
            else:
                selected = raw
        else:
            selected = raw
        if retry is None:
            selected = raw
        self.quality = quality
        was_prepared = self.gate.prepared
        previous_tick = self.gate.last_tick
        self.readiness = self.gate.step(tick, quality)
        delivered = self.exercise != "squats" or self.gate.prepared
        released = []
        if self.exercise == "squats" and not was_prepared:
            if not quality["all_items_observed"] or (
                previous_tick is not None and tick != previous_tick + 1
            ):
                self.ready_buffer.clear()
            if quality["all_items_observed"]:
                self.ready_buffer.append((tick, xy.copy()))
            if self.gate.prepared:
                # Replay only the six actual consecutive ready observations.
                # The readiness check must not discard the first movement.
                for ready_tick, points in self.ready_buffer:
                    self.process_points(points, ready_tick, w, h)
                    released.append(ready_tick)
                self.ready_buffer.clear()
                state = self.snapshot()
            else:
                state = self.snapshot()
        else:
            state = self.process_points(xy, tick, w, h)
        if released:
            state["readiness_released_actual_ticks"] = released
        if self.had_reference and self.tracker.standing_reference is None:
            self.gate.prepared = False
            self.gate.streak = 0
            self.ready_buffer.clear()
            self.had_reference = False
            self.readiness = dict(
                self.readiness, prepared=False, ready_frames=0
            )
            state["scoring_readiness"] = self.readiness
            state["status"] = "preparing"
        if self.tracker.standing_reference is not None:
            self.had_reference = True
        if state is not None and stamp - self.last_preview >= 0.2:
            state["preview_jpeg_b64"], state["face_mosaic"] = (
                encode_masked_preview(frame, xy)
            )
            self.last_preview = stamp
        if state is not None and self.recording:
            self.recording.append(
                tick,
                stamp,
                metadata or {"shape": list(frame.shape)},
                raw,
                retry,
                selected,
                xy,
                quality,
                extraction,
                delivered,
                state,
                state.get("preview_jpeg_b64"),
            )
            state["recording"] = self.recording.snapshot()
        return state

    def reset_motion(self):
        """완료 반복을 보존하며 일시정지 전의 진행 중 움직임을 버립니다."""
        self.flush()
        count = self.tracker.count
        self.tracker = (
            MovementTracker(self.exercise, self.view)
            if self.exercise != "shoulder"
            else ShoulderTracker(self.exercise, self.view)
        )
        self.tracker.count = count
        self.available.clear()
        self.pending.clear()
        self.live_score = None
        self.gate = ScoringReadyGate()
        self.ready_buffer.clear()
        self.had_reference = False
        self.readiness = None

    def finalize_recording(self):
        if self.recording:
            self.recording.finish(self.snapshot(), "stop_requested")

    def close(self):
        """Heavy 추출 자원을 해제하고 미완료 동작을 점수화하지 않습니다."""
        self.pose.close()
        if self.retry_pose:
            self.retry_pose.close()
        if self.recording:
            self.recording.finish(self.snapshot(), "worker_closed")
