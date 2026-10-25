"""팀원의 6Hz·32표본 오류 분류와 legacy Pose 추출 계약을 연결합니다."""

from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from live_app.engine import TemporalGate
from suite_inference import FiveExerciseClassifier

from .contracts import CORRECTION_CODES, ObservedCadence


class CorrectionAdapter:
    """선택한 운동의 교정 모델만 로드하고 부족한 관측을 보류합니다."""

    def __init__(self, exercise):
        self.model = FiveExerciseClassifier().model(
            CORRECTION_CODES.get(exercise, exercise)
        )
        squat = exercise == "squats"
        self.required = (
            [11, 12, 23, 24, 25, 26, 27, 28]
            if squat
            else self.model.saved["required_joints"]
        )
        self.fps = self.model.fps if squat else self.model.saved["sample_fps"]
        self.gate = TemporalGate(
            self.model.length if squat else self.model.saved["sequence_length"]
        )
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.stamps = deque(maxlen=self.gate.length)
        self.clock = ObservedCadence(self.fps)
        self.cadence_resets = 0

    def process(self, frame, stamp):
        """실제 관측 간격을 유지하며 오류 확률을 자세 점수로 변환하지 않습니다."""
        continuous = self.clock.observe(stamp)
        if continuous is None:
            return None
        if not continuous:
            self.cadence_resets += 1
            self.gate.clear()
            self.stamps.clear()
            self.pose.reset()
        h, w = frame.shape[:2]
        resized = (
            cv2.resize(
                frame, (round(w * 480 / max(h, w)), round(h * 480 / max(h, w)))
            )
            if max(h, w) > 480
            else frame
        )
        result = self.pose.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        raw = (
            np.array(
                [
                    [p.x, p.y, p.z, p.visibility]
                    for p in result.pose_landmarks.landmark
                ],
                np.float32,
            )
            if result.pose_landmarks
            else None
        )
        stage = self.gate.append(raw, stamp, w / h, self.required)
        self.stamps.append(stamp)
        while len(self.stamps) > len(self.gate.points):
            self.stamps.popleft()
        prediction = None
        if stage == "ready":
            prediction = self.model.predict_landmarks(
                np.array(self.gate.points), w / h
            )
            if prediction["status"] == "ok":
                stable = self.gate.stabilize(prediction)
                mapping = dict(
                    zip(
                        prediction.get("detected_errors", []),
                        prediction.get("feedbacks", []),
                    )
                )
                prediction["raw_detected_errors"] = prediction.get(
                    "detected_errors", []
                )
                prediction["detected_errors"] = stable
                prediction["feedbacks"] = [
                    mapping[t] for t in stable if t in mapping
                ]
                stage = "predicting"
        return {
            "status": stage,
            "result": prediction,
            "buffer_frames": len(self.gate.points),
            "required_frames": self.gate.length,
            "scope": "선택 오류 분류입니다. 오류 미검출은 전체 정상 판정이 아닙니다.",
            "window_end": stamp,
            "window_seconds": (
                self.stamps[-1] - self.stamps[0] if self.stamps else 0
            ),
            "cadence_resets": self.cadence_resets,
            "message": (
                "6Hz 관측 슬롯이 누락되면 연속 구간을 다시 수집합니다."
                if self.cadence_resets
                else ""
            ),
        }

    def close(self):
        """관절 추출 자원을 해제합니다."""
        self.pose.close()
