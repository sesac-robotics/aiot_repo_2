"""다종목의 저장 모델·정규화·활성 항목을 불러와 오류 확률을 계산합니다. 미지원 항목은 확률 없음으로 구분하고 관절 미관측을 정상으로 보고하지 않습니다.

Exercise-aware four-model inference. Caller must supply exercise, not guessed.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from models import MODELS
from suite_config import BASE, CONFIG_VERSION, SUITE, TASKS
from suite_features import prepare_landmarks


class ExerciseClassifier:
    def __init__(self, exercise, checkpoint=None):
        if exercise not in TASKS:
            raise ValueError(f"Choose one of {list(TASKS)}")
        torch.set_num_threads(2)
        self.exercise = exercise
        saved = torch.load(
            checkpoint or SUITE / exercise / "outputs/best.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            saved["exercise"] != exercise
            or saved["config_version"] != CONFIG_VERSION
        ):
            raise ValueError("Wrong exercise or preprocessing version")
        self.saved = saved
        self.targets = saved["targets"]
        self.mean = np.array(saved["mean"], np.float32)
        self.std = np.array(saved["std"], np.float32)
        self.thresholds = np.array(saved["thresholds"])
        self.active = np.array(saved["active_targets"])
        self.model = MODELS[saved["architecture"]](
            saved["feature_count"], len(self.targets) + 1
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.window = deque(maxlen=saved["sequence_length"])
        self.pose = None
        self.last_tick = None
        self.last_frame = None
        self.aspect = None
        self.feedback = self._feedback()

    def _feedback(self):
        import json

        mapping = json.loads((SUITE / "source_feedback.json").read_text())
        result = {}
        for target in self.targets:
            matches = [
                v for k, v in mapping.items() if k.split(" - ", 1)[-1] == target
            ]
            prefix = {
                "elbow_plank": "elbow plank",
                "shoulder_press": "shoulder press",
            }.get(self.exercise, self.exercise)
            exact = mapping.get(prefix + " - " + target)
            result[target] = exact or (
                matches[0]
                if matches
                else "Compare this movement with the intended reference variation."
            )
        return result

    def predict_landmarks(self, landmarks, aspect=1.0):
        p = np.array(landmarks, np.float32)
        if p.ndim != 3 or p.shape[1:] != (33, 4) or len(p) < 8:
            raise ValueError(
                "Expected at least 8 frames of (33,4) image landmarks"
            )
        if not np.isfinite(p).all() or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("Non-finite coordinates or invalid aspect")
        x, ratio = prepare_landmarks(p, aspect, self.saved["required_joints"])
        if x is None:
            return dict(
                status="insufficient_pose_visibility",
                exercise=self.exercise,
                scores={},
                detected_errors=[],
                feedbacks=[],
                valid_frame_ratio=ratio,
            )
        with torch.inference_mode():
            values = torch.sigmoid(
                self.model(
                    torch.tensor(
                        ((x - self.mean) / self.std)[None], dtype=torch.float32
                    )
                )
            ).numpy()[0]
        detected = [
            t
            for j, t in enumerate(self.targets)
            if self.active[j] and values[j] >= self.thresholds[j]
        ]
        any_error = (
            bool(values[-1] >= self.thresholds[-1]) if self.active[-1] else None
        )
        result = dict(
            status="ok",
            exercise=self.exercise,
            scores={
                t: float(values[j]) if self.active[j] else None
                for j, t in enumerate(self.targets)
            },
            detected_errors=detected,
            feedbacks=[self.feedback[t] for t in detected],
            any_selected_error_score=(
                float(values[-1]) if self.active[-1] else None
            ),
            any_selected_error=any_error,
            valid_frame_ratio=ratio,
            normal_assessment=(
                "unavailable"
                if not self.saved["normal_supported"]
                else "unverified_draft_candidate_only"
            ),
            scope=self.saved["scope"],
        )
        if self.saved["normal_supported"]:
            result["normal_candidate"] = not detected and any_error is False
        return result

    def push_bgr_frame(self, frame, timestamp=None):
        import cv2
        import mediapipe as mp

        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_frame is not None and (
            now <= self.last_frame or now - self.last_frame > 1
        ):
            self.window.clear()
            self.last_tick = None
        self.last_frame = now
        if (
            self.last_tick is not None
            and now - self.last_tick + 1e-8 < 1 / self.saved["sample_fps"]
        ):
            return None
        self.last_tick = now
        h, w = frame.shape[:2]
        aspect = w / h
        if self.aspect is not None and abs(self.aspect - aspect) > 0.01:
            self.window.clear()
        self.aspect = aspect
        if self.pose is None:
            self.pose = mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        if max(h, w) > 480:
            frame = cv2.resize(
                frame, (round(w * 480 / max(h, w)), round(h * 480 / max(h, w)))
            )
        r = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw = (
            np.array(
                [
                    [l.x, l.y, l.z, l.visibility]
                    for l in r.pose_landmarks.landmark
                ],
                np.float32,
            )
            if r.pose_landmarks
            else np.zeros((33, 4), np.float32)
        )
        self.window.append(raw)
        if len(self.window) < self.window.maxlen:
            return dict(
                status="warming_up",
                exercise=self.exercise,
                frames=len(self.window),
                required_frames=self.window.maxlen,
            )
        return self.predict_landmarks(np.array(self.window), aspect)

    def close(self):
        if self.pose:
            self.pose.close()
            self.pose = None


class FiveExerciseClassifier:
    """Lazy dispatcher; one instance per camera/user. Explicit exercise required."""

    def __init__(self):
        self.instances = {}

    def model(self, exercise):
        if exercise not in self.instances:
            if exercise == "squats":
                from inference import PostureErrorClassifier

                self.instances[exercise] = PostureErrorClassifier()
            else:
                self.instances[exercise] = ExerciseClassifier(exercise)
        return self.instances[exercise]

    def predict_landmarks(self, exercise, landmarks, aspect=1.0):
        return self.model(exercise).predict_landmarks(landmarks, aspect)

    def close(self):
        for model in self.instances.values():
            model.close()
