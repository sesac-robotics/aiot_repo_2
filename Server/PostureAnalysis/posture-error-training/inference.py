"""기존 단일 종목 저장 모델과 Train 정규화 통계를 사용해 오류 확률을 계산합니다. 입력 관절의 형식과 가시성 조건을 먼저 검사합니다.

AI Server integration: load once, call predict_landmarks or push_bgr_frame.
Inputs must be MediaPipe Pose image landmarks (33 x x,y,z,visibility)."""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from features import features, resample
from models import MODELS

BASE = Path(__file__).resolve().parent
FEEDBACK = {
    "shallow": "Lower into the demonstrated squat depth before standing back up.",
    "back not straight": "Keep your torso controlled and match the back alignment demonstrated in the reference squat.",
}


class PostureErrorClassifier:
    def __init__(self, checkpoint=BASE / "outputs/best.pt"):
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = MODELS[saved["architecture"]](
            saved["feature_count"], len(saved["targets"])
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.targets = saved["targets"]
        self.mean = np.array(saved["mean"], dtype=np.float32)
        self.std = np.array(saved["std"], dtype=np.float32)
        self.thresholds = np.array(saved["thresholds"])
        self.length = saved["sequence_length"]
        self.fps = saved["sample_fps"]
        self.window = deque(maxlen=self.length)
        self.last_sample = None
        self.last_frame = None
        self.aspect = None
        self.pose = None

    def predict_landmarks(self, landmarks, aspect=1.0):
        p = np.asarray(landmarks, dtype=np.float32)
        if p.ndim != 3 or p.shape[1:] != (33, 4) or len(p) < 8:
            raise ValueError(
                "Need at least 8 frames of (33,4) MediaPipe image landmarks"
            )
        if not np.isfinite(p).all() or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError(
                "Coordinates/aspect must be finite; aspect must be positive"
            )
        valid = np.min(p[:, [11, 12, 23, 24, 25, 26, 27, 28], 3], axis=1) >= 0.3
        if valid.mean() < 0.6 or valid.sum() < 8:
            return dict(
                status="insufficient_pose_visibility",
                scores={},
                detected_errors=[],
                feedbacks=[],
            )
        p = p.copy()
        flat = p.reshape(len(p), -1)
        for j in range(flat.shape[1]):
            flat[:, j] = np.interp(
                np.arange(len(p)), np.flatnonzero(valid), flat[valid, j]
            )
        x = resample(features(p, aspect), self.length)
        with torch.inference_mode():
            scores = torch.sigmoid(
                self.model(
                    torch.tensor(
                        ((x - self.mean) / self.std)[None], dtype=torch.float32
                    )
                )
            ).numpy()[0]
        errors = [
            t
            for t, s, threshold in zip(self.targets, scores, self.thresholds)
            if s >= threshold
        ]
        return dict(
            status="ok",
            exercise="squats",
            scores=dict(zip(self.targets, map(float, scores))),
            thresholds=dict(zip(self.targets, map(float, self.thresholds))),
            detected_errors=errors,
            normal_candidate=not bool(errors),
            normal_scope="No selected error detected; other errors are not assessed.",
            feedbacks=[FEEDBACK[t] for t in errors],
            valid_frame_ratio=float(valid.mean()),
            model_scope="experimental_QEVD_clip_label_classifier",
        )

    def push_bgr_frame(self, frame, timestamp=None):
        """Call for each decoded ESP32-CAM frame. Return None between sampling ticks.
        Keep one instance per camera/session; use a monotonic timestamp in seconds.
        """
        import cv2
        import mediapipe as mp

        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_frame is not None and (
            now <= self.last_frame or now - self.last_frame > 1.0
        ):
            self.window.clear()
            self.last_sample = None
        self.last_frame = now
        if (
            self.last_sample is not None
            and now - self.last_sample + 1e-8 < 1 / self.fps
        ):
            return None
        self.last_sample = now
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
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p = (
            np.array(
                [
                    [l.x, l.y, l.z, l.visibility]
                    for l in result.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )
            if result.pose_landmarks
            else np.zeros((33, 4), dtype=np.float32)
        )
        self.window.append(p)
        if len(self.window) < self.length:
            return dict(
                status="warming_up",
                frames=len(self.window),
                required_frames=self.length,
            )
        return self.predict_landmarks(np.array(self.window), aspect)

    def close(self):
        if self.pose:
            self.pose.close()
            self.pose = None
