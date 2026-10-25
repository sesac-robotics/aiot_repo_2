"""33개 관절 시계열에서 공통 기하·시간 특징과 관측 가능 비율을 계산합니다. 학습과 추론이 동일한 전처리 순서를 사용해야 합니다.

Same 109-feature contract for all four exercise models and their serving code.
"""

import numpy as np

from .squat_features import angle, resample

JOINTS = [
    0,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
FEATURE_NAMES = [
    f"joint_{j}_{axis}"
    for j in JOINTS
    for axis in ("x", "y", "z", "visibility")
] + [
    "left_knee_deg",
    "right_knee_deg",
    "left_hip_deg",
    "right_hip_deg",
    "torso_lean_deg",
    "left_elbow_deg",
    "right_elbow_deg",
    "left_shoulder_deg",
    "right_shoulder_deg",
]


def make_features(landmarks, aspect):
    p = np.array(landmarks, dtype=np.float32, copy=True)
    p[:, :, 0] *= aspect
    p[:, :, 2] *= aspect
    xy = p[:, :, :2].copy()
    hip = (p[:, 23, :3] + p[:, 24, :3]) / 2
    shoulder = (p[:, 11, :3] + p[:, 12, :3]) / 2
    scale = np.maximum(
        np.linalg.norm(shoulder[:, :2] - hip[:, :2], axis=-1), 0.05
    )

    def measured_angle(a, b, c):
        reliable = p[:, [a, b, c], 3].min(axis=1) >= 0.3
        return np.where(reliable, angle(xy[:, a], xy[:, b], xy[:, c]), 0.0)

    angles = [
        measured_angle(a, b, c)
        for a, b, c in [(23, 25, 27), (24, 26, 28), (11, 23, 25), (12, 24, 26)]
    ]
    angles += [
        np.degrees(
            np.arctan2(
                np.abs(shoulder[:, 0] - hip[:, 0]),
                np.abs(shoulder[:, 1] - hip[:, 1]) + 1e-8,
            )
        )
    ]
    angles += [
        measured_angle(a, b, c)
        for a, b, c in [(11, 13, 15), (12, 14, 16), (13, 11, 23), (14, 12, 24)]
    ]
    p[:, :, :3] = (p[:, :, :3] - hip[:, None, :]) / scale[:, None, None]
    p[:, :, :3] *= (p[:, :, 3] >= 0.3)[:, :, None]
    return np.concatenate(
        [p[:, JOINTS].reshape(len(p), -1), np.stack(angles, axis=-1)], axis=1
    ).astype(np.float32)


def prepare_landmarks(raw, aspect, required):
    p = np.array(raw, dtype=np.float32, copy=True)
    valid = np.isfinite(p).all(axis=(1, 2)) & (
        np.nan_to_num(p[:, required, 3], nan=0).min(axis=1) >= 0.3
    )
    if len(p) < 8 or valid.sum() < 8 or valid.mean() < 0.6:
        return None, float(valid.mean()) if len(p) else 0
    flat = p.reshape(len(p), -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(
            np.arange(len(p)), np.flatnonzero(valid), flat[valid, j]
        )
    return resample(make_features(p, aspect)), float(valid.mean())
