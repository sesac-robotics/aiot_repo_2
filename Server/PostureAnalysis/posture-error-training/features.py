"""관절 위치를 신체 중심·크기에 정규화하고 기하 특징과 시간 재표본화를 수행합니다. 저장된 학습 정규화 통계와 현재 입력의 공간 정규화를 구분합니다.

Shared MediaPipe image landmark -> temporal features (offline and live)."""

import numpy as np

SEQUENCE_LENGTH = 32
SAMPLE_FPS = 6
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
TARGETS = ["shallow", "back not straight"]
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
]


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(
        np.arccos(
            np.clip(
                np.sum(u * v, axis=-1)
                / (
                    np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
                    + 1e-8
                ),
                -1,
                1,
            )
        )
    )


def features(landmarks, aspect=1.0):
    """Input (T,33,4) normalized image coords. x/z scaled by image aspect."""
    p = np.asarray(landmarks, dtype=np.float32).copy()
    p[:, :, 0] *= aspect
    p[:, :, 2] *= aspect
    xy = p[:, :, :2]
    hip = (p[:, 23, :3] + p[:, 24, :3]) / 2
    shoulder = (p[:, 11, :3] + p[:, 12, :3]) / 2
    scale = np.maximum(
        np.linalg.norm(shoulder[:, :2] - hip[:, :2], axis=-1), 0.05
    )
    p[:, :, :3] = (p[:, :, :3] - hip[:, None, :]) / scale[:, None, None]
    extras = np.stack(
        [
            angle(xy[:, 23], xy[:, 25], xy[:, 27]),
            angle(xy[:, 24], xy[:, 26], xy[:, 28]),
            angle(xy[:, 11], xy[:, 23], xy[:, 25]),
            angle(xy[:, 12], xy[:, 24], xy[:, 26]),
            np.degrees(
                np.arctan2(
                    np.abs(shoulder[:, 0] - hip[:, 0]),
                    np.abs(shoulder[:, 1] - hip[:, 1]) + 1e-8,
                )
            ),
        ],
        axis=-1,
    )
    return np.concatenate(
        [p[:, JOINTS].reshape(len(p), -1), extras], axis=-1
    ).astype(np.float32)


def resample(x, n=SEQUENCE_LENGTH):
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, n)
    return np.stack(
        [np.interp(new, old, x[:, j]) for j in range(x.shape[1])], axis=1
    ).astype(np.float32)
