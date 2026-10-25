"""연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다.

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
