"""두 점수 방식에 공통으로 쓰는 화면 관절의 기하 특징을 계산한다.

입력은 픽셀 좌표 N×33×2이며 미관측은 NaN이다. 출력 각도는 도,
거리는 몸통 길이 비율이다. 라벨·출처·제공자 문자열은 받지 않는다.
"""

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """같은 좌표계 세 점의 B 내각을 0~180도로 반환한다.

    Args:
        a: N×2 첫 점의 픽셀 좌표.
        b: N×2 중심점의 픽셀 좌표.
        c: N×2 마지막 점의 픽셀 좌표.

    Returns:
        np.ndarray: N개 각도. 누락·영벡터는 NaN이다.

    Raises:
        ValueError: 배열 형상이 서로 맞지 않는 경우.
    """
    u, v = a - b, c - b
    denominator = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosine = np.divide(
        np.sum(u * v, axis=1),
        denominator,
        out=np.full(len(u), np.nan),
        where=denominator > 1e-6,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def channels(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """화면 관절의 시간별 각도·정렬·이동량을 반환한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. N은 1 이상이다.

    Returns:
        dict[str, np.ndarray]: 이름별 N개 수치. 끝이 _deg이면 도,
        그 외는 몸통 길이로 나눈 비율이다. 미관측은 NaN이다.

    Raises:
        ValueError: 입력 차원 또는 관절 수가 다르거나 비어 있는 경우.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 sequence")
    values = {}
    for name, shoulder, elbow, wrist, hip, knee, ankle, heel, toe in (
        ("left", 11, 13, 15, 23, 25, 27, 29, 31),
        ("right", 12, 14, 16, 24, 26, 28, 30, 32),
    ):
        torso = np.linalg.norm(seq[:, shoulder] - seq[:, hip], axis=1)
        torso = np.where(torso > 1, torso, np.nan)
        for joint, points in (
            ("elbow", (shoulder, elbow, wrist)),
            ("knee", (hip, knee, ankle)),
            ("hip", (shoulder, hip, knee)),
            ("line", (shoulder, hip, ankle)),
            ("elevation", (hip, shoulder, elbow)),
        ):
            values[f"{name}_{joint}_deg"] = angle_deg(
                *(seq[:, point] for point in points)
            )
        values[f"{name}_hip_above_knee"] = (
            seq[:, knee, 1] - seq[:, hip, 1]
        ) / torso
        values[f"{name}_elbow_offset"] = (
            np.abs(seq[:, elbow, 0] - seq[:, shoulder, 0]) / torso
        )
        values[f"{name}_hand_above_elbow"] = (
            seq[:, elbow, 1] - seq[:, wrist, 1]
        ) / torso
        values[f"{name}_heel_lift"] = (seq[:, toe, 1] - seq[:, heel, 1]) / torso
        # 손목~발끝 선은 지면의 기하 대리값이며 실제 하중은 알 수 없다.
        floor = seq[:, toe] - seq[:, wrist]
        distance = np.linalg.norm(floor, axis=1)
        normal = np.stack([floor[:, 1], -floor[:, 0]], axis=1)
        sign = np.where(normal[:, 1] < 0, 1, -1)
        normal = normal * sign[:, None]
        denominator = distance * torso
        gap = np.sum((seq[:, knee] - seq[:, wrist]) * normal, axis=1)
        values[f"{name}_knee_floor_gap"] = np.divide(
            gap,
            denominator,
            out=np.full(len(seq), np.nan),
            where=denominator > 1e-6,
        )
    shoulder_mid = (seq[:, 11] + seq[:, 12]) / 2
    hip_mid = (seq[:, 23] + seq[:, 24]) / 2
    trunk = shoulder_mid - hip_mid
    length = np.linalg.norm(trunk, axis=1)
    length = np.where(length > 1, length, np.nan)
    values["frontal_tilt_deg"] = np.degrees(
        np.arctan2(np.abs(trunk[:, 0]), np.abs(trunk[:, 1]))
    )
    values["bilateral_wrist_height_gap"] = (
        np.abs(seq[:, 15, 1] - seq[:, 16, 1]) / length
    )
    for name, knee, ankle in (("left", 25, 27), ("right", 26, 28)):
        values[f"{name}_knee_ankle_offset"] = (
            np.abs(seq[:, knee, 0] - seq[:, ankle, 0]) / length
        )
    return values


def model_features(sequence: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """기하 시계열을 통계·16개 등간격 표본·관측률로 변환한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 누락은 NaN이다.

    Returns:
        tuple[np.ndarray, list[str]]: 특징 벡터와 같은 순서의 이름.
        각도는 도, 거리는 몸통 비율이며 관측률은 0~1이다.

    Raises:
        ValueError: 비어 있거나 관절 좌표 형상이 잘못된 경우.
    """
    values, names = [], []
    for name, channel in channels(sequence).items():
        finite = channel[np.isfinite(channel)]
        stats = (
            np.percentile(finite, [5, 25, 50, 75, 95]).tolist()
            + [float(np.std(finite))]
            if finite.size
            else [np.nan] * 6
        )
        stats.append(float(np.isfinite(channel).mean()))
        for suffix, value in zip(
            ("p05", "p25", "p50", "p75", "p95", "std", "coverage"), stats
        ):
            names.append(name + "_" + suffix)
            values.append(value)
        sampled = channel[np.linspace(0, len(channel) - 1, 16).astype(int)]
        for index, value in enumerate(sampled):
            names.append(f"{name}_time{index:02d}")
            values.append(value)
    return np.asarray(values), names
