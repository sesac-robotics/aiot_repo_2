"""플랭크를 전신 유무 대신 항목별 필수 관절·구도·시간으로 판정한다.

얼굴·손목은 현재 네 항목의 필수 입력이 아니다. 신체 지지와 지지
지속성은 관절 대리값이며 실제 하중·접촉 여부를 보증하지 않는다.
"""

from typing import Any

import numpy as np

from app.scoring import geometry
from app.scoring import rule_score

ITEMS = [
    "trunk_line",
    "elbow_sagittal_position",
    "leg_and_body_support",
    "support_continuity",
]
WEIGHTS = [40, 25, 20, 15]
JOINTS = [(11, 23, 27), (11, 13, 23), (23, 25, 27), (23, 25, 27)]
CHANNELS = ["line_deg", "elbow_offset", "knee_deg", "knee_deg"]


def assess(
    sequence: np.ndarray, duration_seconds: float, permitted: list[bool]
) -> dict:
    """항목마다 충분히 관측되는 한쪽 관절만 골라 등급과 입력을 반환한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        duration_seconds: 실제 평가 구간 길이, 초 단위 양수.
        permitted: 직접 구도 검토에서 허용한 네 항목의 bool 목록.

    Returns:
        dict: 관측률·필수 관절·선택 측면·항목 등급·보류 사유 목록.
        정렬은 5표본 이상, 지속성은 추가로 밀리초로 반올림한 3초 이상을 요구한다.
        선택 관절이 같은 프레임에서 모두 유효한 비율 80%가 기준이다.
        비가시 항목을 보간하거나 부분 점수를 100점으로 환산하지 않는다.

    Raises:
        ValueError: 좌표·시간·허용 목록 형상 또는 범위 오류.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 pixels")
    if (
        not np.isfinite(duration_seconds)
        or duration_seconds <= 0
        or len(permitted) != 4
    ):
        raise ValueError("Positive duration and four permissions required")
    values = geometry.channels(seq)
    rows: list[dict[str, Any]] = []
    for index, (item, joints, channel) in enumerate(
        zip(ITEMS, JOINTS, CHANNELS, strict=True)
    ):
        options = []
        for offset, side in [(0, "left"), (1, "right")]:
            selected = [j + offset for j in joints]
            coverage = float(np.isfinite(seq[:, selected]).all((1, 2)).mean())
            options.append((coverage, side, selected))
        coverage, side, selected = max(options, key=lambda x: x[0])
        reasons = []
        if not permitted[index]:
            reasons.append("view_or_direct_review_not_permitted")
        if len(seq) < 5:
            reasons.append("fewer_than_five_samples")
        if coverage < 0.8:
            reasons.append("item_joint_observation_below_80_percent")
        if index == 3 and round(duration_seconds, 3) < 3:
            reasons.append("continuity_shorter_than_three_seconds")
        source = values[side + "_" + channel]
        if index == 3:
            finite = source[np.isfinite(source)]
            measured = (
                float(np.percentile(finite, 95) - np.percentile(finite, 5))
                if len(finite)
                else None
            )
            grade = rule_score._grade(measured, 8, 20, False)
        else:
            measured = rule_score._stat(source, 90 if index == 1 else 10)
            grade = rule_score._grade(
                measured,
                *[(168, 155, True), (0.2, 0.4, False), (165, 145, True)][index],
            )
        if measured is None:
            reasons.append("geometric_channel_unavailable")
        rows.append(
            dict(
                criterion_id=item,
                required_joint_indices=selected,
                side=side,
                simultaneous_observation=coverage,
                eligible=not reasons,
                reasons=reasons,
                rule_grade=grade if not reasons else None,
                measured=measured,
            )
        )
    return dict(
        items=rows,
        total_score=(
            float(
                sum(
                    r["rule_grade"] * w / 2
                    for r, w in zip(rows, WEIGHTS, strict=True)
                )
            )
            if all(r["eligible"] for r in rows)
            else None
        ),
        contact_is_proxy=True,
    )


def features(
    sequence: np.ndarray, item_index: int, side: str
) -> tuple[np.ndarray, list[str]]:
    """그 항목이 요구하는 관절만 남겨 공통 기하 특징을 계산한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        item_index: 네 항목 중 0~3 정수 인덱스.
        side: left 또는 right. 오른쪽 입력도 왼쪽 인덱스로 정규화한다.

    Returns:
        tuple: 해당 관절만으로 계산한 특징 벡터와 영어 이름 목록.
        각도는 도, 거리는 몸통 비율, 관측률은 0~1이다.

    Raises:
        ValueError: 관절 배열·항목·측면이 유효하지 않은 경우.
    """
    seq = np.asarray(sequence, dtype=float)
    if (
        seq.ndim != 3
        or seq.shape[1:] != (33, 2)
        or not len(seq)
        or item_index not in range(4)
        or side not in ("left", "right")
    ):
        raise ValueError("Invalid sequence, item or side")
    masked = np.full_like(seq, np.nan)
    for joint in JOINTS[item_index]:
        masked[:, joint] = seq[:, joint + (side == "right")]
    return geometry.model_features(masked)
