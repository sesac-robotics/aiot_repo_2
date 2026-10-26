"""동일 프레임의 W 모양을 먼저 채점하는 시간 집계 개발 규칙이다.

항목별 중앙값이 서로 다른 프레임의 모양을 합성하는 문제를 조사한다.
공통 규칙을 변경하지 않으며 라벨에 맞춘 경계 최적화는 하지 않는다.
"""

import numpy as np

import w_morphology

VERSION = "W-frame-coincident-v1-development-trial"


def evaluate(sequence: np.ndarray, start: int, end: int) -> dict:
    """프레임별 동시 양팔 등급의 60% 우세와 부분 수행을 집계한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        start: 시작 인덱스, 포함.
        end: 끝 인덱스, 포함.

    Returns:
        dict: 0·1·2등급, 관측 비율, 등급별 표본 수. 3표본 미만 또는
        관측률 80% 미만은 None이며 보간·만점 채우기를 하지 않는다.
        0 또는 2가 60% 이상이면 해당 등급, 그 외는 부분 수행 1이다.
        유지 구간의 선택 정확도·3D 외회전을 검증하는 규칙은 아니다.

    Raises:
        ValueError: 좌표 형상 또는 포함 구간 범위 오류.
    """
    seq = np.asarray(sequence, dtype=float)
    if (
        seq.ndim != 3
        or seq.shape[1:] != (33, 2)
        or not 0 <= start <= end < len(seq)
    ):
        raise ValueError("Invalid coordinates or inclusive window")
    selected = seq[start : end + 1]
    grades = []
    for frame in selected:
        # 기존 형상 규칙의 최소 유지 길이만 충족시켜 한 프레임의 동시
        # 모양을 측정한다. 반복 프레임을 실제 시간 관측으로 세지 않는다.
        grade = w_morphology.evaluate(np.repeat(frame[None], 3, axis=0), 0, 2)[
            "grade"
        ]
        grades.append(grade)
    observed = [g for g in grades if g is not None]
    counts = {str(g): observed.count(g) for g in [0, 1, 2]}
    fraction = len(observed) / len(selected)
    grade = None
    reason = "insufficient_observed_dwell"
    if len(selected) >= 3 and fraction >= 0.8:
        grade = (
            0
            if counts["0"] / len(observed) >= 0.6
            else (2 if counts["2"] / len(observed) >= 0.6 else 1)
        )
        reason = None
    return dict(
        version=VERSION,
        grade=grade,
        frame_grades=grades,
        counts=counts,
        observed_fraction=fraction,
        hold_reason=reason,
        common_policy_modified=False,
        thresholds_validated=False,
    )
