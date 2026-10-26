"""직접 점수와 독립인 측면 지지·발 들림 관측 특징을 계산한다.

접촉 센서나 실제 하중의 측정이 아닌 화면 기하 대리값이다. 필수 관절이
가려지거나 정면 단축 투영이면 값을 만들지 않으며 결측을 보간하지 않는다.
"""

import numpy as np

import geometry


def plank_clearance(sequence: np.ndarray) -> dict:
    """팔꿈치–발끝 선 위 골반·무릎 높이와 무릎 각도를 요약한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 관측 불가는 NaN이다.

    Returns:
        dict: eligible·feature(22개)·measurement·side·holds. 특징은
        무릎 각도(도), 골반·무릎의 지지선 위 높이(몸통 길이 비율)의
        5·10·25·50·75·90·95백분위와 동시 관측률(0~1)이다.
        measurement는 무릎 각도·골반 높이·무릎 높이의 10백분위다.
        최소 5표본·80% 동시 관측·수평 지지선 길이 몸통의 0.8배를 요구한다.

    Raises:
        ValueError: 비어 있거나 N×33×2 형상이 아닌 좌표 입력.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Nonempty N x 33 x 2 pixel coordinates required")
    channels = geometry.channels(seq)
    choices = []
    for offset, side in ((0, "left"), (1, "right")):
        shoulder, elbow, hip, knee, ankle, toe = [
            j + offset for j in (11, 13, 23, 25, 27, 31)
        ]
        torso = np.linalg.norm(seq[:, shoulder] - seq[:, hip], axis=1)
        span = seq[:, toe, 0] - seq[:, elbow, 0]
        valid = np.isfinite(
            seq[:, [shoulder, elbow, hip, knee, ankle, toe]]
        ).all((1, 2))
        valid &= (torso > 3) & (abs(span) >= 0.8 * torso)
        coverage = float(valid.mean())
        if len(seq) < 5 or coverage < 0.8:
            continue
        heights = []
        for joint in (hip, knee):
            fraction = (seq[valid, joint, 0] - seq[valid, elbow, 0]) / span[
                valid
            ]
            line_y = seq[valid, elbow, 1] + fraction * (
                seq[valid, toe, 1] - seq[valid, elbow, 1]
            )
            heights.append((line_y - seq[valid, joint, 1]) / torso[valid])
        values = [channels[side + "_knee_deg"][valid], *heights]
        if not all(np.isfinite(v).all() for v in values):
            continue
        vector = [
            float(x)
            for v in values
            for x in np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
        ]
        choices.append(
            dict(
                eligible=True,
                feature=vector + [coverage],
                measurement=[float(np.percentile(v, 10)) for v in values],
                side=side,
                coverage=coverage,
                holds=[],
                contact_is_proxy=True,
            )
        )
    return (
        max(choices, key=lambda r: r["coverage"])
        if choices
        else dict(
            eligible=False,
            feature=None,
            measurement=None,
            side=None,
            holds=["support_line_joints_or_side_extent_unavailable"],
            contact_is_proxy=True,
        )
    )


def plank_support_grade(measurement: list[float] | None) -> int | None:
    """무릎 각도와 지지선 높이의 고정 조건으로 추가 지지를 판정한다.

    Args:
        measurement: 무릎 각도(도)·골반 높이·무릎 높이(몸통 길이 비율),
            또는 관측 불가 None. 경계는 개발 실험의 고정 제안값이다.

    Returns:
        int | None: 모든 조건의 최저 0·1·2 등급 또는 관측 불가 None.
        정상 경계는 165도·0.12·0.03, 부분 경계는 145도·0.04·0.005다.
        바닥 접촉·하중이나 전문가 정확도를 인증하지 않는다.

    Raises:
        ValueError: 세 개의 유한 측정값이 아닌 입력.
    """
    if measurement is None:
        return None
    if len(measurement) != 3 or not np.isfinite(measurement).all():
        raise ValueError("Three finite support measurements required")
    return min(
        2 if x >= a else 1 if x >= b else 0
        for x, a, b in zip(measurement, (165, 0.12, 0.03), (145, 0.04, 0.005))
    )


def squat_foot_change(
    sequence: np.ndarray, allow_partial_top: bool = False
) -> dict:
    """선 자세와 하단을 구분하여 발 길이 기준 뒤꿈치 상승을 계산한다.

    Args:
        sequence: N×33×2 원본 픽셀 관절. 관측 불가는 NaN이다.
        allow_partial_top: True이면 완전 신전 대신 최대 125도 이상·상하
            움직임 25도 이상의 부분 복귀도 발 변화 측정에 허용한다.
            복귀 항목의 등급이나 완결 반복·총점 승인과는 별개다.

    Returns:
        dict: eligible·feature(5개)·side·holds. 특징은 상단 발 기울기,
        하단 발 기울기, 그 차이(도), 발 길이의 사분위 범위/중앙값,
        상단 대비 하단 뒤꿈치 높이 상승/고정 발 길이 중앙값이다.
        최소 8표본·관절 동시 관측 80%·상하단 각 3표본·발 길이 3픽셀,
        무릎 최대 150도 이상과 최소 150도 미만의 움직임을 요구한다.

    Raises:
        ValueError: 비어 있거나 N×33×2 형상이 아닌 좌표 입력.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Nonempty N x 33 x 2 pixel coordinates required")
    channels = geometry.channels(seq)
    choices = []
    for offset, side in ((0, "left"), (1, "right")):
        hip, knee, ankle, heel, toe = [j + offset for j in (23, 25, 27, 29, 31)]
        delta = seq[:, toe] - seq[:, heel]
        length = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(seq[:, [hip, knee, ankle, heel, toe]]).all(
            (1, 2)
        ) & (length >= 3)
        if len(seq) < 8 or valid.mean() < 0.8:
            continue
        flex = channels[side + "_knee_deg"][valid]
        phase_observed = (
            flex.max() >= 125 and flex.max() - flex.min() >= 25
            if allow_partial_top
            else flex.max() >= 150
        )
        if (
            not np.isfinite(flex).all()
            or not phase_observed
            or flex.min() >= 150
        ):
            continue
        top, bottom = flex >= np.percentile(flex, 75), flex <= np.percentile(
            flex, 25
        )
        if min(top.sum(), bottom.sum()) < 3:
            continue
        angles = np.degrees(np.arctan2(delta[valid, 1], abs(delta[valid, 0])))
        first, last = float(np.median(angles[top])), float(
            np.median(angles[bottom])
        )
        scale = float(np.median(length[valid]))
        heights = delta[valid, 1]
        lift = max(
            0.0,
            float(np.median(heights[bottom]) - np.median(heights[top])) / scale,
        )
        variability = float(
            np.subtract(*np.percentile(length[valid], [75, 25])) / scale
        )
        if variability > 0.5:
            continue
        choices.append(
            dict(
                eligible=True,
                feature=[first, last, last - first, variability, lift],
                side=side,
                coverage=float(valid.mean()),
                holds=[],
            )
        )
    return (
        max(choices, key=lambda r: r["coverage"])
        if choices
        else dict(
            eligible=False,
            feature=None,
            side=None,
            holds=["foot_tracking_or_top_bottom_observation_unavailable"],
        )
    )
