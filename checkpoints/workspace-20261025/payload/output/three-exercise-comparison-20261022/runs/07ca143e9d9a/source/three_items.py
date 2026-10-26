"""3종 항목별 관측·기하 특징과 독립 규칙 측정값을 계산한다.

점수 AI와 Rule Base에 같은 관절·구간·항목 관측 정책을 적용한다.
필요하지 않은 신체 잘림은 허용하지만 필요한 관절 누락을 채우지 않는다.
"""

import numpy as np

import geometry


def _observed(sequence: np.ndarray | None, joints: list[int]) -> float:
    if sequence is None or len(sequence) < 5:
        return 0.0
    return float(np.isfinite(sequence[:, joints]).all(axis=(1, 2)).mean())


def _stat(series: np.ndarray, quantile: float) -> float:
    finite = series[np.isfinite(series)]
    return (
        float(np.percentile(finite, quantile)) if len(finite) else float("nan")
    )


def describe(
    exercise: str, primary: np.ndarray | None, front: np.ndarray | None
) -> list[dict]:
    """각 항목의 동일 입력 특징과 규칙용 측정값·보류 근거를 반환한다.

    Args:
        exercise: pushups·squats·shoulder 중 운동 코드.
        primary: 측면 또는 숄더 정면의 N×33×2 픽셀 좌표. 없으면 None.
        front: 스쿼트의 동기화 정면 좌표 또는 단독 부분 정면. 없으면 None.

    Returns:
        list[dict]: 네 항목의 eligible·feature·measurement·side·required_joints.
        특징에는 관절 통계만 포함하며 규칙 등급·라벨은 포함하지 않는다.

    Raises:
        ValueError: 미지원 운동·관절 배열 형상 또는 동기화 길이 불일치.
    """
    if exercise not in ("pushups", "squats", "shoulder"):
        raise ValueError("Only three active exercises")
    for seq in (primary, front):
        if seq is not None and (seq.ndim != 3 or seq.shape[1:] != (33, 2)):
            raise ValueError("Expected N x 33 x 2 pixel landmarks")
    if primary is not None and front is not None and len(primary) != len(front):
        raise ValueError("Synchronized inputs require the same sample count")
    channels = geometry.channels(primary) if primary is not None else {}
    frontal = geometry.channels(front) if front is not None else {}
    result = []

    def add(name, seq, required, series, quantile, higher, side, extras=()):
        observed = _observed(seq, required)
        measured = _stat(series, quantile) if len(series) else float("nan")
        feature = [_stat(series, q) for q in (5, 10, 25, 50, 75, 90, 95)]
        feature.extend([observed, *extras])
        result.append(
            dict(
                item=name,
                eligible=bool(observed >= 0.8 and np.isfinite(feature).all()),
                measurement=measured if np.isfinite(measured) else None,
                feature=feature,
                higher_is_better=higher,
                side=side,
                required_joints=required,
                simultaneous_observation=observed,
                holds=(
                    []
                    if observed >= 0.8
                    else [
                        "required_joints_not_simultaneously_observed_in_80_percent"
                    ]
                ),
            )
        )

    def side_for(required):
        return max(
            (0, 1),
            key=lambda offset: _observed(
                primary, [j + offset for j in required]
            ),
        )

    empty = np.array([], dtype=float)
    if exercise == "pushups":
        for name, required, channel, quantile, higher in [
            ("trunk_line", [11, 23, 27], "line_deg", 10, True),
            ("descent_range", [11, 13, 15], "elbow_deg", 10, False),
            ("return_extension", [11, 13, 15], "elbow_deg", 90, True),
            ("support", [11, 15, 23, 25, 31], "knee_floor_gap", 10, True),
        ]:
            offset = side_for(required)
            side = ("left", "right")[offset]
            seq = primary
            series = channels.get(side + "_" + channel, empty)
            if name == "return_extension" and primary is not None:
                end = max(5, len(primary) // 4)
                seq, series = primary[-end:], series[-end:]
            add(
                name,
                seq,
                [j + offset for j in required],
                series,
                quantile,
                higher,
                side,
            )
    elif exercise == "squats":
        for name, required, channel, quantile, higher in [
            ("descent_range", [11, 23, 25], "hip_above_knee", 10, False),
            ("return_extension", [23, 25, 27], "knee_deg", 90, True),
        ]:
            offset = side_for(required)
            side = ("left", "right")[offset]
            seq = primary
            series = channels.get(side + "_" + channel, empty)
            if name == "return_extension" and primary is not None:
                end = max(5, len(primary) // 4)
                seq, series = primary[-end:], series[-end:]
            add(
                name,
                seq,
                [j + offset for j in required],
                series,
                quantile,
                higher,
                side,
            )
        required = [11, 12, 23, 24, 25, 26, 27, 28]
        required_side = [11, 23, 29, 31]
        offset = side_for(required_side)
        side = ("left", "right")[offset]
        heel = _stat(channels.get(side + "_heel_lift", empty), 90)
        offsets = [
            frontal.get(s + "_knee_ankle_offset", empty)
            for s in ("left", "right")
        ]
        knee = np.maximum(*offsets) if all(len(a) for a in offsets) else empty
        add("knee_foot", front, required, knee, 90, False, side, extras=(heel,))
        if _observed(primary, [j + offset for j in required_side]) < 0.8:
            result[-1]["eligible"] = False
            result[-1]["holds"].append("side_heel_and_toe_missing")
        add(
            "trunk_balance",
            front,
            [11, 12, 23, 24],
            frontal.get("frontal_tilt_deg", empty),
            90,
            False,
            "front",
        )
    else:
        peak_ok = False
        if primary is not None and len(primary) >= 5:
            elevation = np.maximum(
                channels["left_elevation_deg"], channels["right_elevation_deg"]
            )
            high = elevation >= _stat(elevation, 80)
            peak_ok = bool(
                high.any()
                and np.isfinite(primary[high][:, [15, 16]])
                .all(axis=(1, 2))
                .mean()
                >= 0.8
            )
        arm = [11, 12, 13, 14, 15, 16, 23, 24]
        upper = np.minimum(
            channels.get("left_elevation_deg", empty),
            channels.get("right_elevation_deg", empty),
        )
        add("overhead_reach", primary, arm, upper, 90, True, "front")
        if not peak_ok:
            result[-1]["eligible"] = False
            result[-1]["holds"].append("wrists_not_visible_at_overhead_peak")
        end = max(5, len(primary) // 4) if primary is not None else 5
        lower = np.maximum(
            channels.get("left_elbow_deg", empty),
            channels.get("right_elbow_deg", empty),
        )[-end:]
        add(
            "lower_return",
            primary[-end:] if primary is not None else None,
            [11, 12, 13, 14, 15, 16],
            lower,
            50,
            False,
            "front",
        )
        add(
            "bilateral_timing",
            primary,
            [11, 12, 15, 16, 23, 24],
            channels.get("bilateral_wrist_height_gap", empty),
            90,
            False,
            "front",
        )
        if not peak_ok:
            result[-1]["eligible"] = False
            result[-1]["holds"].append("bilateral_peak_not_fully_visible")
        add(
            "frontal_trunk_hold",
            primary,
            [11, 12, 23, 24],
            channels.get("frontal_tilt_deg", empty),
            90,
            False,
            "front",
        )
    return result


def total(levels: list[float | None], weights: list[int]) -> float | None:
    """네 등급이 모두 있을 때만 100점 배점 총점을 반환한다.

    Args:
        levels: 네 개의 0~2 연속 등급 또는 None.
        weights: 같은 순서의 합계 100인 배점.

    Returns:
        float | None: 총점. 한 항목이라도 미관측이면 None이며 재환산하지 않는다.

    Raises:
        ValueError: 항목 수·등급·배점 계약 오류.
    """
    if len(levels) != 4 or len(weights) != 4 or sum(weights) != 100:
        raise ValueError("Four items and 100 total weights required")
    if any(
        not np.isfinite(g) or not 0 <= g <= 2 for g in levels if g is not None
    ):
        raise ValueError("Grades must be finite and within zero to two")
    return None if None in levels else float(np.dot(levels, weights) / 2)
