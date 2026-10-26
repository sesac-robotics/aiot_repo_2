"""3종 항목별 관측·기하 특징과 독립 규칙 측정값을 계산한다.

점수 AI와 Rule Base에 같은 관절·구간·항목 관측 정책을 적용한다.
필요하지 않은 신체 잘림은 허용하지만 필요한 관절 누락을 채우지 않는다.
"""

import copy

import numpy as np

import geometry

VERSION = "observable-item-geometry-v3-static-and-dynamic-medial-knee"


def apply_view_policy(
    exercise: str, notes: list[dict], views: list[str]
) -> list[dict]:
    """명시된 촬영 구도가 해당 항목을 식별할 수 있는지 양쪽 방식에 적용한다.

    Args:
        exercise: 운동 코드. 현재 숄더 프레스 정면 몸통 항목을 검사한다.
        notes: 기하 관측 결과 네 항목. 입력은 변경하지 않는다.
        views: 입력 영상의 확인된 촬영 방향. 점수 라벨과 독립인 메타데이터다.

    Returns:
        list[dict]: 사선 숄더 프레스의 몸통 좌우 항목을 보류한 사본.
        사선에서 보이는 양팔 항목까지 함께 제거하지 않는다.
    """
    result = copy.deepcopy(notes)
    if exercise == "shoulder" and "front_oblique" in views:
        result[3]["eligible"] = False
        result[3]["holds"].append("frontal_trunk_requires_front_view")
    return result


def mask_image_border(
    sequence: np.ndarray, width: float, height: float
) -> np.ndarray:
    """영상 경계 2% 안쪽 관절을 불확실로 처리한 좌표 사본을 반환한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표.
        width: 디코더가 반환한 원본 가로 픽셀 수.
        height: 디코더가 반환한 원본 세로 픽셀 수.

    Returns:
        np.ndarray: 경계 관절만 NaN으로 바꾼 사본. 얼굴·머리처럼 항목에
        필요 없는 관절 누락은 describe가 전신 보류로 확대하지 않는다.

    Raises:
        ValueError: 유효하지 않은 원본 크기 또는 좌표 형상.
    """
    if (
        width <= 0
        or height <= 0
        or sequence.ndim != 3
        or sequence.shape[1:] != (33, 2)
    ):
        raise ValueError("Valid original dimensions and N x 33 x 2 required")
    result = sequence.copy()
    low = np.array([width, height]) * 0.02
    high = np.array([width, height]) * 0.98
    border = ((result < low) | (result > high)).any(axis=2)
    result[border] = np.nan
    return result


def _observed(
    sequence: np.ndarray | None, joints: list[int], minimum: int = 5
) -> float:
    if sequence is None or len(sequence) < minimum:
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

    def add(
        name,
        seq,
        required,
        series,
        quantile,
        higher,
        side,
        extras=(),
        minimum=5,
    ):
        observed = _observed(seq, required, minimum)
        measured = _stat(series, quantile) if len(series) else float("nan")
        feature = [_stat(series, q) for q in (5, 10, 25, 50, 75, 90, 95)]
        feature.extend([observed, *extras])
        holds = []
        if observed < 0.8:
            holds.append(
                "required_joints_not_simultaneously_observed_in_80_percent"
            )
        if not np.isfinite(feature).all():
            holds.append("required_measurement_unavailable")
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
                holds=holds,
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
                end = min(5, max(2, round(len(primary) * 0.15)))
                seq, series = primary[-end:], series[-end:]
            add(
                name,
                seq,
                [j + offset for j in required],
                series,
                quantile,
                higher,
                side,
                minimum=2 if name == "return_extension" else 5,
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
            if name == "descent_range" and primary is not None:
                thigh = primary[:, 25 + offset] - primary[:, 23 + offset]
                series = thigh[:, 1] / np.maximum(
                    np.linalg.norm(thigh, axis=1), 1
                )
            if name == "return_extension" and primary is not None:
                end = min(5, max(2, round(len(primary) * 0.15)))
                seq, series = primary[-end:], series[-end:]
            add(
                name,
                seq,
                [j + offset for j in required],
                series,
                quantile,
                higher,
                side,
                minimum=2 if name == "return_extension" else 5,
            )
        required = [11, 12, 23, 24, 25, 26, 27, 28]
        required_side = [11, 23, 29, 31]
        offset = side_for(required_side)
        side = ("left", "right")[offset]
        heels = channels.get(side + "_heel_lift", empty)
        heel_change = (
            _stat(heels - _stat(heels[:3], 50), 90)
            if len(heels)
            else float("nan")
        )
        heel = (
            max(0.0, heel_change) if np.isfinite(heel_change) else float("nan")
        )
        offsets = []
        if front is not None:
            center = (front[:, 23, 0] + front[:, 24, 0]) / 2
            torso = np.linalg.norm(
                (front[:, 11] + front[:, 12] - front[:, 23] - front[:, 24]) / 2,
                axis=1,
            )
            for knee_joint, ankle_joint in [(25, 27), (26, 28)]:
                hip_joint = knee_joint - 2
                relative = (
                    np.abs(front[:, knee_joint, 0] - center)
                    - np.abs(front[:, ankle_joint, 0] - center)
                ) / np.maximum(torso, 1)
                drift = np.maximum(0, _stat(relative[:3], 50) - relative)
                hip, knee_point, ankle = (
                    front[:, hip_joint],
                    front[:, knee_joint],
                    front[:, ankle_joint],
                )
                height = ankle[:, 1] - hip[:, 1]
                fraction = np.divide(
                    knee_point[:, 1] - hip[:, 1],
                    height,
                    out=np.full(len(height), np.nan),
                    where=np.abs(height) > 1,
                )
                expected_x = hip[:, 0] + np.clip(fraction, 0, 1) * (
                    ankle[:, 0] - hip[:, 0]
                )
                inward = (
                    np.abs(expected_x - center)
                    - np.abs(knee_point[:, 0] - center)
                ) / np.maximum(torso, 1)
                angle = geometry.angle_deg(hip, knee_point, ankle)
                static = np.where(inward > 0.025, (180 - angle) / 180, 0.0)
                static[~np.isfinite(inward) | ~np.isfinite(angle)] = np.nan
                offsets.append(np.maximum(drift, static))
        knee = np.maximum(*offsets) if all(len(a) for a in offsets) else empty
        add("knee_foot", front, required, knee, 90, False, side, extras=(heel,))
        result[-1][
            "measurement_kind"
        ] = "maximum_medial_drift_torso_ratio_or_inward_frontal_bend_fraction_of_180_degrees_plus_relative_heel_lift"
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
        extension = np.minimum(
            channels.get("left_elbow_deg", empty),
            channels.get("right_elbow_deg", empty),
        )
        upper = np.minimum(upper, extension)
        add("overhead_reach", primary, arm, upper, 90, True, "front")
        result[-1][
            "measurement_kind"
        ] = "minimum_bilateral_elevation_and_elbow_extension_degrees"
        if not peak_ok:
            result[-1]["eligible"] = False
            result[-1]["holds"].append("wrists_not_visible_at_overhead_peak")
        end = (
            min(5, max(2, round(len(primary) * 0.15)))
            if primary is not None
            else 2
        )
        lower = np.maximum(
            channels.get("left_elbow_deg", empty),
            channels.get("right_elbow_deg", empty),
        )[-end:]
        lower_elevation = np.maximum(
            channels.get("left_elevation_deg", empty),
            channels.get("right_elevation_deg", empty),
        )[-end:]
        lower = np.maximum(lower, lower_elevation)
        add(
            "lower_return",
            primary[-end:] if primary is not None else None,
            [11, 12, 13, 14, 15, 16, 23, 24],
            lower,
            50,
            False,
            "front",
            minimum=2,
        )
        phase_gap = empty
        moving_arms = []
        if primary is not None:
            phase = []
            for shoulder, wrist, hip in [(11, 15, 23), (12, 16, 24)]:
                torso = np.linalg.norm(
                    primary[:, shoulder] - primary[:, hip], axis=1
                )
                height = (
                    primary[:, shoulder, 1] - primary[:, wrist, 1]
                ) / np.maximum(torso, 1)
                low, high_value = _stat(height, 5), _stat(height, 95)
                moving_arms.append(
                    bool(
                        np.isfinite(high_value - low)
                        and high_value - low > 0.15
                    )
                )
                phase.append(
                    (height - low) / (high_value - low)
                    if high_value - low > 0.15
                    else np.where(np.isfinite(height), 0.0, np.nan)
                )
            phase_gap = np.abs(phase[0] - phase[1])
        add(
            "bilateral_timing",
            primary,
            [11, 12, 15, 16, 23, 24],
            phase_gap,
            90,
            False,
            "front",
        )
        result[-1][
            "measurement_kind"
        ] = "normalized_left_right_lift_phase_difference"
        if moving_arms and not any(moving_arms):
            result[-1]["eligible"] = False
            result[-1]["holds"].append("no_observed_lift_motion")
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
