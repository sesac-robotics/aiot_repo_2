"""5종의 관절 기하 대리값으로 항목 등급을 추정하는 실험 규칙이다.

등급 0·1·2와 배점은 라벨 기준을 따르지만 각도 경계는 미검증 초안이다.
영상 속 실제 하중·바닥 접촉·견갑 기능은 관절만으로 확정하지 않는다.
이 모듈의 출력은 AI 학습 정답으로 사용하지 않는다.
"""

import numpy as np

import geometry
import ytw_phases

VERSION = "geometric-proxy-rules-v2-phase-dwell-unvalidated"


def _stat(channel: np.ndarray, percentile: float) -> float | None:
    good = channel[np.isfinite(channel)]
    if len(channel) < 5 or len(good) / len(channel) < 0.8:
        return None
    return float(np.percentile(good, percentile))


def _grade(
    value: float | None, good: float, partial: float, higher: bool = True
) -> int | None:
    if value is None:
        return None
    if higher:
        return 2 if value >= good else 1 if value >= partial else 0
    return 2 if value <= good else 1 if value <= partial else 0


def _side(values: dict[str, np.ndarray]) -> str:
    return max(
        ("left", "right"),
        key=lambda side: sum(
            np.isfinite(values[f"{side}_{joint}_deg"]).sum()
            for joint in ("elbow", "knee", "line")
        ),
    )


def evaluate(
    exercise: str,
    primary: np.ndarray,
    frontal: np.ndarray | None,
    criterion_ids: list[str],
    weights: list[int],
) -> dict:
    """같은 구간의 관절에서 항목별 추정 등급과 보류 사유를 반환한다.

    Args:
        exercise: 5종 중 영어 운동 코드.
        primary: N×33×2 주 시점 픽셀 좌표. 미관측은 NaN이다.
        frontal: 스쿼트 동기화 정면 좌표. 없으면 관련 항목을 보류한다.
        criterion_ids: 적용할 기준 버전의 항목 ID 순서.
        weights: 같은 순서의 합계 100 배점.

    Returns:
        dict: 0·1·2 또는 null 등급, 합산 점수, 경계·측정값·보류 항목.
        하나라도 판정 불가이면 총점은 None이다.

    Raises:
        ValueError: 운동 코드·항목·배점 또는 좌표 형상이 잘못된 경우.
    """
    if len(criterion_ids) != len(weights) or sum(weights) != 100:
        raise ValueError("Criterion IDs and weights must match and sum to 100")
    values = geometry.channels(primary)
    side = _side(values)
    end = max(5, len(primary) // 4)
    levels = {}
    evidence = {}

    def use(
        item: str,
        channel: str,
        percentile: float,
        good: float,
        partial: float,
        higher: bool = True,
        tail: bool = False,
    ) -> None:
        source = values[channel][-end:] if tail else values[channel]
        value = _stat(source, percentile)
        levels[item] = _grade(value, good, partial, higher)
        evidence[item] = dict(
            channel=channel,
            percentile=percentile,
            measured=value,
            good_threshold=good,
            partial_threshold=partial,
            higher_is_better=higher,
            tail_only=tail,
        )

    if exercise in ("pushups", "plank"):
        use("trunk_line", side + "_line_deg", 10, 168, 155)
    if exercise == "pushups":
        use("descent_range", side + "_elbow_deg", 10, 95, 125, False)
        use("return_extension", side + "_elbow_deg", 90, 160, 140, tail=True)
        use("support", side + "_knee_floor_gap", 10, 0.15, 0.05)
    elif exercise == "squats":
        use("descent_range", side + "_hip_above_knee", 10, 0.1, 0.5, False)
        use("return_extension", side + "_knee_deg", 90, 165, 145, tail=True)
        heel = _stat(values[side + "_heel_lift"], 90)
        if frontal is not None:
            front = geometry.channels(frontal)
            tilt = _stat(front["frontal_tilt_deg"], 90)
            offsets = [
                _stat(front[s + "_knee_ankle_offset"], 90)
                for s in ("left", "right")
            ]
            levels["trunk_balance"] = _grade(tilt, 8, 15, False)
            if heel is not None and all(x is not None for x in offsets):
                knee_grade = _grade(max(offsets), 0.25, 0.5, False)
                heel_grade = _grade(heel, 0.15, 0.3, False)
                levels["knee_foot"] = min(knee_grade, heel_grade)
            else:
                levels["knee_foot"] = None
            evidence["knee_foot"] = dict(
                knee_ankle_offsets=offsets,
                heel_lift_ratio=heel,
                warning_ko="투영 정렬·뒤꿈치 높이 대리값. 실제 접지 미검증.",
            )
            evidence["trunk_balance"] = dict(
                tilt_deg=tilt, thresholds_deg=[8, 15]
            )
    elif exercise == "plank":
        use(
            "elbow_sagittal_position",
            side + "_elbow_offset",
            90,
            0.2,
            0.4,
            False,
        )
        use("leg_and_body_support", side + "_knee_deg", 10, 165, 145)
        # 지지 중단은 단순히 나쁜 정렬이 유지되는 상태와 구별한다.
        knee = values[side + "_knee_deg"]
        good = knee[np.isfinite(knee)]
        if len(good) / len(knee) >= 0.8:
            spread = float(np.percentile(good, 95) - np.percentile(good, 5))
            levels["support_continuity"] = _grade(spread, 8, 20, False)
            evidence["support_continuity"] = dict(
                knee_angle_spread_deg=spread,
                warning_ko="관절 변화 대리값이며 실제 하중 지속은 미검증.",
            )
    elif exercise == "shoulder":
        upper = []
        lower = []
        for s in ("left", "right"):
            upper.append(
                _grade(_stat(values[s + "_elevation_deg"], 90), 150, 125)
            )
            lower.append(
                _grade(
                    _stat(values[s + "_elbow_deg"][-end:], 50), 110, 140, False
                )
            )
        levels["overhead_reach"] = min(upper) if None not in upper else None
        levels["lower_return"] = min(lower) if None not in lower else None
        use(
            "bilateral_timing",
            "bilateral_wrist_height_gap",
            90,
            0.15,
            0.35,
            False,
        )
        use("frontal_trunk_hold", "frontal_tilt_deg", 90, 8, 15, False)
        evidence["overhead_reach"] = dict(
            left_right_proxy_grades=upper, thresholds_deg=[150, 125]
        )
        evidence["lower_return"] = dict(
            left_right_proxy_grades=lower, thresholds_deg=[110, 140]
        )
    elif exercise == "ytw":
        phases = ytw_phases.detect(primary)
        evidence["phase_detection"] = phases
        for name, item, channel, target, tolerance in (
            ("T_attempt", "t_reach", "elevation", 90, 15),
            ("W_attempt", "w_shape", "elbow", 90, 20),
        ):
            window = phases["phases"].get(name)
            measured = None
            if window:
                a, b = window["start_index"], window["end_index"] + 1
                series = (
                    values["left_" + channel + "_deg"]
                    + values["right_" + channel + "_deg"]
                ) / 2
                subset = series[a:b]
                if np.isfinite(subset).mean() >= 0.8:
                    measured = float(np.nanmedian(subset))
            levels[item] = _grade(
                abs(measured - target) if measured is not None else None,
                tolerance,
                tolerance * 2,
                False,
            )
            evidence[item] = dict(
                measured_median_deg=measured,
                target_deg=target,
                good_error_deg=tolerance,
                partial_error_deg=tolerance * 2,
                window=window,
            )
        # 시도 구간의 식별을 정상 Y의 전방 깊이 확인으로 바꾸지 않는다.
        levels["y_reach"] = None
        evidence["y_reach"] = dict(
            hold_reason_ko="정면 시도 구간은 식별해도 Y 전방 도달·펴짐을 확정할 수 없음."
        )
        # 정면의 전경 깊이는 화면 관절만으로 안정적으로 판단하지 못한다.
        levels["hinge_and_trunk_hold"] = None
        evidence["hinge_and_trunk_hold"] = dict(
            hold_reason_ko="정면 투영만으로 전경 깊이 규칙을 검증하지 못함."
        )
    else:
        raise ValueError("Unknown exercise")
    ordered = [levels.get(item) for item in criterion_ids]
    return dict(
        version=VERSION,
        criterion_ids=criterion_ids,
        levels=ordered,
        score=(
            sum(weight * level / 2 for weight, level in zip(weights, ordered))
            if None not in ordered
            else None
        ),
        evidence=evidence,
        hold_items=[
            item for item, level in zip(criterion_ids, ordered) if level is None
        ],
        expert_validated=False,
        thresholds_validated=False,
        limitations_ko=[
            "모든 수치 경계는 검증 전 공학 초안이다.",
            "관절 기하의 대리값으로 실제 접지·하중을 확정하지 않는다.",
            "등급 null은 0점이나 만점으로 채우지 않는다.",
        ],
    )
