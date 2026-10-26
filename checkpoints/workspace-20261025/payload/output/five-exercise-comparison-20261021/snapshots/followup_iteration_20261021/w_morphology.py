"""W의 화면 모양을 양팔 굽힘·손 위치·좌우 차이로 평가하는 개발 규칙이다.

정확한 3D 각도가 아닌 라벨에 명시한 화면 모양을 평가한다. 수치 경계는
공학 초안이며 전문가 검증값이 아니다. 제공자 문자열·점수 라벨을 받지 않는다.
"""

import numpy as np

import geometry

VERSION = "W-visible-morphology-v2-development"
POLICY = dict(
    observed_fraction=0.8,
    minimum_samples=3,
    bend_good_deg=20.0,
    bend_partial_deg=8.0,
    hand_good_torso_ratio=0.05,
    hand_partial_torso_ratio=-0.05,
    symmetry_good_deg=35.0,
    elbow_up_good_arm_ratio=0.25,
    elbow_up_partial_arm_ratio=0.5,
)


def evaluate(sequence: np.ndarray, start: int, end: int) -> dict:
    """W 유지 구간의 양팔 화면 모양과 0·1·2등급 또는 보류를 반환한다.

    Args:
        sequence: N×33×2 픽셀 관절 좌표. 미관측은 NaN이다.
        start: 유지 구간 시작 인덱스, 포함.
        end: 유지 구간 끝 인덱스, 포함.

    Returns:
        dict: 양팔별 굽힘(도)·손 높이(몸통 길이 비율)·관측률,
        팔꿈치 상승(상완 길이 비율), 좌우 굽힘 차이(도), 등급과 미검증 경계·근거. 손 높이는 몸통의
        어깨 방향이 양수이며 화면 회전에 불변이다. 3D 외회전·견갑·하중은
        평가하지 않는다. 관측률 80% 미만 또는 3표본 미만은 grade=None이다.

    Raises:
        ValueError: 좌표 형상 또는 구간 범위가 잘못된 경우.
    """
    seq = np.asarray(sequence, dtype=float)
    if (
        seq.ndim != 3
        or seq.shape[1:] != (33, 2)
        or not 0 <= start <= end < len(seq)
    ):
        raise ValueError("Invalid W coordinates or window")
    seq = seq[start : end + 1]
    result = dict(
        version=VERSION,
        grade=None,
        policy=POLICY.copy(),
        arms={},
        independent_validated=False,
        scope="visible_projected_W_shape_only",
        hold_reason=None,
    )
    if len(seq) < POLICY["minimum_samples"]:
        result["hold_reason"] = "insufficient_dwell_samples"
        return result
    shoulder_mid = (seq[:, 11] + seq[:, 12]) / 2
    hip_mid = (seq[:, 23] + seq[:, 24]) / 2
    up = shoulder_mid - hip_mid
    torso = np.linalg.norm(up, axis=1)
    valid_torso = np.isfinite(torso) & (torso > 1)
    up = np.divide(
        up,
        torso[:, None],
        out=np.full_like(up, np.nan),
        where=valid_torso[:, None],
    )
    grades = []
    bends = []
    for side, indices in (("left", (11, 13, 15)), ("right", (12, 14, 16))):
        shoulder, elbow, wrist = (seq[:, i] for i in indices)
        valid = np.isfinite(seq[:, indices]).all((1, 2)) & valid_torso
        coverage = float(valid.mean())
        if coverage < POLICY["observed_fraction"]:
            result["arms"][side] = dict(observed_fraction=coverage)
            result["hold_reason"] = "insufficient_bilateral_joint_observation"
            return result
        bend = 180 - geometry.angle_deg(shoulder, elbow, wrist)
        height = np.sum((wrist - elbow) * up, axis=1) / torso
        upper_arm = elbow - shoulder
        length = np.linalg.norm(upper_arm, axis=1)
        elbow_up = np.divide(
            np.sum(upper_arm * up, axis=1),
            length,
            out=np.full_like(length, np.nan),
            where=length > 1,
        )
        finite = (
            valid
            & np.isfinite(bend)
            & np.isfinite(height)
            & np.isfinite(elbow_up)
        )
        if finite.mean() < POLICY["observed_fraction"]:
            result["hold_reason"] = "degenerate_arm_or_trunk_geometry"
            return result
        measured_bend = float(np.median(bend[finite]))
        measured_height = float(np.median(height[finite]))
        bend_grade = (
            2
            if measured_bend >= POLICY["bend_good_deg"]
            else 1 if measured_bend >= POLICY["bend_partial_deg"] else 0
        )
        hand_grade = (
            2
            if measured_height >= POLICY["hand_good_torso_ratio"]
            else (
                1
                if measured_height >= POLICY["hand_partial_torso_ratio"]
                else 0
            )
        )
        measured_elbow_up = float(np.median(elbow_up[finite]))
        # W는 팔꿈치가 어깨 위로 크게 올라가는 Y형 굽힘과 구별한다.
        elbow_grade = (
            2
            if measured_elbow_up <= POLICY["elbow_up_good_arm_ratio"]
            else (
                1
                if measured_elbow_up <= POLICY["elbow_up_partial_arm_ratio"]
                else 0
            )
        )
        grade = min(bend_grade, hand_grade, elbow_grade)
        result["arms"][side] = dict(
            observed_fraction=coverage,
            projected_bend_deg=measured_bend,
            hand_above_elbow_torso_ratio=measured_height,
            elbow_up_upper_arm_ratio=measured_elbow_up,
            elbow_position_grade=elbow_grade,
            bend_grade=bend_grade,
            hand_grade=hand_grade,
            grade=grade,
        )
        grades.append(grade)
        bends.append(measured_bend)
    difference = abs(bends[0] - bends[1])
    symmetry = 2 if difference <= POLICY["symmetry_good_deg"] else 1
    # 한쪽만 충족하거나 양쪽 모양이 다른 경우는 기준의 부분 수행이다.
    grade = (
        2 if min(grades) == 2 and symmetry == 2 else 1 if max(grades) > 0 else 0
    )
    result.update(
        bilateral_bend_difference_deg=difference,
        bilateral_shape_grade=symmetry,
        grade=grade,
    )
    return result
