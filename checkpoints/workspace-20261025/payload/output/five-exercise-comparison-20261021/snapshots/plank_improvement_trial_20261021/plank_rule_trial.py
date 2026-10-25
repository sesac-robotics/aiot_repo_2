"""측면 투영 골반 이탈·골반 높이 대리값을 추가하는 별도 규칙 실험이다.

경계는 검증 전 공학 초안이다. 접촉·하중을 확정하지 않으며 원래 공통
규칙·라벨·서비스를 변경하지 않는다. 알려진 개발 실패를 보고 설계했다.
"""

import copy

import numpy as np

import rule_score

VERSION = "plank-side-deviation-clearance-v1-development"
POLICY = dict(
    trunk_good_below=-0.04,
    trunk_good_above=0.10,
    trunk_partial_below=-0.08,
    trunk_partial_above=0.16,
    hip_clearance_good=0.40,
    hip_clearance_partial=0.25,
    minimum_observation=0.8,
)


def evaluate(sequence: np.ndarray) -> dict:
    """동일 관절에서 골반 이탈과 바닥선 높이 대리값으로 항목을 계산한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 미관측은 NaN이다.

    Returns:
        dict: 네 항목의 0·1·2 또는 None 등급과 100점 배점 합계.
        몸통 이탈은 어깨–발목 길이 비율, 골반 높이는 몸통 길이 비율이다.
        카메라가 수평에 가까운 측면에 한정한다. 미관측은 보간하지 않고
        항목을 None으로 유지하며 골반 높이를 실제 접촉 사실로 해석하지 않는다.

    Raises:
        ValueError: 좌표 형상 또는 빈 시계열 오류.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 pixels")
    ids = [
        "trunk_line",
        "elbow_sagittal_position",
        "leg_and_body_support",
        "support_continuity",
    ]
    result = copy.deepcopy(
        rule_score.evaluate("plank", seq, None, ids, [40, 25, 20, 15])
    )
    side = result["evidence"]["trunk_line"]["channel"].split("_")[0]
    shoulder, elbow, hip, ankle, toe = (
        [11, 13, 23, 27, 31] if side == "left" else [12, 14, 24, 28, 32]
    )
    sh, hp, an = (seq[:, i] for i in [shoulder, hip, ankle])
    axis = an - sh
    length = np.linalg.norm(axis, axis=1)
    normal = np.stack([axis[:, 1], -axis[:, 0]], axis=1)
    normal *= np.where(normal[:, 1] <= 0, 1, -1)[:, None]
    deviation = np.divide(
        np.sum((hp - sh) * normal, axis=1),
        length**2,
        out=np.full(len(seq), np.nan),
        where=length > 1,
    )
    finite = deviation[np.isfinite(deviation)]
    trunk = None
    bounds = None
    if (
        len(seq) >= 5
        and len(finite) / len(seq) >= POLICY["minimum_observation"]
    ):
        low, high = np.percentile(finite, [10, 90])
        bounds = [float(low), float(high)]
        trunk = (
            2
            if low >= POLICY["trunk_good_below"]
            and high <= POLICY["trunk_good_above"]
            else (
                1
                if low >= POLICY["trunk_partial_below"]
                and high <= POLICY["trunk_partial_above"]
                else 0
            )
        )
    el, foot = (seq[:, i] for i in [elbow, toe])
    floor = foot - el
    distance = np.linalg.norm(floor, axis=1)
    normal = np.stack([floor[:, 1], -floor[:, 0]], axis=1)
    normal *= np.where(normal[:, 1] <= 0, 1, -1)[:, None]
    torso = np.linalg.norm(sh - hp, axis=1)
    gap = np.divide(
        np.sum((hp - el) * normal, axis=1),
        distance * torso,
        out=np.full(len(seq), np.nan),
        where=(distance > 1) & (torso > 1),
    )
    observed = gap[np.isfinite(gap)]
    clearance = None
    measured = None
    if (
        len(seq) >= 5
        and len(observed) / len(seq) >= POLICY["minimum_observation"]
    ):
        measured = float(np.percentile(observed, 10))
        clearance = (
            2
            if measured >= POLICY["hip_clearance_good"]
            else 1 if measured >= POLICY["hip_clearance_partial"] else 0
        )
    knee = result["levels"][2]
    support = (
        min(knee, clearance)
        if knee is not None and clearance is not None
        else None
    )
    result["levels"][0] = trunk
    result["levels"][2] = support
    result.update(
        version=VERSION,
        score=(
            float(np.dot(result["levels"], [40, 25, 20, 15]) / 2)
            if all(x is not None for x in result["levels"])
            else None
        ),
        policy=POLICY.copy(),
        expert_validated=False,
        contact_is_proxy=True,
        common_policy_modified=False,
    )
    result["hold_items"] = [
        name
        for name, grade in zip(ids, result["levels"], strict=True)
        if grade is None
    ]
    result["evidence"]["trial"] = dict(
        selected_side=side,
        trunk_height_p10_p90_body_ratio=bounds,
        hip_floor_clearance_p10_torso_ratio=measured,
        clearance_grade=clearance,
        note_ko="팔꿈치–발끝 관절선은 바닥·실제 접촉의 대리값이다. 투영·체형·관절 중심 편향을 포함한다.",
    )
    return result
