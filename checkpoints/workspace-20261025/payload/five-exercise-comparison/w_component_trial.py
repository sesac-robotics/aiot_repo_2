"""보이는 W 성분의 부분 수행을 보존하는 별도 개발 규칙이다.

기존 규칙은 양팔 모두 손 위치를 놓치면 굽힘이 있어도 0을 부여했다.
양팔의 뚜렷한 굽힘·낮은 팔꿈치가 보이면 부분 수행 1을 보존한다.
2등급 경계와 관측 계약은 기존판과 동일하며 3D 외회전은 평가하지 않는다.
"""

import numpy as np

import w_morphology

VERSION = "W-explicit-partial-components-v1-development"


def evaluate(sequence: np.ndarray, start: int, end: int) -> dict:
    """같은 유지 구간의 기존 등급과 부분 성분 근거를 반환한다.

    Args:
        sequence: N×33×2 픽셀 관절. 미관측은 NaN이다.
        start: 시작 표본 인덱스, 포함.
        end: 끝 표본 인덱스, 포함.

    Returns:
        dict: grade 0·1·2 또는 None과 기존판 근거. 양팔 굽힘이 기존의
        뚜렷한 굽힘 경계 20도 이상이고 팔꿈치 상승이 기존 부분 허용 비율
        0.5 이하이면, 기존 0등급만 부분 수행 1로 바꾼다. 정상 2는 늘리지 않는다.

    Raises:
        ValueError: 좌표 형상 또는 포함 구간 오류.
    """
    result = w_morphology.evaluate(sequence, start, end)
    result["baseline_version"] = result["version"]
    result["version"] = VERSION
    result["baseline_grade"] = result["grade"]
    arms = result["arms"]
    partial = len(arms) == 2 and all(
        a.get("projected_bend_deg", -1) >= w_morphology.POLICY["bend_good_deg"]
        and a.get("elbow_up_upper_arm_ratio", 1)
        <= w_morphology.POLICY["elbow_up_partial_arm_ratio"]
        for a in arms.values()
    )
    if result["grade"] == 0 and partial:
        result["grade"] = 1
    result["bilateral_partial_component_present"] = partial
    result["common_policy_modified"] = False
    return result
