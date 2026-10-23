"""학습된 명시 경계로 독립 Rule 점수를 계산한다. AI 출력을 읽지 않는다."""

import numpy as np


def rule_grade(note: dict, boundaries: tuple[float, ...]) -> int | None:
    """관측 통과 항목의 측정값과 명시 경계로 0·1·2 등급을 계산한다.

    Args:
        note: three_items.describe의 항목 기록. 각도는 도, 거리는 몸통 비율.
        boundaries: 만점·부분 경계. 무릎·발은 뒤꿈치 경계 두 개도 뒤에 둔다.
            기존 뒤꿈치는 몸통 길이 비율이며 side_foot 계열의 마지막
            특징은 고정 발 길이 대비 상단→하단 뒤꿈치 들림 변화 비율이다.

    Returns:
        int | None: 관측된 등급 또는 미판정. 무릎·발은 뒤꿈치 보조 조건도 적용.

    Raises:
        ValueError: 만점·부분 경계 순서가 측정 방향과 반대인 경우.
    """
    if not note["eligible"] or note["measurement"] is None:
        return None
    good, partial = boundaries[:2]
    high = note["higher_is_better"]
    if high and good <= partial or (not high and good >= partial):
        raise ValueError("Invalid ordered rule boundaries")
    value = note["measurement"]
    grade = (
        (2 if value >= good else 1 if value >= partial else 0)
        if high
        else 2 if value <= good else 1 if value <= partial else 0
    )
    if note["item"] == "knee_foot":
        heel = note["feature"][-1]
        heel_good, heel_partial = (
            boundaries[2:] if len(boundaries) == 4 else (0.025, 0.08)
        )
        if heel_good >= heel_partial:
            raise ValueError("Invalid ordered heel boundaries")
        grade = min(
            grade, 2 if heel <= heel_good else 1 if heel <= heel_partial else 0
        )
    return grade


def classify(features: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """기존 W 조건에 팔꿈치 아래 방향 경계를 더해 등급을 계산한다.

    Args:
        features: N×7 기존 관측 기하. 팔꿈치 상승 비율은 몸통 위 방향과
            상완의 내적이므로 -1에 가까우면 상완이 몸통 아래에 붙는다.
        policies: P×9. 기존 일곱 경계와 정상·부분 수행의 팔꿈치 상승 하한.

    Returns:
        np.ndarray: P×N 정수 등급. 추가 관절·AI 출력·정답 등급을 읽지 않는다.

    Raises:
        ValueError: 배열 형상·유한성·경계 순서 위반.
    """
    if (
        features.ndim != 2
        or features.shape[1] != 7
        or policies.ndim != 2
        or (policies.shape[1] != 9)
        or (not np.isfinite(features).all())
        or (not np.isfinite(policies).all())
        or np.any(policies[:, 8] > policies[:, 7])
    ):
        raise ValueError("Expected finite N x 7 and ordered P x 9")
    p = policies[:, :, None]
    arms = []
    for index in (0, 3):
        bend, hand, up = (features[None, :, index + j] for j in range(3))
        arms.append(
            np.minimum.reduce(
                [
                    np.where(
                        bend >= p[:, 0], 2, np.where(bend >= p[:, 1], 1, 0)
                    ),
                    np.where(
                        hand >= p[:, 2], 2, np.where(hand >= p[:, 3], 1, 0)
                    ),
                    np.where(up <= p[:, 4], 2, np.where(up <= p[:, 5], 1, 0)),
                    np.where(up >= p[:, 7], 2, np.where(up >= p[:, 8], 1, 0)),
                ]
            )
        )
    return np.where(
        (np.minimum(*arms) == 2) & (features[None, :, 6] <= p[:, 6]),
        2,
        np.where(np.maximum(*arms) > 0, 1, 0),
    )


def grades(x: np.ndarray, policy: tuple, item: int) -> np.ndarray:
    """관측된 화면 기하에 명시 조건을 적용해 0·1·2 등급을 반환한다.

    Args:
        x: N×6(Y·T·전경) 또는 N×7(W) 관측 특징. 결측은 허용하지 않는다.
        policy: 정상·부분 각도(도), 몸통 비율, 흔들림 경계.
        item: Y=0, T=1, W=2, 전경·흔들림=3.

    Returns:
        np.ndarray: 명시 조건의 최저 충족 등급. W는 양팔 굽힘·손 위치·
        팔꿈치 상하·대칭 조건이다. 전경은 화면 몸통 비율의 대리값이며
        숨은 실제 고관절 각도를 복원하거나 안전·교정 정답을 의미하지 않는다.

    Raises:
        ValueError: 미관측 특징·항목·경계 순서 불일치.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite observed features required")
    if item == 2:
        return classify(x, np.asarray([policy]))[0]
    if x.shape[1] != 6 or item not in (0, 1, 3):
        raise ValueError("Six observed projection features required")
    if item in (0, 1):
        good, partial, elbow_good, elbow_partial = policy
        if good <= partial or elbow_good <= elbow_partial:
            raise ValueError("Ordered elevation and elbow boundaries required")
        elevation = x[:, 1] if item == 0 else 90 - abs(x[:, 1] - 90)
        elbow = x[:, 4]
        return np.minimum(
            np.where(
                elevation >= good, 2, np.where(elevation >= partial, 1, 0)
            ),
            np.where(
                elbow >= elbow_good, 2, np.where(elbow >= elbow_partial, 1, 0)
            ),
        )
    good, partial, stable_good, stable_partial = policy
    if good <= partial or stable_good >= stable_partial:
        raise ValueError(
            "Ordered projected torso and stability boundaries required"
        )
    return np.minimum(
        np.where(x[:, 1] >= good, 2, np.where(x[:, 1] >= partial, 1, 0)),
        np.where(
            x[:, 5] - x[:, 3] <= stable_good,
            2,
            np.where(x[:, 5] - x[:, 3] <= stable_partial, 1, 0),
        ),
    )
