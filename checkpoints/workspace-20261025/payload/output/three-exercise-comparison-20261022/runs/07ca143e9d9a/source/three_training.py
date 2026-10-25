"""관측 항목별 AI와 학습 초안으로 경계를 보정한 명시적 규칙을 비교한다.

교차 검증에서는 참여자·원본 전체를 제외한다. 규칙 출력은 학습 정답이
아니며 경계 보정과 AI 학습 모두 직접 등급 초안만 사용한다.
"""

import collections

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import three_items

BOUNDARIES = {
    "trunk_line": ([155, 160, 165, 168, 172, 176], [140, 150, 155, 160]),
    "pushups_descent_range": (
        [80, 90, 95, 100, 110],
        [110, 120, 125, 135, 145],
    ),
    "return_extension": (
        [150, 155, 160, 165, 170, 175],
        [130, 140, 145, 150, 155],
    ),
    "support": ([0.08, 0.12, 0.15, 0.2, 0.25], [0, 0.03, 0.05, 0.08, 0.1]),
    "squats_descent_range": (
        [0, 0.1, 0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5, 0.65, 0.8],
    ),
    "knee_foot": ([0.15, 0.2, 0.25, 0.3, 0.4], [0.35, 0.45, 0.5, 0.6, 0.7]),
    "trunk_balance": ([5, 8, 10, 12], [12, 15, 18, 22]),
    "overhead_reach": ([135, 145, 150, 155, 165], [100, 115, 125, 135]),
    "lower_return": ([95, 105, 110, 120], [125, 135, 140, 150]),
    "bilateral_timing": ([0.08, 0.12, 0.15, 0.2], [0.25, 0.3, 0.35, 0.45]),
    "frontal_trunk_hold": ([5, 8, 10, 12], [12, 15, 18, 22]),
}
DEFAULTS = {
    "trunk_line": (168, 155),
    "pushups_descent_range": (95, 125),
    "return_extension": (160, 140),
    "support": (0.15, 0.05),
    "squats_descent_range": (0.1, 0.5),
    "knee_foot": (0.25, 0.5),
    "trunk_balance": (8, 15),
    "overhead_reach": (150, 125),
    "lower_return": (110, 140),
    "bilateral_timing": (0.15, 0.35),
    "frontal_trunk_hold": (8, 15),
}


def rule_grade(note: dict, boundaries: tuple[float, float]) -> int | None:
    """관측 통과 항목의 측정값과 명시 경계로 0·1·2 등급을 계산한다.

    Args:
        note: three_items.describe의 항목 기록. 각도는 도, 거리는 몸통 비율.
        boundaries: 만점·부분 수행의 순서인 두 임계값.

    Returns:
        int | None: 관측된 등급 또는 미판정. 무릎·발은 뒤꿈치 보조 조건도 적용.

    Raises:
        ValueError: 만점·부분 경계 순서가 측정 방향과 반대인 경우.
    """
    if not note["eligible"] or note["measurement"] is None:
        return None
    good, partial = boundaries
    high = note["higher_is_better"]
    if (high and good <= partial) or (not high and good >= partial):
        raise ValueError("Invalid ordered rule boundaries")
    value = note["measurement"]
    grade = (
        (2 if value >= good else 1 if value >= partial else 0)
        if high
        else (2 if value <= good else 1 if value <= partial else 0)
    )
    if note["item"] == "knee_foot":
        heel = note["feature"][-1]
        grade = min(grade, 2 if heel <= 0.15 else 1 if heel <= 0.3 else 0)
    return grade


def fit(rows: list[dict]) -> dict:
    """입력된 학습 폴드만으로 네 항목 AI와 규칙 경계를 각각 학습·보정한다.

    Args:
        rows: 같은 운동의 notes·levels·person·exercise·weights 레코드.

    Returns:
        dict: 항목 AI·규칙 경계·항목별 자료 수·평균 점수 기준선.
        규칙은 단일 기하 측정의 명시 조건이며 AI 회귀 결과와 독립이다.

    Raises:
        ValueError: 빈 학습 폴드 또는 운동 혼합.
    """
    if not rows or len({r["exercise"] for r in rows}) != 1:
        raise ValueError("Nonempty single-exercise training rows required")
    exercise = rows[0]["exercise"]
    heads, rules, counts = [], [], []
    for item in range(4):
        fitting = [
            r
            for r in rows
            if r["levels"][item] is not None and r["notes"][item]["eligible"]
        ]
        counts.append(
            dict(
                item=rows[0]["notes"][item]["item"],
                records=len(fitting),
                people=len({r["person"] for r in fitting}),
                grade_counts=dict(
                    collections.Counter(r["levels"][item] for r in fitting)
                ),
                error_people=len(
                    {r["person"] for r in fitting if r["levels"][item] < 2}
                ),
            )
        )
        if not fitting:
            heads.append(None)
            rules.append(None)
            continue
        group_counts = collections.Counter(
            (r["levels"][item], r["person"]) for r in fitting
        )
        grade_people = collections.Counter(g for g, _ in group_counts)
        weights = [
            1
            / (
                grade_people[r["levels"][item]]
                * group_counts[(r["levels"][item], r["person"])]
            )
            for r in fitting
        ]
        model = ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=221022, n_jobs=1
        )
        model.fit(
            [r["notes"][item]["feature"] for r in fitting],
            [r["levels"][item] for r in fitting],
            sample_weight=weights,
        )
        heads.append(model)
        note = fitting[0]["notes"][item]
        name = note["item"]
        key = exercise + "_" + name if name == "descent_range" else name
        if (
            counts[-1]["error_people"] < 2
            or len(counts[-1]["grade_counts"]) < 2
        ):
            default = (
                (165, 145)
                if exercise == "squats" and name == "return_extension"
                else DEFAULTS[key]
            )
            rules.append(default)
            counts[-1]["rule_calibrated"] = False
            continue
        counts[-1]["rule_calibrated"] = True
        candidates = [
            (g, p)
            for g in BOUNDARIES[key][0]
            for p in BOUNDARIES[key][1]
            if (g > p if note["higher_is_better"] else g < p)
        ]
        rules.append(
            min(
                candidates,
                key=lambda bounds: float(
                    np.average(
                        [
                            abs(
                                rule_grade(r["notes"][item], bounds)
                                - r["levels"][item]
                            )
                            for r in fitting
                        ],
                        weights=weights,
                    )
                ),
            )
        )
    scores = [three_items.total(r["levels"], r["weights"]) for r in rows]
    scores = [s for s in scores if s is not None]
    return dict(
        heads=heads,
        rules=rules,
        item_training=counts,
        training_mean_score=float(np.mean(scores)) if scores else None,
        exercise=exercise,
        rule_targets_used=False,
    )


def predict(model: dict, row: dict) -> dict:
    """같은 관측 항목의 AI·규칙을 계산하고 미관측 총점을 보류한다.

    Args:
        model: fit이 반환한 같은 운동의 학습 완료 사전.
        row: notes·weights·exercise가 있는 관절 좌표 기반 레코드.

    Returns:
        dict: 항목별 연속 AI 등급·규칙 등급, 100점 총점 또는 None.

    Raises:
        ValueError: 운동 또는 등급·배점 계약 불일치.
    """
    if model["exercise"] != row["exercise"]:
        raise ValueError("Exercise-specific model mismatch")
    ai, rules = [], []
    for head, boundary, note in zip(
        model["heads"], model["rules"], row["notes"], strict=True
    ):
        ai.append(
            float(np.clip(head.predict([note["feature"]])[0], 0, 2))
            if head is not None and note["eligible"]
            else None
        )
        rules.append(
            rule_grade(note, boundary) if boundary is not None else None
        )
    return dict(
        ai_levels=ai,
        rule_levels=rules,
        ai_score=three_items.total(ai, row["weights"]),
        rule_score=three_items.total(rules, row["weights"]),
    )
