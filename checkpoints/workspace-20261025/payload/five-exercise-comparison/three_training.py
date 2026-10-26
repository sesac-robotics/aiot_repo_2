"""관측 항목별 AI와 학습 초안으로 경계를 보정한 명시적 규칙을 비교한다.

교차 검증에서는 참여자·원본 전체를 제외한다. 규칙 출력은 학습 정답이
아니며 경계 보정과 AI 학습 모두 직접 등급 초안만 사용한다.
"""

import collections

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

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
    "squats_descent_range": (0.25, 0.6),
    "knee_foot": (0.25, 0.5, 0.025, 0.08),
    "trunk_balance": (8, 15),
    "overhead_reach": (150, 125),
    "lower_return": (110, 140),
    "bilateral_timing": (0.15, 0.35),
    "frontal_trunk_hold": (8, 15),
}


def select_trunk_head(
    rows: list[dict], item: int, forest: object, weights: list[float]
) -> tuple[object, dict]:
    """학습 참여자 내부 분할에서 몸통 정렬의 단조 회귀 후보를 비교한다.

    Args:
        rows: 관측과 직접 등급이 모두 있는 현재 학습 폴드의 몸통 항목 자료.
        item: 네 항목 중 몸통 정렬 인덱스.
        forest: 현재 학습 폴드로 학습된 ExtraTrees 모델.
        weights: 현재 학습 폴드의 참여자·등급 균형 가중치.

    Returns:
        tuple[object, dict]: 선택된 AI 모델과 내부 참여자 분할 오차.
        단조 회귀도 직접 등급을 학습하며 규칙 계산값은 사용하지 않는다.

    Raises:
        ValueError: 관측 자료·그룹 또는 회귀 입력이 잘못된 경우.
    """
    if (
        len({r["person"] for r in rows}) < 6
        or len({r["levels"][item] for r in rows}) < 2
    ):
        return forest, dict(
            kind="extra_trees",
            selection="insufficient_groups_for_inner_selection",
        )
    groups = [r["person"] for r in rows]
    errors = {
        "extra_trees": collections.defaultdict(list),
        "isotonic": collections.defaultdict(list),
    }
    for train_indices, held_indices in GroupKFold(3).split(rows, groups=groups):
        fitting = [rows[int(i)] for i in train_indices]
        counts = collections.Counter(
            (r["levels"][item], r["person"]) for r in fitting
        )
        grade_people = collections.Counter(g for g, _ in counts)
        fitting_weights = [
            1
            / (
                grade_people[r["levels"][item]]
                * counts[(r["levels"][item], r["person"])]
            )
            for r in fitting
        ]
        inner = ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=221022, n_jobs=1
        )
        inner.fit(
            [r["notes"][item]["feature"] for r in fitting],
            [r["levels"][item] for r in fitting],
            sample_weight=fitting_weights,
        )
        curve = IsotonicRegression(
            increasing=True, out_of_bounds="clip", y_min=0, y_max=2
        )
        curve.fit(
            [r["notes"][item]["measurement"] for r in fitting],
            [r["levels"][item] for r in fitting],
            sample_weight=fitting_weights,
        )
        for index in held_indices:
            row = rows[int(index)]
            predictions = dict(
                extra_trees=float(
                    inner.predict([row["notes"][item]["feature"]])[0]
                ),
                isotonic=float(
                    curve.predict([row["notes"][item]["measurement"]])[0]
                ),
            )
            for kind, value in predictions.items():
                errors[kind][row["person"]].append(
                    abs(value - row["levels"][item])
                )
    losses = {
        kind: float(np.mean([np.mean(values) for values in people.values()]))
        for kind, people in errors.items()
    }
    kind = min(losses, key=losses.get)
    if kind == "isotonic":
        curve = IsotonicRegression(
            increasing=True, out_of_bounds="clip", y_min=0, y_max=2
        )
        curve.fit(
            [r["notes"][item]["measurement"] for r in rows],
            [r["levels"][item] for r in rows],
            sample_weight=weights,
        )
        return dict(kind="isotonic", model=curve), dict(
            kind=kind,
            inner_group_folds=3,
            inner_MAE_grade_units=losses,
            outer_validation_targets_used=False,
        )
    return forest, dict(
        kind=kind,
        inner_group_folds=3,
        inner_MAE_grade_units=losses,
        outer_validation_targets_used=False,
    )


def rule_grade(note: dict, boundaries: tuple[float, ...]) -> int | None:
    """관측 통과 항목의 측정값과 명시 경계로 0·1·2 등급을 계산한다.

    Args:
        note: three_items.describe의 항목 기록. 각도는 도, 거리는 몸통 비율.
        boundaries: 만점·부분 경계. 무릎·발은 뒤꿈치 경계 두 개도 뒤에 둔다.
            뒤꿈치 들림 변화는 몸통 길이 대비 비율이다.

    Returns:
        int | None: 관측된 등급 또는 미판정. 무릎·발은 뒤꿈치 보조 조건도 적용.

    Raises:
        ValueError: 만점·부분 경계 순서가 측정 방향과 반대인 경우.
    """
    if not note["eligible"] or note["measurement"] is None:
        return None
    good, partial = boundaries[:2]
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
        heel_good, heel_partial = (
            boundaries[2:] if len(boundaries) == 4 else (0.025, 0.08)
        )
        if heel_good >= heel_partial:
            raise ValueError("Invalid ordered heel boundaries")
        grade = min(
            grade, 2 if heel <= heel_good else 1 if heel <= heel_partial else 0
        )
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
        if (
            exercise == "pushups"
            and rows[0]["notes"][item]["item"] == "trunk_line"
        ):
            model, selection = select_trunk_head(fitting, item, model, weights)
            counts[-1]["AI_head_selection"] = selection
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
        if key == "knee_foot":
            candidates = [
                (*pair, heel_good, heel_partial)
                for pair in candidates
                for heel_good in (0.01, 0.025, 0.04, 0.06)
                for heel_partial in (0.04, 0.06, 0.08, 0.12, 0.16)
                if heel_good < heel_partial
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
        feature_version=three_items.VERSION,
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
    if (
        model["exercise"] != row["exercise"]
        or model.get("feature_version") != three_items.VERSION
    ):
        raise ValueError("Exercise-specific model mismatch")
    ai, rules = [], []
    for head, boundary, note in zip(
        model["heads"], model["rules"], row["notes"], strict=True
    ):
        value = None
        if head is not None and note["eligible"]:
            value = (
                head["model"].predict([note["measurement"]])[0]
                if isinstance(head, dict) and head["kind"] == "isotonic"
                else head.predict([note["feature"]])[0]
            )
        ai.append(float(np.clip(value, 0, 2)) if value is not None else None)
        rules.append(
            rule_grade(note, boundary) if boundary is not None else None
        )
    return dict(
        ai_levels=ai,
        rule_levels=rules,
        ai_score=three_items.total(ai, row["weights"]),
        rule_score=three_items.total(rules, row["weights"]),
    )
