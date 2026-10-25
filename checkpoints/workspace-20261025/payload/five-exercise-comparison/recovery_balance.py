"""희소 중간 등급의 과도한 가중치를 학습 그룹 내부 검증으로 점검한다."""

import collections

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


def weights_for(rows: list[dict], item: int, floor: int) -> list[float]:
    """참여자·등급 중복을 보정하고 희소 등급의 가중치 상한을 적용한다.

    Args:
        rows: 한 구도·항목의 관측 가능한 학습 행.
        item: 0부터 시작하는 채점 항목 인덱스.
        floor: 등급별 참여자 수의 분모 하한. 1은 기존 완전 등급 균형이다.

    Returns:
        list[float]: 행별 양의 상대 가중치. 같은 참여자의 반복을 늘려도
        그 참여자·등급의 총 가중치는 늘어나지 않는다.

    Raises:
        ValueError: 하한이 1보다 작을 경우.
    """
    if floor < 1:
        raise ValueError("Positive class support floor required")
    counts = collections.Counter((r["levels"][item], r["person"]) for r in rows)
    people = collections.Counter(g for g, p in counts)
    return [
        1
        / (
            max(floor, people[r["levels"][item]])
            * counts[(r["levels"][item], r["person"])]
        )
        for r in rows
    ]


def select(rows: list[dict], item: int) -> tuple[object, dict]:
    """학습 참여자 내부 3폴드에서 회귀기·희소 등급 가중치 후보를 선택한다.

    Args:
        rows: 측면 푸시업 몸통 항목의 학습 자료만 포함한 목록.
        item: 몸통 항목 인덱스. 측정값은 관절 기하 각도(도)다.

    Returns:
        tuple: 선택된 회귀기와 후보별 참여자 평균 등급 MAE·가중치 하한.
        외부 개발 검증·최종 시험 목표나 규칙 출력은 사용하지 않는다.

    Raises:
        ValueError: 학습 그룹·특징·관절 측정값이 불충분한 경우.
    """
    groups = [r["person"] for r in rows]
    if len(set(groups)) < 3:
        raise ValueError("Three training participant groups required")

    def fit_model(fitting, kind, floor):
        weights = weights_for(fitting, item, floor)
        model = (
            IsotonicRegression(
                increasing=True, out_of_bounds="clip", y_min=0, y_max=2
            )
            if kind == "isotonic"
            else ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=221022,
                n_jobs=1,
            )
        )
        x = [
            (
                r["notes"][item]["measurement"]
                if kind == "isotonic"
                else r["notes"][item]["feature"]
            )
            for r in fitting
        ]
        model.fit(
            x, [r["levels"][item] for r in fitting], sample_weight=weights
        )
        return model

    losses = {}
    for kind in ["extra_trees", "isotonic"]:
        for floor in [1, 3, 6]:
            errors = collections.defaultdict(list)
            for fitting, held in GroupKFold(3).split(rows, groups=groups):
                model = fit_model([rows[int(i)] for i in fitting], kind, floor)
                for index in held:
                    row = rows[int(index)]
                    x = (
                        row["notes"][item]["measurement"]
                        if kind == "isotonic"
                        else row["notes"][item]["feature"]
                    )
                    errors[row["person"]].append(
                        abs(float(model.predict([x])[0]) - row["levels"][item])
                    )
            losses[(kind, floor)] = float(
                np.mean([np.mean(v) for v in errors.values()])
            )
    kind, floor = min(losses, key=lambda k: (losses[k], k[1], k[0]))
    model = fit_model(rows, kind, floor)
    return (
        dict(kind="isotonic", model=model) if kind == "isotonic" else model
    ), dict(
        kind=kind,
        class_support_floor=floor,
        inner_group_folds=3,
        inner_MAE_grade_units={
            f"{k[0]}_floor{k[1]}": v for k, v in losses.items()
        },
        outer_validation_targets_used=False,
    )
