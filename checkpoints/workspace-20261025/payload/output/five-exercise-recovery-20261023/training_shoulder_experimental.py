"""분할 감사 후 독립 시각 등급으로 기존 관측 묶음의 AI 후보만 재학습한다.

최종 시험은 예약 메타데이터만 검사하며 특징·등급을 읽지 않는다.
명시 Rule 경계는 이 모듈에서 바꾸지 않고 회귀 목표로도 사용하지 않는다.
"""

import collections
import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from app.scoring.partition import audit_partitions


def refit_heads(
    template: dict,
    records: Sequence[dict],
    *,
    reserved_records: Sequence[dict] = (),
    historical_people: Sequence[str] = (),
    historical_sources: Sequence[str] = (),
    historical_recordings: Sequence[str] = (),
    historical_duplicate_groups: Sequence[str] = (),
) -> tuple[dict, dict]:
    """학습 연결 그룹 내부 CV로 AI 후보를 골라 별도 번들을 반환한다.

    Args:
        template: 한 운동의 기존 항목·구도별 모델. 입력은 변경하지 않는다.
        records: partition 계약과 levels·items·label_origin을 가진 개발 행.
            levels는 독립 시각 초안 0·1·2 또는 None이다. 각 item은 item·bank·
            eligible·feature 필드가 필요하다. label_origin은 direct_visual_draft
            또는 expert_review다. 최종 시험 역할 행은 받을 수 없다.
        reserved_records: 최종 시험의 지문·참여자 예약 행. 등급·특징은 읽지 않는다.
        historical_people: 과거 개발에 노출된 참여자 ID.
        historical_sources: 과거 개발에 노출된 원본 SHA-256.
        historical_recordings: 과거 개발에 노출된 촬영 묶음 ID.
        historical_duplicate_groups: 과거 개발에 노출된 파생·재인코딩 묶음 ID.

    Returns:
        tuple: 기존 Rule 경계를 보존한 모델 사본과 항목별 선택 진단.
        회귀 출력은 0~2 등급이다. CV 오차 단위도 등급이며 후보 선택용이므로
        독립 성능 추정이 아니다. 평균·중앙값 ExtraTrees와 해당 지지 항목의
        단조 제약 회귀를 학습 그룹 CV로 비교한다. 오류·정상 각각 두 연결 그룹이 없으면 해당
        항목은 기존 모델을 유지한다. 새 구도 묶음을 임의 생성하지 않는다.

    Raises:
        ValueError: 역할 누출, 미확인 라벨 출처, 등급·특징 형상·유한성 오류.
    """
    if not records:
        raise ValueError("Development records cannot be empty")
    if len({row.get("exercise") for row in records}) != 1:
        raise ValueError("One exercise per scoring model required")
    if any(row.get("split") not in ("train", "validation") for row in records):
        raise ValueError("Final test records cannot enter fitting or selection")
    if any(row.get("split") != "final_test" for row in reserved_records):
        raise ValueError("Reserved records must be final_test metadata")
    audit = audit_partitions(
        [*records, *reserved_records],
        historical_people=historical_people,
        historical_sources=historical_sources,
        historical_recordings=historical_recordings,
        historical_duplicate_groups=historical_duplicate_groups,
    )
    for row in records:
        if row.get("label_origin") not in (
            "direct_visual_draft",
            "expert_review",
        ):
            raise ValueError(
                "Independent direct visual criterion labels required"
            )
        if len(row.get("levels", [])) != 4 or len(row.get("items", [])) != 4:
            raise ValueError(
                "Exactly four criterion labels and observations required"
            )
        if any(
            type(g) is not int or g not in (0, 1, 2)
            for g in row["levels"]
            if g is not None
        ):
            raise ValueError(
                "Criterion grades must be integer zero to two or None"
            )
    groups = {
        identifier: index
        for index, group in enumerate(audit["connected_groups"])
        for identifier in group["ids"]
    }
    result = copy.deepcopy(template)
    reports = []
    for key, head in result.items():
        index = (
            int(key)
            if isinstance(key, int)
            else next(
                i
                for i, item in enumerate(records[0]["items"])
                if item["item"] == key[1]
            )
        )
        bucket = [
            row
            for row in records
            if row["split"] == "train"
            and row["levels"][index] is not None
            and row["items"][index]["eligible"]
            and (isinstance(key, int) or row["items"][index]["bank"] == key[2])
        ]
        normal = {groups[r["id"]] for r in bucket if r["levels"][index] == 2}
        errors = {groups[r["id"]] for r in bucket if r["levels"][index] < 2}
        if min(len(normal), len(errors)) < 2:
            reports.append(
                dict(
                    head=str(key),
                    retained=True,
                    reason="insufficient_normal_error_groups",
                )
            )
            continue
        x = np.asarray(
            [r["items"][index]["feature"] for r in bucket], dtype=float
        )
        y = np.asarray([r["levels"][index] for r in bucket], dtype=float)
        group_ids = np.asarray([groups[r["id"]] for r in bucket])
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError(
                "Finite observation features of one fixed dimension required"
            )

        def weights(indices: np.ndarray) -> np.ndarray:
            counts = collections.Counter((y[i], group_ids[i]) for i in indices)
            per_grade = collections.Counter(grade for grade, _ in counts)
            return np.asarray(
                [
                    1 / (per_grade[y[i]] * counts[y[i], group_ids[i]])
                    for i in indices
                ]
            )

        constraints = None
        exercise = records[0]["exercise"]
        if exercise == "squats" and index == 2 and x.shape[1] == 13:
            # 무릎 안쪽 이탈·뒤꿈치 상승이 커질수록 점수가 증가하지 않게 한다.
            constraints = [-1] * 7 + [0, 0, -1, -1, 0, -1]
        elif exercise == "plank" and index == 2 and x.shape[1] in (22, 29):
            # 무릎 신전과 지지선 위 골반·무릎 높이는 지지 항목만의 대리값이다.
            constraints = [1] * 21 + [0] + [1] * (x.shape[1] - 22)

        def estimator(name: str) -> Any:
            if name == "monotone_hist":
                return HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=80,
                    max_leaf_nodes=5,
                    min_samples_leaf=3,
                    l2_regularization=1,
                    monotonic_cst=constraints,
                    random_state=261023,
                )
            median = name == "median_forest"
            return ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2 if median else int(name[-1]),
                criterion="absolute_error" if median else "squared_error",
                random_state=261023,
                n_jobs=1,
            )

        indices = np.arange(len(y))
        losses = {}
        names = ["forest2", "forest4", "forest8", "median_forest"]
        if constraints is not None:
            names.append("monotone_hist")
        for name in names:
            predicted = np.full(len(y), np.nan)
            for fitting, held in GroupKFold(min(3, len(set(group_ids)))).split(
                x, groups=group_ids
            ):
                model = estimator(name)
                model.fit(
                    x[fitting], y[fitting], sample_weight=weights(fitting)
                )
                predicted[held] = model.predict(x[held])
            losses[name] = float(
                np.average(abs(predicted - y), weights=weights(indices))
            )
        selected = min(
            names, key=lambda name: (losses[name], names.index(name))
        )
        model = estimator(selected)
        model.fit(x, y, sample_weight=weights(indices))
        head.update(
            model=model,
            records=len(bucket),
            people=len(set(group_ids)),
            grade_counts={
                str(int(k)): v for k, v in collections.Counter(y).items()
            },
            error_people=len(errors),
            single_grade_model=False,
            training_only_model_selection=dict(
                kind="group_balanced_regressor",
                selected_estimator=selected,
                group_cv_selection_MAE=losses,
                independent_performance_estimate=False,
            ),
        )
        reports.append(
            dict(
                head=str(key),
                records=len(bucket),
                connected_groups=len(set(group_ids)),
                selected_estimator=selected,
                selection_MAE=losses,
                retained=False,
            )
        )
    return result, dict(
        heads=reports,
        validation_used_for_fitting=False,
        final_test_used=False,
        rule_boundaries_unchanged=True,
        selection_cv_is_not_final_performance=True,
    )
