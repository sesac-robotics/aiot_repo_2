"""직접 초안에서 기하-등급 관계를 학습하는 단조 회귀 비교 실험이다.

라벨 수가 적을 때 수백 특징을 쓰는 나무 모델의 과적합을 진단한다.
네 관계의 증가·감소 방향만 사전에 정하고 수치 경계와 등급은 직접
초안의 학습 폴드에서 학습한다. 규칙 등급을 학습 목표로 사용하지 않는다.
"""

import collections

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

import audit
import benchmark
import geometry
import rule_score
from followup_learning import ITEMS, WEIGHTS, total_metrics
from plank_partial_item_trial import load_rows
from cpu_analysis_guard import check_resources


def measurements(sequence: np.ndarray) -> np.ndarray | None:
    """공통 규칙과 같은 측면의 원시 기하 통계 4개만 반환한다.

    Args:
        sequence: N×33×2 픽셀 좌표, 미관측은 NaN.

    Returns:
        np.ndarray | None: 몸통 선 p10 각도(도), 팔꿈치 오프셋 p90
        몸통 비율, 무릎 p10 각도(도), 무릎 p95-p5 변화폭(도).
        규칙 경계·등급은 읽지 않는다. 필수 통계가 없으면 None.

    Raises:
        ValueError: 좌표 형상 오류.
    """
    values = geometry.channels(sequence)
    side = rule_score._side(values)
    line = rule_score._stat(values[side + "_line_deg"], 10)
    elbow = rule_score._stat(values[side + "_elbow_offset"], 90)
    knee = values[side + "_knee_deg"]
    finite = knee[np.isfinite(knee)]
    if len(finite) / len(knee) < 0.8 or line is None or elbow is None:
        return None
    return np.asarray(
        [
            line,
            elbow,
            np.percentile(finite, 10),
            np.percentile(finite, 95) - np.percentile(finite, 5),
        ]
    )


def fit(rows: list[dict]) -> list[IsotonicRegression]:
    """학습 초안만 사용해 네 기하-등급 함수를 각각 학습한다.

    Args:
        rows: x 4통계, grades 네 직접 등급, person이 있는 학습 목록.

    Returns:
        list[IsotonicRegression]: 증가·감소 방향이 고정된 네 회귀기.
        한 등급만 있는 머리는 상수 기준선이며 오류 판별 학습 완료가 아니다.

    Raises:
        ValueError: 빈 자료·특징·직접 등급 오류.
    """
    if not rows:
        raise ValueError("Nonempty direct grade cohort required")
    result = []
    for item, increasing in enumerate([True, False, True, False]):
        counts = collections.Counter(
            (r["grades"][item], r["person"]) for r in rows
        )
        groups = {
            g: len({p for k, p in counts if k == g})
            for g in {r["grades"][item] for r in rows}
        }
        weights = [
            1
            / (
                groups[r["grades"][item]]
                * counts[(r["grades"][item], r["person"])]
            )
            for r in rows
        ]
        model = IsotonicRegression(
            y_min=0, y_max=2, increasing=increasing, out_of_bounds="clip"
        )
        model.fit(
            [r["x"][item] for r in rows],
            [r["grades"][item] for r in rows],
            sample_weight=weights,
        )
        result.append(model)
    return result


def predict(
    models: list, x: np.ndarray
) -> tuple[float, list[float], list[bool]]:
    """학습한 네 등급을 합산하고 학습 통계 범위를 벗어난 항목을 표시한다.

    Args:
        models: 항목 순서의 네 학습 회귀기.
        x: 같은 순서의 네 관측 기하 통계.

    Returns:
        tuple: 0~100 총점, 연속 0~2 등급 네 개, 학습 범위 밖 여부 네 개.
        범위 밖은 단조 관계의 끝값 연장이다. 새 실측 정확도 보증이 아니다.
    """
    levels = [float(m.predict([v])[0]) for m, v in zip(models, x, strict=True)]
    outside = [
        bool(v < m.X_min_ or v > m.X_max_)
        for m, v in zip(models, x, strict=True)
    ]
    return float(np.dot(levels, WEIGHTS) / 2), levels, outside


def run() -> dict:
    """같은 12구간 참여자 제외 조건에서 나무 모델·단조 AI·규칙을 비교한다.

    Returns:
        dict: 학습된 매핑, 상수 머리, 범위 밖 입력과 점수 오차.
        앞선 개발 실패를 보고 설계한 후향 실험이며 새 독립 최종 시험이 아니다.

    Raises:
        OSError: 직접 초안·관절·모델 저장 실패.
        ValueError: 이전 비교와 대상 집합 불일치.
        RuntimeError: 자원 보호 실패.
    """
    cohort = []
    validation = []
    for row in load_rows(True):
        if row["split"] != "train" or row["partial"]:
            continue
        x = measurements(row["seq"])
        if x is not None:
            cohort.append(
                dict(row, x=x, score=float(np.dot(row["grades"], WEIGHTS) / 2))
            )
    for row in audit.read_json(audit.OUT / "followup_plank_direct.json")[
        "reviews"
    ]:
        seq, _, holds = benchmark.load_input(dict(row, exercise="plank"))
        x = measurements(seq)
        if holds or x is None:
            continue
        entry = dict(
            id=row["id"],
            person=row["person"],
            source=row["video_sha256"],
            seq=seq,
            x=x,
            grades=row["draft_levels"],
            score=row["total_score"],
        )
        (cohort if row["split"] == "train" else validation).append(entry)
    tree = audit.read_json(audit.OUT / "followup_plank_learning_results.json")
    previous = {r["id"]: r for r in tree["LOPO_predictions"]}
    if {r["id"] for r in cohort} != set(previous):
        raise ValueError("Monotone and tree comparison cohorts differ")
    predictions = []
    for person in sorted({r["person"] for r in cohort}):
        check_resources()
        testing = [r for r in cohort if r["person"] == person]
        sources = {r["source"] for r in testing}
        models = fit(
            [
                r
                for r in cohort
                if r["person"] != person and r["source"] not in sources
            ]
        )
        for row in testing:
            score, levels, outside = predict(models, row["x"])
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target_score=row["score"],
                    before_ai=previous[row["id"]]["after_ai"],
                    after_ai=score,
                    rule_score=previous[row["id"]]["rule_score"],
                    levels=levels,
                    outside_training_feature_range=outside,
                )
            )
    models = fit(cohort)
    folder = audit.OUT / "followup_plank_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        dict(
            models=models,
            train_ids=[r["id"] for r in cohort],
            input="four_raw_geometry_statistics",
            expert_validated=False,
        ),
        folder / "monotone.joblib",
    )
    new = []
    for row in validation:
        score, levels, outside = predict(models, row["x"])
        previous_row = next(
            r for r in tree["new_validation"] if r["id"] == row["id"]
        )
        new.append(
            dict(
                id=row["id"],
                target_score=row["score"],
                before_ai=previous_row["after_ai"],
                after_ai=score,
                rule_score=previous_row["rule_score"],
                levels=levels,
                outside_training_feature_range=outside,
            )
        )
    report = dict(
        LOPO_predictions=predictions,
        metrics=total_metrics(predictions),
        before_ai_meaning="augmented_ExtraTrees",
        after_ai_meaning="learned_monotone_regression",
        new_validation=new,
        new_validation_metrics=total_metrics(new),
        train_records=len(cohort),
        training_people=len({r["person"] for r in cohort}),
        item_grade_counts=[
            dict(collections.Counter(r["grades"][i] for r in cohort))
            for i in range(4)
        ],
        mappings=[
            dict(
                item=ITEMS[i],
                x=m.X_thresholds_.tolist(),
                y=m.y_thresholds_.tolist(),
            )
            for i, m in enumerate(models)
        ],
        rule_targets_used=False,
        posthoc_development_trial=True,
        expert_validated=False,
        independent_final_test=False,
        common_model_replaced=False,
        model_sha256=audit.digest(folder / "monotone.joblib"),
    )
    audit.save_json(audit.OUT / "plank_monotone_trial_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["metrics"])
