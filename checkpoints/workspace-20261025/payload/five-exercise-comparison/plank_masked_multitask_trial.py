"""관측 가능한 직접 항목 라벨을 보존하여 플랭크 네 머리를 별도로 학습한다.

전신 총점이 없어도 직접 확인한 항목만 학습한다. 규칙 출력은 정답으로
사용하지 않으며 상지 사건 이진 라벨을 지속성 등급으로 변환하지 않는다.
모든 항목의 입력을 관측한 경우에만 100점 배점의 총점을 계산한다.
"""

import collections

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import audit
import benchmark
from cpu_analysis_guard import check_resources
from followup_learning import ITEMS, WEIGHTS, total_metrics
from plank_item_observation import assess, features
from plank_partial_item_trial import load_rows


def fit(rows: list[dict]) -> tuple[list, list[dict]]:
    """각 항목의 직접 등급·관측을 통과한 학습 폴드만으로 머리를 학습한다.

    Args:
        rows: seq(픽셀), grades(0·1·2 또는 None), duration(초), person 자료.

    Returns:
        tuple: 네 회귀 파이프라인 또는 None과 항목별 학습 구간·참여자·등급 수.
        라벨 None은 결측 특징 대체와 달리 학습 목표로 채우지 않는다.

    Raises:
        ValueError: 관절·시간·등급 계약 오류 또는 특징 순서 불일치.
    """
    models = []
    counts = []
    for item in range(4):
        fitting = []
        for row in rows:
            if row["grades"][item] is None:
                continue
            observation = assess(
                row["seq"],
                row["duration"],
                [g is not None for g in row["grades"]],
            )["items"][item]
            if not observation["eligible"]:
                continue
            x, names = features(row["seq"], item, observation["side"])
            fitting.append(
                dict(
                    id=row["id"],
                    x=x,
                    target=row["grades"][item],
                    person=row["person"],
                )
            )
        counts.append(
            dict(
                item=ITEMS[item],
                records=len(fitting),
                people=len({r["person"] for r in fitting}),
                grade_counts=dict(
                    collections.Counter(r["target"] for r in fitting)
                ),
                nonfull_grade_people=len(
                    {r["person"] for r in fitting if r["target"] != 2}
                ),
            )
        )
        if not fitting:
            models.append(None)
            continue
        pair_counts = collections.Counter(
            (r["target"], r["person"]) for r in fitting
        )
        groups = {
            g: len({p for k, p in pair_counts if k == g})
            for g in {r["target"] for r in fitting}
        }
        weights = [
            1 / (groups[r["target"]] * pair_counts[(r["target"], r["person"])])
            for r in fitting
        ]
        model = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=210211,
                n_jobs=1,
            ),
        )
        model.fit(
            [r["x"] for r in fitting],
            [r["target"] for r in fitting],
            extratreesregressor__sample_weight=weights,
        )
        models.append(model)
    return models, counts


def predict(
    models: list,
    sequence: np.ndarray,
    duration: float,
    *,
    permitted_items: list[bool] | None = None,
) -> dict:
    """두 방식에 같은 항목별 관측·측면 정책을 적용해 등급과 총점을 반환한다.

    Args:
        models: fit에서 반환한 네 항목 모델 또는 None 목록.
        sequence: N×33×2 픽셀 좌표, NaN은 미관측.
        duration: 평가 구간 초 단위 길이.
        permitted_items: 촬영 구도로 허용한 네 항목. 생략 시 모두 보류.

    Returns:
        dict: 연속 AI 등급·동일 관측 정책의 규칙 등급·총점 또는 None.
        한 항목이라도 미관측·모델 없음이면 AI 총점은 None이다.

    Raises:
        ValueError: 네 머리가 아니거나 관측 입력 계약 오류.
    """
    if len(models) != 4:
        raise ValueError("Four item heads required")
    observed = assess(sequence, duration, permitted_items or [False] * 4)
    levels = []
    for item, (model, note) in enumerate(
        zip(models, observed["items"], strict=True)
    ):
        if model is None or not note["eligible"]:
            levels.append(None)
            continue
        x, _ = features(sequence, item, note["side"])
        levels.append(float(np.clip(model.predict([x])[0], 0, 2)))
    return dict(
        ai_levels=levels,
        rule_levels=[r["rule_grade"] for r in observed["items"]],
        ai_total=(
            float(np.dot(levels, WEIGHTS) / 2)
            if all(g is not None for g in levels)
            else None
        ),
        rule_total=observed["total_score"],
        observation=observed,
    )


def run() -> dict:
    """같은 12개 총점 구간을 제외 참여자 단위로 비교하고 부분 라벨을 보강한다.

    Returns:
        dict: 전신 모델·부분 항목 모델·같은 항목 정책 규칙의 개발 비교.
        기존 새 검증 출처를 다시 평가하지 않으며 전문가·독립 최종 성능이 아니다.

    Raises:
        OSError: 라벨·캐시·모델·결과 접근 실패.
        ValueError: 원본·참여자 누출 또는 기존 총점 비교 집합 불일치.
        RuntimeError: CPU 분석 보호 조건 위반.
    """
    rows = [r for r in load_rows(True) if r["split"] == "train"]
    for draft in audit.read_json(audit.OUT / "followup_plank_direct.json")[
        "reviews"
    ]:
        if draft["split"] != "train":
            continue
        seq, _, holds = benchmark.load_input(dict(draft, exercise="plank"))
        if holds:
            continue
        segment = draft["segment"]
        rows.append(
            dict(
                id=draft["id"],
                seq=seq,
                grades=draft["draft_levels"],
                person=draft["person"],
                source=draft["video_sha256"],
                duration=(segment["end_frame"] - segment["start_frame"] + 1)
                / segment["fps"],
                partial=False,
            )
        )
    previous = audit.read_json(
        audit.OUT / "followup_plank_learning_results.json"
    )
    paired = {r["id"]: r for r in previous["LOPO_predictions"]}
    testing = [r for r in rows if r["id"] in paired]
    if {r["id"] for r in testing} != set(paired):
        raise ValueError("Same full-score cohort required")
    predictions = []
    fold_counts = []
    for person in sorted({r["person"] for r in testing}):
        check_resources()
        held = [r for r in testing if r["person"] == person]
        sources = {r["source"] for r in held}
        fitting = [
            r
            for r in rows
            if r["person"] != person and r["source"] not in sources
        ]
        if {r["person"] for r in fitting} & {person} or {
            r["source"] for r in fitting
        } & sources:
            raise ValueError("Person/source leakage in item fitting")
        models, counts = fit(fitting)
        fold_counts.append(dict(held_person=person, item_training=counts))
        for row in held:
            result = predict(
                models, row["seq"], row["duration"], permitted_items=[True] * 4
            )
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target_score=paired[row["id"]]["target_score"],
                    before_ai=paired[row["id"]]["after_ai"],
                    after_ai=result["ai_total"],
                    rule_score=result["rule_total"],
                    original_common_rule_score=paired[row["id"]]["rule_score"],
                    **result,
                )
            )
    models, counts = fit(rows)
    folder = audit.OUT / "followup_plank_model"
    path = folder / "masked_item_heads.joblib"
    joblib.dump(
        dict(
            models=models,
            weights=WEIGHTS.tolist(),
            train_ids=[r["id"] for r in rows],
            item_training=counts,
            expert_validated=False,
        ),
        path,
    )
    report = dict(
        metrics=total_metrics(predictions),
        predictions=predictions,
        item_training=counts,
        fold_counts=fold_counts,
        source_training_segments=len(rows),
        full_score_paired_records=len(testing),
        rule_policy="same_per_item_joint_observation_and_side_selection_as_AI",
        before_ai_meaning="augmented_full_feature_ExtraTrees",
        after_ai_meaning="masked_item_heads_with_direct_partial_labels",
        binary_events_converted_to_score_labels=False,
        rule_targets_used=False,
        observed_missing_targets_filled=False,
        partial_totals_renormalized=False,
        validation_evaluation_repeated=False,
        posthoc_development_trial=True,
        expert_validated=False,
        independent_final_test=False,
        common_model_replaced=False,
        model_sha256=audit.digest(path),
    )
    audit.save_json(
        audit.OUT / "plank_masked_multitask_trial_results.json", report
    )
    return report


if __name__ == "__main__":
    print(run()["metrics"])
