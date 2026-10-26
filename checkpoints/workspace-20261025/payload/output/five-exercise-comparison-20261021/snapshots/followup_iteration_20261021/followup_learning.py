"""플랭크 오류 원본·상지 대조 사례를 추가해 별도 모델을 재학습한다.

기존 시험 출처·공통 모델·라벨은 교체하지 않는다. 같은 제외 참여자와
원본에서 보강 전·후 AI를 비교하며 제공자 문자열이나 규칙 점수를 학습하지 않는다.
"""

import collections

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import audit
import benchmark
import rule_score
from plank_balance_trial import fitting_weights
from plank_partial_item_trial import load_rows
from resources import check_resources
from upper_support_event_model import event_features, fit as fit_event

WEIGHTS = np.asarray([40, 25, 20, 15])
ITEMS = [
    "trunk_line",
    "elbow_sagittal_position",
    "leg_and_body_support",
    "support_continuity",
]


def fit_total(rows: list[dict]):
    """학습 폴드만으로 결측 대체·점수 클래스·참여자 가중치를 학습한다.

    Args:
        rows: x, grades(0·1·2 네 개), score(0~100), person이 있는 자료.

    Returns:
        sklearn.Pipeline: 기존 balanced_leaf2와 같은 설정의 등급 회귀기.

    Raises:
        ValueError: 빈 자료 또는 특징·목표 형상 오류.
    """
    if not rows:
        raise ValueError("Empty full plank training fold")
    model = make_pipeline(
        SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ),
        ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=21021, n_jobs=1
        ),
    )
    weights = fitting_weights(
        [dict(score=r["score"]) for r in rows], [r["person"] for r in rows]
    )
    model.fit(
        [r["x"] for r in rows],
        [r["grades"] for r in rows],
        extratreesregressor__sample_weight=weights,
    )
    return model


def total_metrics(rows: list[dict]) -> dict:
    """동일 표본에서 보강 전·후 AI와 고정 규칙의 점수 오차를 집계한다.

    Args:
        rows: target_score, before_ai, after_ai, rule_score가 있는 예측 목록.

    Returns:
        dict: 평가 수와 0~100점 단위 MAE. 미평가 값은 제외하고 수를 보존한다.
    """
    result = {}
    for name in ["before_ai", "after_ai", "rule_score"]:
        pairs = [r for r in rows if r.get(name) is not None]
        result[name] = dict(
            records=len(pairs),
            MAE_points=(
                float(
                    np.mean([abs(r[name] - r["target_score"]) for r in pairs])
                )
                if pairs
                else None
            ),
        )
    return result


def run_total() -> dict:
    """새 정상·높은 골반 초안으로 총점 모델을 별도 재학습·검증한다.

    Returns:
        dict: 실제 관측 통과 수, 같은 사람·원본 제외 예측과 새 정상 개발
        검증. 정상 1원본의 결과를 오류 정확도나 전체 정확도로 표현하지 않는다.

    Raises:
        OSError: 원본·관절·모델 접근 실패.
        ValueError: 특징 순서·원본 또는 참여자 누출.
        RuntimeError: 자원 보호 실패.
    """
    old = []
    for row in load_rows(True):
        if row["split"] != "train" or row["partial"]:
            continue
        x, names = benchmark.input_features(row["seq"], None)
        old.append(
            dict(row, x=x, score=float(np.dot(row["grades"], WEIGHTS) / 2))
        )
    added = []
    validation = []
    excluded = []
    for draft in audit.read_json(audit.OUT / "followup_plank_direct.json")[
        "reviews"
    ]:
        row = dict(draft, exercise="plank")
        seq, front, holds = benchmark.load_input(row)
        if holds:
            excluded.append(dict(id=row["id"], holds=holds))
            continue
        x, new_names = benchmark.input_features(seq, front)
        if names != new_names:
            raise ValueError("Total feature order differs")
        entry = dict(
            id=row["id"],
            seq=seq,
            x=x,
            grades=row["draft_levels"],
            score=row["total_score"],
            person=row["person"],
            source=row["video_sha256"],
            split=row["split"],
        )
        (added if row["split"] == "train" else validation).append(entry)
    cohort = old + added
    predictions = []
    for person in sorted({r["person"] for r in cohort}):
        check_resources()
        held = [r for r in cohort if r["person"] == person]
        sources = {r["source"] for r in held}
        groups = [
            r
            for r in cohort
            if r["person"] != person and r["source"] not in sources
        ]
        before = fit_total(
            [
                r
                for r in old
                if r["person"] != person and r["source"] not in sources
            ]
        )
        after = fit_total(groups)
        for row in held:
            rule = rule_score.evaluate(
                "plank", row["seq"], None, ITEMS, WEIGHTS.tolist()
            )
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target_score=row["score"],
                    before_ai=float(
                        np.dot(
                            np.clip(before.predict([row["x"]])[0], 0, 2),
                            WEIGHTS,
                        )
                        / 2
                    ),
                    after_ai=float(
                        np.dot(
                            np.clip(after.predict([row["x"]])[0], 0, 2), WEIGHTS
                        )
                        / 2
                    ),
                    rule_score=rule["score"],
                    rule_details=rule,
                )
            )
    before, after = fit_total(old), fit_total(cohort)
    folder = audit.OUT / "followup_plank_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        dict(
            model=after,
            names=names,
            weights=WEIGHTS.tolist(),
            train_ids=[r["id"] for r in cohort],
            train_people=sorted({r["person"] for r in cohort}),
            expert_validated=False,
        ),
        folder / "model.joblib",
    )
    held_results = []
    for row in validation:
        if row["person"] in {r["person"] for r in cohort} or row["source"] in {
            r["source"] for r in cohort
        }:
            raise ValueError("Validation source or person overlaps fitting")
        rule = rule_score.evaluate(
            "plank", row["seq"], None, ITEMS, WEIGHTS.tolist()
        )
        held_results.append(
            dict(
                id=row["id"],
                target_score=row["score"],
                before_ai=float(
                    np.dot(
                        np.clip(before.predict([row["x"]])[0], 0, 2), WEIGHTS
                    )
                    / 2
                ),
                after_ai=float(
                    np.dot(np.clip(after.predict([row["x"]])[0], 0, 2), WEIGHTS)
                    / 2
                ),
                rule_score=rule["score"],
                rule_details=rule,
            )
        )
    report = dict(
        old_training_records=len(old),
        new_training_records=len(added),
        total_training_records=len(cohort),
        training_person_groups=len({r["person"] for r in cohort}),
        train_total_score_counts=dict(
            collections.Counter(r["score"] for r in cohort)
        ),
        excluded=excluded,
        LOPO_predictions=predictions,
        LOPO_metrics=total_metrics(predictions),
        new_validation=held_results,
        new_validation_metrics=total_metrics(held_results),
        new_validation_has_error_examples=any(
            r["target_score"] < 100 for r in held_results
        ),
        model_sha256=audit.digest(folder / "model.joblib"),
        common_model_replaced=False,
        all_five_final_test=False,
        expert_validated=False,
        autonomous_prospective_sources_used_for_fitting=False,
        source_null_fallback_fixed=True,
    )
    audit.save_json(audit.OUT / "followup_plank_learning_results.json", report)
    return report


def event_metrics(rows: list[dict]) -> dict:
    """상지 사건의 같은 표본에 대한 혼동 수를 반환한다.

    Args:
        rows: target 0·1과 before_event, after_event, rule_event가 있는 목록.

    Returns:
        dict: 세 방법의 TP·FN·FP·TN. 총점 오차와 혼합하지 않는다.
    """
    return {
        method: dict(
            TP=sum(r[method] and r["target"] == 1 for r in rows),
            FN=sum(not r[method] and r["target"] == 1 for r in rows),
            FP=sum(r[method] and r["target"] == 0 for r in rows),
            TN=sum(not r[method] and r["target"] == 0 for r in rows),
        )
        for method in ["before_event", "after_event", "rule_event"]
    }


def run_events() -> dict:
    """추가 상지 사건과 머리·골반 움직임 대조로 사건 모델을 재학습한다.

    Returns:
        dict: 같은 제외 참여자별 보강 전·후 확률·고정 0.25 경계 혼동 수.
        경계는 이전 개발 실험에서 정한 값으로 이번 예측 후 조정하지 않는다.

    Raises:
        OSError: 초안·관절·모델 접근 실패.
        ValueError: 사건 특징 누락·원본 또는 참여자 분할 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    base_rows = {r["id"]: r for r in load_rows(True) if r["split"] == "train"}
    entries = audit.read_json(audit.OUT / "qevd_upper_event_extraction.json")[
        "records"
    ]
    direct = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "qevd_upper_event_direct.json")[
            "reviews"
        ]
    }
    for entry in entries:
        draft = direct[entry["id"]]
        with np.load(entry["extraction"]["cache"]) as cache:
            seq = cache["image_xy"].copy()
        base_rows[entry["id"]] = dict(
            seq=seq, person=draft["person"], source=draft["video_sha256"]
        )
    old = []
    targets = audit.read_json(audit.OUT / "upper_support_event_results.json")[
        "records"
    ]
    for target in targets:
        row = base_rows[target["id"]]
        x, note = event_features(row["seq"])
        if x is None or not row["source"]:
            raise ValueError("Existing event inputs lost eligibility or source")
        old.append(
            dict(
                target,
                person=row["person"],
                source=row["source"],
                x=x,
                observation=note,
            )
        )
    paths = {
        r["id"]: r["cache"]
        for r in audit.read_json(audit.OUT / "followup_extraction.json")[
            "event_records"
        ]
    }
    added = []
    excluded = []
    for draft in audit.read_json(
        audit.OUT / "followup_upper_event_direct.json"
    )["reviews"]:
        with np.load(paths[draft["id"]]) as cache:
            seq = cache["image_xy"].copy()
        x, note = event_features(seq)
        if x is None:
            excluded.append(dict(id=draft["id"], observation=note))
            continue
        added.append(
            dict(
                id=draft["id"],
                person=draft["person"],
                source=draft["video_sha256"],
                x=x,
                target=draft["visible_upper_support_change"],
                observation=note,
            )
        )
    all_rows = old + added
    predictions = []
    for person in sorted({r["person"] for r in all_rows}):
        check_resources()
        testing = [r for r in all_rows if r["person"] == person]
        sources = {r["source"] for r in testing}
        before = fit_event(
            [
                r
                for r in old
                if r["person"] != person and r["source"] not in sources
            ]
        )
        after = fit_event(
            [
                r
                for r in all_rows
                if r["person"] != person and r["source"] not in sources
            ]
        )
        for row in testing:
            probs = []
            for model in [before, after]:
                classes = list(model.classes_)
                probs.append(
                    float(model.predict_proba([row["x"]])[0][classes.index(1)])
                    if 1 in classes
                    else 0.0
                )
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target=row["target"],
                    before_probability=probs[0],
                    after_probability=probs[1],
                    before_event=probs[0] >= 0.25,
                    after_event=probs[1] >= 0.25,
                    rule_event=row["observation"]["relative_displacement"]
                    >= 0.2,
                )
            )
    model = fit_event(all_rows)
    folder = audit.OUT / "followup_event_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        dict(
            model=model,
            threshold=0.25,
            train_ids=[r["id"] for r in all_rows],
            output="upper_support_change_probability_not_posture_score",
            expert_validated=False,
        ),
        folder / "model.joblib",
    )
    report = dict(
        training_records=len(all_rows),
        new_training_records=len(added),
        training_people=len({r["person"] for r in all_rows}),
        positive_person_groups=len(
            {r["person"] for r in all_rows if r["target"] == 1}
        ),
        excluded=excluded,
        predictions=predictions,
        metrics=event_metrics(predictions),
        fixed_threshold=0.25,
        threshold_tuned_on_current_results=False,
        source_null_fallback_fixed=True,
        old_historical_fold_scores_not_overwritten=True,
        new_independent_final_test=False,
        expert_validated=False,
        common_model_replaced=False,
        model_sha256=audit.digest(folder / "model.joblib"),
    )
    audit.save_json(audit.OUT / "followup_event_learning_results.json", report)
    return report


if __name__ == "__main__":
    print(run_total()["LOPO_metrics"])
    print(run_events()["metrics"])
