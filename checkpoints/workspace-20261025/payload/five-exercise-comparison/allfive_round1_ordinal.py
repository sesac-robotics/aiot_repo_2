"""학습 그룹 내부에서 연속 회귀·등급 분류·발 특징 선택을 비교한다."""

import collections
import copy
import json
import pathlib

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import audit
import recovery_train
from allfive_round1 import OUT
from allfive_round1_inputs import DEFINITIONS
from allfive_round1_report import paired_summary
from resources import check_resources

DEST = OUT / "ordinal_trial"


def estimator(kind: str, columns: list[int], dimension: int) -> Pipeline:
    """관측 특징 열과 회귀·분류의 고정 후보를 구성한다.

    Args:
        kind: regression 또는 ordinal_classification.
        columns: 사용할 관측 특징 열 번호. 라벨·규칙 점수는 포함하지 않는다.
        dimension: 전체 입력 특징 수.

    Returns:
        sklearn.Pipeline: 선택 열과 ExtraTrees 128개. 분류도 0·1·2 등급을
        예측하는 별도 AI이며 규칙 라벨이나 정상 문자열을 목표로 쓰지 않는다.

    Raises:
        ValueError: 알 수 없는 후보 또는 잘못된 특징 열.
    """
    if (
        kind not in ("regression", "ordinal_classification")
        or not columns
        or any(i < 0 or i >= dimension for i in columns)
    ):
        raise ValueError("Invalid estimator policy")
    cls = ExtraTreesRegressor if kind == "regression" else ExtraTreesClassifier
    return Pipeline(
        [
            (
                "columns",
                ColumnTransformer(
                    [("observed", "passthrough", columns)], remainder="drop"
                ),
            ),
            (
                "forest",
                cls(
                    n_estimators=128,
                    min_samples_leaf=2,
                    random_state=221022,
                    n_jobs=1,
                ),
            ),
        ]
    )


def weights(rows: list[dict]) -> list[float]:
    """학습 등급과 메타 참여자별 총 가중치를 같게 반환한다.

    Args:
        rows: grade·person이 있는 학습 행.

    Returns:
        list[float]: 해당 학습 폴드만으로 구한 표본 가중치.
    """
    counts = collections.Counter((r["grade"], r["person"]) for r in rows)
    groups = collections.Counter(g for g, p in counts)
    return [
        1 / (groups[r["grade"]] * counts[r["grade"], r["person"]]) for r in rows
    ]


def choose(rows: list[dict]) -> tuple[Pipeline, dict]:
    """같은 학습 연결 그룹 폴드로만 예측 방식과 특징 열을 선택한다.

    Args:
        rows: train 역할의 관측 특징·직접 등급·원본·참여자 연결 행.

    Returns:
        tuple: 전체 학습으로 재학습한 선택 모델과 내부 선택 오차. 이 오차는
        선택에 사용했으므로 독립 성능 검증으로 보고하지 않는다.

    Raises:
        ValueError: 검증 역할 혼입·관측 특징·그룹 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Training-only selection required")
    dimension = len(rows[0]["x"])
    all_columns = list(range(dimension))
    column_sets = [all_columns]
    if rows[0]["view"] == "side_pitch_v1":
        column_sets.append(list(range(8)) + [10])
    policies = [
        (kind, columns)
        for columns in column_sets
        for kind in ("regression", "ordinal_classification")
    ]
    x = np.asarray([r["x"] for r in rows])
    y = np.asarray([r["grade"] for r in rows])
    groups = recovery_train.component_groups(rows)
    results = []
    for kind, columns in policies:
        check_resources()
        oof = np.full(len(rows), np.nan)
        for train, held in GroupKFold(min(3, len(set(groups)))).split(
            rows, groups=groups
        ):
            fitted = estimator(kind, columns, dimension)
            fitted.fit(
                x[train],
                y[train],
                forest__sample_weight=weights([rows[int(i)] for i in train]),
            )
            oof[held] = fitted.predict(x[held])
        results.append(
            dict(
                kind=kind,
                columns=columns,
                selection_group_cv_MAE=float(
                    np.average(abs(oof - y), weights=weights(rows))
                ),
            )
        )
    selected = min(
        range(len(results)),
        key=lambda i: (results[i]["selection_group_cv_MAE"], i),
    )
    policy = results[selected]
    model = estimator(policy["kind"], policy["columns"], dimension)
    model.fit(x, y, forest__sample_weight=weights(rows))
    return model, dict(
        selected=policy,
        candidates=results,
        training_only=True,
        selection_metric_is_not_independent_evaluation=True,
        group_folds=3,
    )


def run() -> dict:
    """기존 5종 후보를 보존하고 선택된 AI만 같은 고정 개발 입력에 비교한다.

    Returns:
        dict: 항목별 학습 내부 선택·고정 개발 비교·모델 보존 지문. 규칙은
        첫 후보 그대로이며 검증 라벨로 후보나 수치 경계를 선택하지 않는다.

    Raises:
        ValueError: 기준·입력·모델 지문·분할 계약 불일치.
        OSError: 입력·모델·감사 결과 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    DEST.mkdir(exist_ok=True)
    (DEST / "models").mkdir(exist_ok=True)
    report = audit.read_json(OUT / "comparison.json")
    inputs = audit.read_json(OUT / "candidate_input_manifest.json")
    manifest_sha = audit.digest(OUT / "candidate_input_manifest.json")
    by_id = {r["id"]: r for r in inputs}
    experiments = []
    for exercise in report["exercises"]:
        check_resources()
        e = exercise["exercise"]
        model = joblib.load(OUT / "models" / (e + ".joblib"))
        model_sha = audit.digest(OUT / "models" / (e + ".joblib"))
        buckets = collections.defaultdict(list)
        for row in inputs:
            if row["exercise"] != e or row["split"] != "train":
                continue
            for i, (note, grade) in enumerate(zip(row["items"], row["levels"])):
                if (
                    not note["eligible"]
                    or grade is None
                    or note["feature"] is None
                ):
                    continue
                key = (e, note["item"], note["bank"])
                buckets[key].append(
                    dict(
                        id=row["id"],
                        person=row["person"],
                        sources=set(row["sources"]),
                        recording_groups=row.get("recording_groups", []),
                        split="train",
                        x=note["feature"],
                        grade=grade,
                        view=note["bank"],
                    )
                )
        selections = {}
        for key, bucket in buckets.items():
            grade_people = collections.defaultdict(set)
            for r in bucket:
                grade_people[r["grade"]].add(r["person"])
            if (
                len(grade_people) < 2
                or min(len(p) for p in grade_people.values()) < 2
                or len({r["person"] for r in bucket}) < 6
                or key[:2] == ("pushups", "trunk_line")
            ):
                continue
            selected, diagnostic = choose(bucket)
            model[key]["model"] = selected
            model[key]["AI_selection"] = diagnostic
            selections[key] = diagnostic
        predictions = []
        for previous in exercise["fixed_predictions"]:
            row = by_id[previous["id"]]
            p = copy.deepcopy(previous)
            p.pop("baseline", None)
            p["baseline"] = copy.deepcopy(previous)
            p["baseline"].pop("baseline", None)
            for i, note in enumerate(row["items"]):
                key = (e, note["item"], note["bank"])
                if key in selections and note["eligible"]:
                    p["ai_levels"][i] = float(
                        np.clip(
                            model[key]["model"].predict([note["feature"]])[0],
                            0,
                            2,
                        )
                    )
            p["ai_score"] = (
                sum(
                    w * g / 2 for w, g in zip(DEFINITIONS[e][1], p["ai_levels"])
                )
                if row.get("whole_attempt_observed", True)
                and None not in p["ai_levels"]
                else None
            )
            predictions.append(p)
        joblib.dump(model, DEST / "models" / (e + ".joblib"))
        if audit.digest(OUT / "models" / (e + ".joblib")) != model_sha:
            raise ValueError("Round1 model changed")
        experiments.append(
            dict(
                exercise=e,
                selection=[
                    dict(item=k[1], view=k[2], **v)
                    for k, v in selections.items()
                ],
                fixed_predictions=predictions,
                paired_comparison=paired_summary(predictions),
                fixed_metrics=recovery_train.summarize(predictions),
                model_sha256=audit.digest(DEST / "models" / (e + ".joblib")),
                preserved_round1_model_sha256=model_sha,
            )
        )
        audit.save_json(
            DEST / "comparison.json",
            dict(exercises=experiments, completed=False),
        )
    result = dict(
        exercises=experiments,
        completed=True,
        input_sha256=manifest_sha,
        round1_comparison_sha256=audit.digest(OUT / "comparison.json"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        validation_used_for_selection=False,
        rule_changed=False,
        selection_cv_not_independent_metric=True,
        independent_final_test=False,
        expert_reference=False,
        promoted=False,
    )
    audit.save_json(DEST / "comparison.json", result)
    return dict(
        exercises=[
            dict(exercise=r["exercise"], paired=r["paired_comparison"])
            for r in experiments
        ]
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
