"""관측 구도별 부분 라벨을 실제 학습하고 5종 AI·규칙을 비교한다.

같은 사람·원본은 연결 성분으로 묶는다. 기존 모델과 서비스 코드는
교체하지 않으며 미관측 항목을 채우거나 규칙 출력을 정답으로 쓰지 않는다.
"""

import collections
import hashlib
import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

import audit
import three_items
import three_training
from recovery_features import load
from recovery_prepare import ROOT
from resources import check_resources


def component_groups(rows: list[dict]) -> list[str]:
    """참여자 또는 원본을 공유하는 행을 하나의 연결 그룹으로 묶는다.

    Args:
        rows: person·sources·split 필드를 가진 개발 자료.

    Returns:
        list[str]: 입력 행 순서의 그룹 ID. 전이적으로 연결된 원본도 합친다.

    Raises:
        ValueError: 같은 연결 그룹이 학습과 검증 분할에 걸친 경우.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = ["person:" + row["person"]] + [
            "source:" + s for s in row["sources"]
        ]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    roles = collections.defaultdict(set)
    for i, row in enumerate(rows):
        roles[find(i)].add(row["split"])
    if any(len(v) > 1 for v in roles.values()):
        raise ValueError("Connected source/person crosses development roles")
    return [str(find(i)) for i in range(len(rows))]


def head_key(row: dict, index: int) -> tuple[str, str, str]:
    """운동·항목·관측 방향을 분리한 학습 묶음 식별자를 반환한다.

    Args:
        row: 운동 코드·네 항목 ID·관측 특징 행.
        index: 항목 인덱스, 0~3 정수.

    Returns:
        tuple[str, str, str]: 운동·항목·구도 이름. 물리 단위 없음.

    Raises:
        KeyError: 필수 메타데이터 누락.
        IndexError: 항목 인덱스 범위 이탈.
    """
    return (
        row["exercise"],
        row["criterion_ids"][index],
        row["notes"][index]["bank"],
    )


def fit(rows: list[dict]) -> dict:
    """학습 행만 사용해 항목 AI와 명시 규칙 경계를 각각 학습한다.

    Args:
        rows: 관측 기하 특징과 독립 직접 등급·참여자 행 목록.

    Returns:
        dict: 구도별 회귀기·규칙 경계·실제 학습 수·등급 분포.
        회귀 목표는 0~2 등급이며 규칙 출력은 학습 정답이 아니다.

    Raises:
        ValueError: 특징 차원 또는 회귀 입력 불일치.
    """
    buckets = collections.defaultdict(list)
    for row in rows:
        for index, note in enumerate(row["notes"]):
            grade = row["levels"][index]
            if grade is None or not note["eligible"] or note["feature"] is None:
                continue
            if not np.isfinite(note["feature"]).all():
                continue
            buckets[head_key(row, index)].append((row, index, note, grade))
    models = {}
    for key, bucket in buckets.items():
        counts = collections.Counter((g, r["person"]) for r, i, n, g in bucket)
        grade_people = collections.Counter(g for g, p in counts)
        weights = [
            1 / (grade_people[g] * counts[(g, r["person"])])
            for r, i, n, g in bucket
        ]
        forest = ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=221022, n_jobs=1
        )
        forest.fit(
            [n["feature"] for r, i, n, g in bucket],
            [g for r, i, n, g in bucket],
            sample_weight=weights,
        )
        name = key[1]
        default_key = key[0] + "_" + name if name == "descent_range" else name
        boundary = three_training.DEFAULTS.get(default_key)
        error_people = {r["person"] for r, i, n, g in bucket if g < 2}
        can_calibrate = (
            len(error_people) >= 2
            and len(grade_people) >= 2
            and all(n.get("measurement") is not None for r, i, n, g in bucket)
        )
        if (
            key[2] == "front"
            and key[0] in ("pushups", "squats")
            and name != "trunk_balance"
            and not can_calibrate
        ):
            boundary = None
        if can_calibrate and default_key in three_training.BOUNDARIES:
            good_values, partial_values = three_training.BOUNDARIES[default_key]
            choices = []
            for good in good_values:
                for partial in partial_values:
                    high = bucket[0][2]["higher_is_better"]
                    if (high and good <= partial) or (
                        not high and good >= partial
                    ):
                        continue
                    b = (good, partial)
                    loss = np.average(
                        [
                            abs(three_training.rule_grade(n, b) - g)
                            for r, i, n, g in bucket
                        ],
                        weights=weights,
                    )
                    choices.append((loss, b))
            if choices:
                boundary = min(choices, key=lambda x: x[0])[1]
        models[key] = dict(
            model=forest,
            boundary=boundary,
            records=len(bucket),
            people=len({r["person"] for r, i, n, g in bucket}),
            grade_counts=dict(
                collections.Counter(str(g) for r, i, n, g in bucket)
            ),
            error_people=len(error_people),
            single_grade_model=len(grade_people) < 2,
        )
    return models


def predict(models: dict, row: dict) -> dict:
    """관측 항목만 AI·규칙으로 예측하고 미학습 구도는 None으로 반환한다.

    Args:
        models: fit이 반환한 구도별 모델 사전.
        row: 관측 특징·항목 배점을 가진 평가 입력.

    Returns:
        dict: 0~2 항목 예측과 0~100점 총점 또는 None. 누락을 재환산하지 않는다.

    Raises:
        ValueError: 특징 차원·배점·회귀 입력 오류.
    """
    ai = []
    rules = []
    for index, note in enumerate(row["notes"]):
        head = models.get(head_key(row, index))
        a = None
        r = None
        if (
            head is not None
            and note["eligible"]
            and note["feature"] is not None
            and np.isfinite(note["feature"]).all()
        ):
            a = float(
                np.clip(head["model"].predict([note["feature"]])[0], 0, 2)
            )
            if "rule_grade" in note:
                r = note["rule_grade"]
            elif head["boundary"] is not None:
                r = three_training.rule_grade(note, head["boundary"])
        ai.append(a)
        rules.append(r)
    return dict(
        ai_levels=ai,
        rule_levels=rules,
        ai_score=three_items.total(ai, row["weights"]),
        rule_score=three_items.total(rules, row["weights"]),
    )


def summarize(rows: list[dict]) -> dict:
    """동일 평가 행의 총점·항목 오차와 평가 가능 수를 집계한다.

    Args:
        rows: 직접 등급·총점과 두 방식의 같은 입력 예측 행 목록.

    Returns:
        dict: 공통 표본 총점 MAE(점), 항목 MAE(등급), 판정 가능 수.

    Raises:
        KeyError: 필수 예측·참조 필드 누락.
    """
    result = dict(submitted=len(rows))
    paired = [
        r
        for r in rows
        if r["target_score"] is not None
        and r["ai_score"] is not None
        and r["rule_score"] is not None
    ]
    result["paired_total_records"] = len(paired)
    result["paired_total_MAE_points"] = {
        k: (
            float(np.mean([abs(r[k] - r["target_score"]) for r in paired]))
            if paired
            else None
        )
        for k in ["ai_score", "rule_score"]
    }
    item_counts = collections.defaultdict(
        lambda: dict(pairs=0, ai_error=0.0, rule_error=0.0)
    )
    for r in rows:
        for name, g, a, b in zip(
            r["criterion_ids"], r["levels"], r["ai_levels"], r["rule_levels"]
        ):
            if g is None or a is None or b is None:
                continue
            c = item_counts[name]
            c["pairs"] += 1
            c["ai_error"] += abs(a - g)
            c["rule_error"] += abs(b - g)
    result["items"] = {
        k: dict(
            paired_records=v["pairs"],
            ai_MAE_grade_units=v["ai_error"] / v["pairs"],
            rule_MAE_grade_units=v["rule_error"] / v["pairs"],
        )
        for k, v in item_counts.items()
    }
    result["ai_total_available"] = sum(r["ai_score"] is not None for r in rows)
    result["rule_total_available"] = sum(
        r["rule_score"] is not None for r in rows
    )
    return result


def run() -> dict:
    """5종의 별도 후보를 학습하고 고정 개발 검증·연결 그룹 교차검증을 저장한다.

    Returns:
        dict: 구도별 실제 학습 수·등급 분포·고정 검증·교차검증 결과.
        70개 목표나 독립 최종 시험 완료를 주장하지 않는다.

    Raises:
        OSError: 입력·캐시·모델 저장 실패.
        ValueError: 개발 분할 누출·원본 계약 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    base, extra = load()
    all_rows = base + extra
    report = dict(
        exercises=[],
        independent_final_test=False,
        expert_reference=False,
        all_five_complete=False,
        external_discovery_paused=True,
    )
    for exercise in ("pushups", "squats", "shoulder", "plank", "ytw"):
        check_resources()
        rows = [x for x in all_rows if x["exercise"] == exercise]
        groups = component_groups(rows)
        train = [x for x in rows if x["split"] == "train"]
        val = [x for x in rows if x["split"] == "validation"]
        before = fit(
            [
                x
                for x in base
                if x["exercise"] == exercise and x["split"] == "train"
            ]
        )
        model = fit(train)
        predictions = []
        for row in val:
            predictions.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    criterion_ids=row["criterion_ids"],
                    levels=row["levels"],
                    target_score=three_items.total(
                        row["levels"], row["weights"]
                    ),
                    before=predict(before, row),
                    **predict(model, row)
                )
            )
        folds = []
        train_indices = [i for i, x in enumerate(rows) if x["split"] == "train"]
        training_groups = [groups[i] for i in train_indices]
        if len(set(training_groups)) >= 3:
            for fitting, held in GroupKFold(
                min(5, len(set(training_groups)))
            ).split(train, groups=training_groups):
                check_resources()
                fitting_rows = [train[int(i)] for i in fitting]
                held_rows = [train[int(i)] for i in held]
                assert not set().union(
                    *(r["sources"] for r in fitting_rows)
                ) & set().union(*(r["sources"] for r in held_rows))
                folded = fit(fitting_rows)
                for row in held_rows:
                    folds.append(
                        dict(
                            id=row["id"],
                            person=row["person"],
                            criterion_ids=row["criterion_ids"],
                            levels=row["levels"],
                            target_score=three_items.total(
                                row["levels"], row["weights"]
                            ),
                            **predict(folded, row)
                        )
                    )
        path = ROOT / "models" / (exercise + ".joblib")
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)
        effective = {
            r["id"]
            for r in train
            if any(
                n["eligible"] and g is not None
                for n, g in zip(r["notes"], r["levels"])
            )
        }
        entry = dict(
            exercise=exercise,
            training_submitted=len(train),
            training_effective_segments=len(effective),
            training_people=len({r["person"] for r in train}),
            target_segments=70,
            target_met=len(effective) >= 70,
            model_path=str(path),
            model_sha256=audit.digest(path),
            heads=[
                dict(
                    key=list(k), **{n: v for n, v in h.items() if n != "model"}
                )
                for k, h in model.items()
            ],
            fixed_validation=summarize(predictions),
            fixed_predictions=predictions,
            group_cv=summarize(folds),
            group_cv_predictions=folds,
            cv_scope="up_to_five_connected_person_source_folds_not_LOPO",
            complete=False,
        )
        report["exercises"].append(entry)
        audit.save_json(ROOT / "comparison.json", report)
    report["code_sha256"] = {
        p.name: audit.digest(p)
        for p in pathlib.Path(__file__).parent.glob("recovery_*.py")
    }
    report["input_plan_sha256"] = audit.digest(ROOT / "input_plan.json")
    audit.save_json(ROOT / "comparison.json", report)
    return dict(
        exercises=[
            dict(
                exercise=x["exercise"],
                training=x["training_effective_segments"],
                target_met=x["target_met"],
            )
            for x in report["exercises"]
        ]
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
