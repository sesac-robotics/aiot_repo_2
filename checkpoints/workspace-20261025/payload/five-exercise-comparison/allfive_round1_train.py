"""보유 5종 보강 입력으로 AI·명시 규칙 후보를 별도 학습·비교한다."""

import collections
import copy
import itertools
import json
import pathlib

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

import audit
import recovery_train
import three_training
from allfive_round1 import OUT
from allfive_round1_features import load
from multiview_squat_batch2 import OUT as SQUAT_BATCH2
from multiview_squat_batch2_train import prediction
from recovery_prepare import ROOT
from recovery_w_calibration import DEFAULT as W_DEFAULT
from recovery_w_shape_trial import classify as w_classify
from resources import check_resources

PLANK_BOUNDS = {
    "trunk_line": (
        (168, 155),
        (155, 160, 165, 168, 170, 175),
        (140, 145, 150, 155, 160),
    ),
    "elbow_sagittal_position": (
        (0.2, 0.4),
        (0.1, 0.15, 0.2, 0.25, 0.3),
        (0.3, 0.4, 0.5, 0.6),
    ),
    "leg_and_body_support": (
        (165, 145),
        (155, 160, 165, 170, 175),
        (130, 140, 145, 150, 155),
    ),
    "support_continuity": ((8, 20), (5, 8, 10, 15), (15, 20, 25, 30)),
}


def balanced(bucket: list[tuple]) -> list[float]:
    """등급·참여자 메타 그룹별 총 가중치를 같게 계산한다.

    Args:
        bucket: (행, 항목, 등급)의 학습 묶음.

    Returns:
        list[float]: 행별 양수 가중치. 검증 데이터는 입력하지 않는다.
    """
    counts = collections.Counter((g, r["person"]) for r, n, g in bucket)
    groups = collections.Counter(g for g, p in counts)
    return [1 / (groups[g] * counts[g, r["person"]]) for r, n, g in bucket]


def w_policies() -> np.ndarray:
    """기존 별도 W 실험의 손 위치·팔꿈치 하한 후보를 반환한다.

    Returns:
        np.ndarray: P×9 명시 경계. 신규 검증 점수로 범위를 추가하지 않는다.
    """
    policies = []
    for hand, good, partial in itertools.product(
        (0, 0.025, 0.05, 0.1),
        (-1, -0.95, -0.9, -0.85, -0.8),
        (-1, -0.98, -0.95, -0.9),
    ):
        if partial > good:
            continue
        p = list(W_DEFAULT) + [good, partial]
        p[2] = hand
        policies.append(p)
    return np.asarray(policies)


def fit(rows: list[dict]) -> dict:
    """학습 역할만으로 회귀·규칙 경계·측면 발 기울기 조건을 선택한다.

    Args:
        rows: 직접 독립 등급·관측 특징이 연결된 train 역할 행.

    Returns:
        dict: 항목·구도별 ExtraTrees와 명시 규칙 경계. 검증 라벨·제공자
        문자열은 선택에 쓰지 않으며 오류 그룹이 부족하면 기존 규칙을 유지한다.

    Raises:
        ValueError: 검증 역할 혼입·특징 차원·경계 계약 위반.
    """
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Training-only calibration required")
    model = recovery_train.fit(rows)
    buckets = collections.defaultdict(list)
    for r in rows:
        for i, (n, g) in enumerate(zip(r["notes"], r["levels"])):
            if (
                n["eligible"]
                and g is not None
                and n["feature"] is not None
                and np.isfinite(n["feature"]).all()
            ):
                buckets[recovery_train.head_key(r, i)].append((r, n, g))
    for key, bucket in buckets.items():
        head = model[key]
        weights = balanced(bucket)
        y = np.asarray([g for r, n, g in bucket])
        groups = collections.defaultdict(set)
        for r, n, g in bucket:
            groups[g].add(r["person"])
        sufficient = (
            len(groups) >= 2
            and min(len(v) for v in groups.values()) >= 2
            and len({r["person"] for r, n, g in bucket if g < 2}) >= 2
        )
        if key[0] == "plank" and key[2] == "side_video" and sufficient:
            default, good, partial = PLANK_BOUNDS[key[1]]
            high = bucket[0][1]["higher_is_better"]
            candidates = (
                [
                    (a, b)
                    for a, b in itertools.product(good, partial)
                    if a > b
                    if high
                ]
                if high
                else [
                    (a, b) for a, b in itertools.product(good, partial) if a < b
                ]
            )
            candidates.append(default)

            def loss(bounds):
                return float(
                    np.average(
                        [
                            abs(three_training.rule_grade(n, bounds) - g)
                            for r, n, g in bucket
                        ],
                        weights=weights,
                    )
                )

            selected = min(
                candidates,
                key=lambda p: (
                    loss(p),
                    sum(abs(a - b) for a, b in zip(p, default)),
                ),
            )
            head.update(
                explicit_rule_kind="plank_scalar_v1",
                boundary=selected,
                rule_calibration=dict(
                    default=default,
                    selected=selected,
                    training_weighted_MAE_before=loss(default),
                    training_weighted_MAE_after=loss(selected),
                    training_only=True,
                ),
            )
        if key[0] == "ytw" and key[1] == "w_shape" and sufficient:
            policies = w_policies()
            x = np.asarray([n["feature"] for r, n, g in bucket])
            grades = w_classify(x, policies)
            losses = np.average(
                abs(grades - y[None, :]), axis=1, weights=weights
            )
            prior = np.asarray(list(W_DEFAULT) + [-1, -1])
            distance = np.sum(
                abs(policies - prior)
                / np.asarray([20, 8, 0.1, 0.1, 0.5, 0.5, 35, 0.2, 0.1]),
                axis=1,
            )
            idx = min(
                range(len(policies)), key=lambda i: (losses[i], distance[i], i)
            )
            head.update(
                explicit_rule_kind="W_shape_v1",
                W_policy=policies[idx].tolist(),
                rule_calibration=dict(
                    training_only=True,
                    weighted_grade_MAE=float(losses[idx]),
                    candidate_policies=len(policies),
                    structure_previously_informed_by_development_failures=True,
                ),
            )
        if key[0] == "squats" and key[2] == "side_pitch_v1" and sufficient:
            prior = head["boundary"] or (0.25, 0.5, 0.025, 0.08)
            prior = tuple(prior) if len(prior) == 4 else (*prior, 0.025, 0.08)
            boundary_candidates = [
                (*prior[:2], a, b)
                for a, b in itertools.product(
                    (0.01, 0.025, 0.04, 0.06), (0.04, 0.06, 0.08, 0.12, 0.16)
                )
                if a < b
            ]
            policies = [
                (b, p)
                for b in boundary_candidates
                for p in [
                    None,
                    *[
                        (a, c)
                        for a, c in itertools.product(
                            (10, 15, 20, 25, 30), (25, 35, 45, 60)
                        )
                        if a < c
                    ],
                ]
            ]

            def pitch_loss(policy):
                b, pitch = policy
                values = []
                for r, n, g in bucket:
                    grade = three_training.rule_grade(n, b)
                    if pitch is not None:
                        grade = min(
                            grade,
                            (
                                2
                                if n["feature"][9] <= pitch[0]
                                else 1 if n["feature"][9] <= pitch[1] else 0
                            ),
                        )
                    values.append(abs(grade - g))
                return float(np.average(values, weights=weights))

            selected = min(
                policies,
                key=lambda p: (
                    pitch_loss(p),
                    p[1] is not None,
                    sum(abs(a - b) for a, b in zip(p[0], prior)),
                ),
            )
            head.update(
                explicit_rule_kind="knee_heel_pitch_v1",
                boundary=selected[0],
                pitch_boundary=selected[1],
                rule_calibration=dict(
                    training_only=True,
                    selected_pitch_boundaries_degrees=selected[1],
                    training_weighted_MAE=pitch_loss(selected),
                    pitch_condition_can_be_disabled=True,
                ),
            )
    return model


def predict(model: dict, row: dict) -> dict:
    """같은 관측 항목에 저장된 후보 규칙·회귀를 적용한다.

    Args:
        model: 학습 역할에서 선택한 항목·구도별 모델과 명시 경계.
        row: 직접 등급과 독립적으로 계산한 관측 특징 행.

    Returns:
        dict: 0~2 항목 등급·0~100 총점·보류 근거. 미관측 항목은
        채우지 않으며 입력 행과 기존 공통 규칙은 변경하지 않는다.

    Raises:
        ValueError: 특징 차원·규칙 경계·회귀 입력 계약 불일치.
    """
    local = dict(row, notes=[dict(n) for n in row["notes"]])
    for i, n in enumerate(local["notes"]):
        head = model.get(recovery_train.head_key(local, i))
        if head is None or not n["eligible"]:
            continue
        kind = head.get("explicit_rule_kind")
        if kind == "plank_scalar_v1":
            n["rule_grade"] = three_training.rule_grade(n, head["boundary"])
        elif kind == "W_shape_v1":
            n["rule_grade"] = int(
                w_classify(
                    np.asarray([n["feature"]]), np.asarray([head["W_policy"]])
                )[0, 0]
            )
        elif kind == "knee_heel_pitch_v1":
            grade = three_training.rule_grade(n, head["boundary"])
            pitch = head["pitch_boundary"]
            if pitch is not None:
                grade = min(
                    grade,
                    (
                        2
                        if n["feature"][9] <= pitch[0]
                        else 1 if n["feature"][9] <= pitch[1] else 0
                    ),
                )
            n["rule_grade"] = grade
    return prediction(model, local)


def manifest(rows: list[dict]) -> list[dict]:
    """점수 입력·관측·배점·출처만 JSON 직렬화 가능한 목록으로 반환한다.

    Args:
        rows: 기하 특징과 직접 등급이 연결된 행.

    Returns:
        list[dict]: 관측 불가 특징은 null인 감사 입력. 실제 좌표·정답을
        특징에 덧붙이지 않으며 구도별 특징 차원을 함께 기록한다.
    """
    return [
        dict(
            id=r["id"],
            exercise=r["exercise"],
            person=r["person"],
            split=r["split"],
            sources=sorted(r["sources"]),
            recording_groups=r.get("recording_groups", []),
            levels=r["levels"],
            reference_total_allowed=r.get("reference_total_allowed", True),
            whole_attempt_observed=r.get("whole_attempt_observed", True),
            items=[
                dict(
                    item=n["item"],
                    bank=n["bank"],
                    eligible=n["eligible"],
                    feature=n["feature"] if n["eligible"] else None,
                    holds=n.get("holds", []),
                )
                for n in r["notes"]
            ],
        )
        for r in rows
    ]


def run() -> dict:
    """5종별 같은 개발 검증과 연결 그룹 교차검증을 수행해 후보를 저장한다.

    Returns:
        dict: 항목·총점 오차·관측률·라벨 분포·고정 검증과 내부 교차검증.
        전문가 정답·독립 최종 시험·자동 모델 승격을 뜻하지 않는다.

    Raises:
        ValueError: 원본·참여자 분할·보존 지문·특징 계약 불일치.
        OSError: 입력·모델·감사 결과 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    protected = [
        ROOT / "comparison.json",
        ROOT / "fit_input_manifest.json",
        SQUAT_BATCH2 / "comparison.json",
        SQUAT_BATCH2 / "squats_candidate.joblib",
        *list((ROOT / "models").glob("*.joblib")),
    ]
    preserved = {str(p): audit.digest(p) for p in protected}
    before, after, changes = load()
    recovery_train.component_groups(before)
    recovery_train.component_groups(after)
    audit.save_json(OUT / "feature_changes.json", changes)
    audit.save_json(OUT / "baseline_input_manifest.json", manifest(before))
    audit.save_json(OUT / "candidate_input_manifest.json", manifest(after))
    old_by_id = {r["id"]: r for r in before}
    reports = []
    (OUT / "models").mkdir(exist_ok=True)
    for exercise in ("pushups", "squats", "shoulder", "plank", "ytw"):
        check_resources()
        rows = [r for r in after if r["exercise"] == exercise]
        train = [r for r in rows if r["split"] == "train"]
        val = [r for r in rows if r["split"] == "validation"]
        groups = recovery_train.component_groups(train)
        model = fit(train)
        joblib.dump(model, OUT / "models" / (exercise + ".joblib"))
        baseline = joblib.load(
            SQUAT_BATCH2 / "squats_candidate.joblib"
            if exercise == "squats"
            else ROOT / "models" / (exercise + ".joblib")
        )
        fixed = [
            dict(
                predict(model, r),
                baseline=prediction(baseline, old_by_id[r["id"]]),
            )
            for r in val
        ]
        oof = []
        if len(set(groups)) >= 3:
            for fitting, held in GroupKFold(min(3, len(set(groups)))).split(
                train, groups=groups
            ):
                check_resources()
                fold = fit([train[int(i)] for i in fitting])
                oof.extend(predict(fold, train[int(i)]) for i in held)
        added = [
            r
            for r in train
            if "_round1_" in r["id"] or r["id"].startswith("held_navy")
        ]
        effective = [
            r
            for r in train
            if any(
                n["eligible"] and g is not None
                for n, g in zip(r["notes"], r["levels"])
            )
        ]
        reports.append(
            dict(
                exercise=exercise,
                training_segments=len(effective),
                training_metadata_groups=len({r["person"] for r in effective}),
                training_full_observed=sum(
                    all(n["eligible"] for n in r["notes"])
                    and None not in r["levels"]
                    and r.get("reference_total_allowed", True)
                    for r in train
                ),
                new_score_drafts=len(added),
                new_eligible_item_labels=sum(
                    n["eligible"] and g is not None
                    for r in added
                    for n, g in zip(r["notes"], r["levels"])
                ),
                heads=[
                    dict(
                        item=k[1],
                        view=k[2],
                        records=v["records"],
                        metadata_groups=v["people"],
                        grade_counts=v["grade_counts"],
                        error_metadata_groups=v["error_people"],
                        single_grade_model=v["single_grade_model"],
                        boundary=v.get("boundary"),
                        explicit_rule_kind=v.get("explicit_rule_kind"),
                        rule_calibration=v.get("rule_calibration"),
                    )
                    for k, v in model.items()
                ],
                fixed_baseline=recovery_train.summarize(
                    [p["baseline"] for p in fixed]
                ),
                fixed_candidate=recovery_train.summarize(fixed),
                fixed_predictions=fixed,
                group_cv=recovery_train.summarize(oof),
                group_cv_predictions=oof,
                model_sha256=audit.digest(
                    OUT / "models" / (exercise + ".joblib")
                ),
            )
        )
        audit.save_json(
            OUT / "comparison.json",
            dict(exercises=reports, completed=False, preserved=preserved),
        )
    if not all(
        audit.digest(pathlib.Path(p)) == sha for p, sha in preserved.items()
    ):
        raise ValueError("Preserved model changed")
    report = dict(
        exercises=reports,
        completed=True,
        preserved=preserved,
        code_sha256={
            name: audit.digest(pathlib.Path(__file__).with_name(name))
            for name in (
                "allfive_round1_train.py",
                "allfive_round1_features.py",
                "allfive_round1_inputs.py",
                "recovery_features.py",
                "recovery_train.py",
                "recovery_w_phase.py",
                "recovery_w_shape_trial.py",
                "three_items.py",
                "three_training.py",
                "geometry.py",
                "plank_item_observation.py",
            )
        },
        input_sha256={
            name: audit.digest(OUT / name)
            for name in (
                "score_annotations.json",
                "subject_tracking_review.json",
                "manual_decisions.json",
                "extraction.json",
                "candidate_input_manifest.json",
                "baseline_input_manifest.json",
            )
        },
        validation_used_for_numeric_fitting=False,
        independent_final_test=False,
        expert_reference=False,
        promoted=False,
        external_discovery=False,
    )
    audit.save_json(OUT / "comparison.json", report)
    return dict(
        exercises=[
            dict(
                exercise=r["exercise"],
                training=r["training_segments"],
                new_eligible_item_labels=r["new_eligible_item_labels"],
                fixed=r["fixed_candidate"],
            )
            for r in reports
        ]
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
