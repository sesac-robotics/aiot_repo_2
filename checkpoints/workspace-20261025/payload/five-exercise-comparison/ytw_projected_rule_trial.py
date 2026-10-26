"""동일 직접 초안의 Y·T·W·전경 화면 특징으로 명시 규칙 후보를 검증한다.

각도는 화면 투영 값이며 실제 고관절·견갑 각도가 아니다. 관측·직접 라벨
교집합에서만 개발 비교하고 원본 구도 계약·배점·정답 등급은 변경하지 않는다.
"""

import collections
import itertools
import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

import audit
import recovery_train
from allfive_round1 import OUT
from allfive_round1_train import w_policies
from recovery_w_calibration import DEFAULT as W_DEFAULT
from recovery_w_shape_trial import classify as w_classify
from resources import check_resources

DEST = OUT / "ytw_projected_rule_trial"


def grades(x: np.ndarray, policy: tuple, item: int) -> np.ndarray:
    """관측된 화면 기하에 명시 조건을 적용해 0·1·2 등급을 반환한다.

    Args:
        x: N×6(Y·T·전경) 또는 N×7(W) 관측 특징. 결측은 허용하지 않는다.
        policy: 정상·부분 각도(도), 몸통 비율, 흔들림 경계.
        item: Y=0, T=1, W=2, 전경·흔들림=3.

    Returns:
        np.ndarray: 명시 조건의 최저 충족 등급. W는 양팔 굽힘·손 위치·
        팔꿈치 상하·대칭 조건이다. 전경은 화면 몸통 비율의 대리값이며
        숨은 실제 고관절 각도를 복원하거나 안전·교정 정답을 의미하지 않는다.

    Raises:
        ValueError: 미관측 특징·항목·경계 순서 불일치.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Finite observed features required")
    if item == 2:
        return w_classify(x, np.asarray([policy]))[0]
    if x.shape[1] != 6 or item not in (0, 1, 3):
        raise ValueError("Six observed projection features required")
    if item in (0, 1):
        good, partial, elbow_good, elbow_partial = policy
        if good <= partial or elbow_good <= elbow_partial:
            raise ValueError("Ordered elevation and elbow boundaries required")
        elevation = x[:, 1] if item == 0 else 90 - abs(x[:, 1] - 90)
        elbow = x[:, 4]
        return np.minimum(
            np.where(
                elevation >= good, 2, np.where(elevation >= partial, 1, 0)
            ),
            np.where(
                elbow >= elbow_good, 2, np.where(elbow >= elbow_partial, 1, 0)
            ),
        )
    good, partial, stable_good, stable_partial = policy
    if good <= partial or stable_good >= stable_partial:
        raise ValueError(
            "Ordered projected torso and stability boundaries required"
        )
    return np.minimum(
        np.where(x[:, 1] >= good, 2, np.where(x[:, 1] >= partial, 1, 0)),
        np.where(
            x[:, 5] - x[:, 3] <= stable_good,
            2,
            np.where(x[:, 5] - x[:, 3] <= stable_partial, 1, 0),
        ),
    )


def policies(item: int) -> list[tuple]:
    """공통 운동 기준에 대응하는 학습 내부 명시 경계 후보를 반환한다.

    Args:
        item: 0~3 항목 인덱스.

    Returns:
        list[tuple]: 순서가 유효한 고정 물리·비율 경계 후보.
    """
    if item == 2:
        return [tuple(p) for p in w_policies()]
    if item in (0, 1):
        good = (110, 120, 130, 140, 150) if item == 0 else (65, 70, 75, 80)
        partial = (65, 80, 95, 105) if item == 0 else (40, 50, 60, 70)
        return [
            (a, b, c, d)
            for a, b, c, d in itertools.product(
                good, partial, (130, 145, 160), (80, 100, 120)
            )
            if a > b and c > d
        ]
    return [
        (a, b, c, d)
        for a, b, c, d in itertools.product(
            (1.2, 1.5, 1.8, 2, 2.5, 3, 4),
            (0.7, 0.9, 1, 1.2, 1.5, 2),
            (5, 10, 15),
            (15, 25, 40),
        )
        if a > b and c < d
    ]


def fit(rows: list[dict]) -> dict:
    """동일 학습 그룹의 직접 등급으로 AI와 항목별 명시 경계를 구한다.

    Args:
        rows: train 역할의 4항목 화면 특징·관측·직접 초안 행.

    Returns:
        dict: 항목별 회귀기·경계·등급/참여자 수. 각 경계는 학습 역할에서만
        선택하며 정상 Y가 적다는 한계는 유지한다.

    Raises:
        ValueError: 검증 역할 혼입·유효 특징 부족·차원 불일치.
    """
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Training-only policy fitting")
    heads = {}
    defaults = {
        0: (135, 95, 150, 100),
        1: (75, 60, 145, 100),
        2: tuple(W_DEFAULT) + (-1, -1),
        3: (1.8, 1, 10, 25),
    }
    for item in range(4):
        bucket = [
            r
            for r in rows
            if r["items"][item]["eligible"]
            and r["levels"][item] is not None
            and r["items"][item]["feature"] is not None
        ]
        if not bucket:
            continue
        x = np.asarray([r["items"][item]["feature"] for r in bucket])
        y = np.asarray([r["levels"][item] for r in bucket])
        pairs = collections.Counter((g, r["person"]) for r, g in zip(bucket, y))
        people = collections.Counter(g for g, p in pairs)
        weights = [
            1 / (people[g] * pairs[g, r["person"]]) for r, g in zip(bucket, y)
        ]
        model = ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=221022, n_jobs=1
        )
        model.fit(x, y, sample_weight=weights)
        candidates = (
            policies(item)
            if len(people) >= 2 and min(people.values()) >= 2
            else []
        )
        candidates.append(defaults[item])
        scores = [
            float(np.average(abs(grades(x, p, item) - y), weights=weights))
            for p in candidates
        ]
        chosen = min(
            range(len(candidates)),
            key=lambda i: (scores[i], candidates[i] != defaults[item], i),
        )
        heads[item] = dict(
            model=model,
            policy=candidates[chosen],
            records=len(bucket),
            grade_counts=dict(collections.Counter(str(g) for g in y)),
            grade_metadata_groups={
                str(int(g)): int(n) for g, n in people.items()
            },
            training_only=True,
            used_default_for_insufficient_grade_groups=len(people) < 2
            or min(people.values()) < 2,
        )
    return heads


def predict(heads: dict, row: dict) -> dict:
    """직접 확인된 관측 항목만 채점하고 누락 총점은 None으로 반환한다.

    Args:
        heads: 학습 역할에서 선택한 네 항목 모델·규칙.
        row: 원본·구도·관측·직접 라벨이 연결된 오프라인 개발 입력.

    Returns:
        dict: AI·Rule 항목과 0~100 총점 또는 None. 직접 라벨 존재는 평가
        가능 여부에만 쓰며 등급 값은 예측에 사용하지 않는다. 실제 배포 시
        자동 구도 관측 판정의 별도 검증이 필요하다.
    """
    ai = []
    rule = []
    for i, n in enumerate(row["items"]):
        if (
            i not in heads
            or not n["eligible"]
            or row["levels"][i] is None
            or n["feature"] is None
        ):
            ai.append(None)
            rule.append(None)
            continue
        ai.append(
            float(np.clip(heads[i]["model"].predict([n["feature"]])[0], 0, 2))
        )
        rule.append(
            int(grades(np.asarray([n["feature"]]), heads[i]["policy"], i)[0])
        )
    complete = row.get("reference_total_allowed", True) and row.get(
        "whole_attempt_observed", True
    )
    total = lambda levels: (
        sum(w * g / 2 for w, g in zip([30, 25, 25, 20], levels))
        if complete and None not in levels
        else None
    )
    return dict(
        id=row["id"],
        person=row["person"],
        criterion_ids=["y_reach", "t_reach", "w_shape", "hinge_and_trunk_hold"],
        levels=row["levels"],
        ai_levels=ai,
        rule_levels=rule,
        ai_score=total(ai),
        rule_score=total(rule),
        target_score=total(row["levels"]),
    )


def run() -> dict:
    """투영 규칙 후보를 연결 그룹 제외와 기존 고정 개발 자료에서 비교한다.

    Returns:
        dict: 항목·총점 교집합과 오차·평가 범위·관측·단일 등급 경고.
        공통 규칙·기존 모델을 교체하지 않고 독립 최종 정확도로 부르지 않는다.

    Raises:
        ValueError: 입력·분할·기존 결과 지문·특징 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 후보·감사 자료 저장 실패.
    """
    DEST.mkdir(exist_ok=True)
    rows = [
        r
        for r in audit.read_json(OUT / "candidate_input_manifest.json")
        if r["exercise"] == "ytw"
    ]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    groups = recovery_train.component_groups(train)
    oof = []
    for fitting, held in GroupKFold(3).split(train, groups=groups):
        check_resources()
        heads = fit([train[int(i)] for i in fitting])
        oof.extend(predict(heads, train[int(i)]) for i in held)
    heads = fit(train)
    joblib.dump(heads, DEST / "model.joblib")
    fixed = [predict(heads, r) for r in val]
    report = dict(
        completed=True,
        training_attempts=len(train),
        training_metadata_groups=len({r["person"] for r in train}),
        training_heads=[
            dict(item=i, **{k: v for k, v in h.items() if k != "model"})
            for i, h in heads.items()
        ],
        connected_group_cv=recovery_train.summarize(oof),
        group_cv_predictions=oof,
        fixed_development=recovery_train.summarize(fixed),
        fixed_predictions=fixed,
        model_sha256=audit.digest(DEST / "model.joblib"),
        input_sha256=audit.digest(OUT / "candidate_input_manifest.json"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        manual_observability_intersection=True,
        automatic_deployment_view_gate_verified=False,
        anatomical_hinge_angle_measured=False,
        projection_only=True,
        expert_reference=False,
        independent_final_test=False,
        promoted=False,
        validation_used_for_numeric_fitting=False,
    )
    audit.save_json(DEST / "comparison.json", report)
    return dict(
        group_cv=report["connected_group_cv"], fixed=report["fixed_development"]
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
