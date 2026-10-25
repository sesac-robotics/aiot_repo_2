"""직접 W 등급 초안으로 별도 항목 AI를 학습하고 규칙과 비교한다.

입력은 관측된 화면 관절의 기하뿐이다. 규칙 등급·출처·참여자·직접 구간
시각은 예측 특징에 넣지 않는다. 직접 구간 조건부 성능과 자동 단계 성능을
구별하며, Y·전경이나 YTW 총점을 만들지 않는다.
"""

import collections
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import audit
import w_morphology
import w_component_trial
from resources import check_resources

NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("bend_deg", "hand_height_torso_ratio", "elbow_up_arm_ratio")
]
NAMES += ["bend_difference_deg"]


def features(sequence: np.ndarray, start: int, end: int) -> np.ndarray | None:
    """양팔 동시 관측이 충분한 구간의 기하 중앙값 7개를 반환한다.

    Args:
        sequence: N×33×2 픽셀 좌표, 미관측은 NaN.
        start: 시작 포함 표본 인덱스.
        end: 끝 포함 표본 인덱스.

    Returns:
        np.ndarray | None: 굽힘 2개·손 높이 비율 2개·팔꿈치 상승 비율
        2개·좌우 굽힘 차이(도). 3표본·관측률 80% 미달이면 None.
        규칙의 계산 등급은 특징에서 제외한다.

    Raises:
        ValueError: 좌표 또는 인덱스 오류.
    """
    measured = w_morphology.evaluate(sequence, start, end)
    if measured["grade"] is None:
        return None
    vector = [
        measured["arms"][s][k]
        for s in ("left", "right")
        for k in (
            "projected_bend_deg",
            "hand_above_elbow_torso_ratio",
            "elbow_up_upper_arm_ratio",
        )
    ]
    vector.append(measured["bilateral_bend_difference_deg"])
    return np.asarray(vector, dtype=float)


def fit(rows: list[dict]) -> ExtraTreesRegressor:
    """학습 폴드 안에서만 등급·참여자 빈도를 균형화하여 회귀기를 학습한다.

    Args:
        rows: x, target(0·1·2), person 키가 있는 직접 초안 목록.

    Returns:
        ExtraTreesRegressor: 연속 W 등급 예측 모델. 출력 단위는 0~2 등급.

    Raises:
        ValueError: 비어 있거나 등급·특징이 유효하지 않은 학습 목록.
    """
    if not rows or any(r["target"] not in (0, 1, 2) for r in rows):
        raise ValueError("Direct W grade fitting rows required")
    counts = collections.Counter((r["target"], r["person"]) for r in rows)
    groups = {
        g: len({p for k, p in counts if k == g})
        for g in {r["target"] for r in rows}
    }
    weights = [
        1 / (groups[r["target"]] * counts[(r["target"], r["person"])])
        for r in rows
    ]
    model = ExtraTreesRegressor(
        n_estimators=128, min_samples_leaf=2, random_state=210211, n_jobs=1
    )
    return model.fit(
        [r["x"] for r in rows],
        [r["target"] for r in rows],
        sample_weight=weights,
    )


def load_old_rows() -> list[dict]:
    """기존 직접 W 구간·등급과 실제 캐시를 연결한다.

    Returns:
        list[dict]: 세 표본 이상인 기존 개발 구간. 보류 라벨·불충분 구간을
        제외하며 직접 검토 이력은 그대로 유지한다. 새 검토로 세지 않는다.

    Raises:
        OSError: 초안·캐시 읽기 실패.
        ValueError: 같은 ID의 서로 다른 라벨·원본 계보 충돌.
    """
    rows = {}
    for stem in ["w_challenge", "w_challenge_v2", "w_targeted", "next_w"]:
        outputs = {
            r["id"]: r
            for r in audit.read_json(audit.OUT / f"{stem}_results.json")[
                "results"
            ]
        }
        for draft in audit.read_json(audit.OUT / f"{stem}_direct_review.json")[
            "reviews"
        ]:
            grade, core = draft["W_grade_draft"], draft["direct_W_core_frames"]
            if grade is None or not core:
                continue
            entry = outputs[draft["id"]]
            with np.load(entry["extraction"]["cache"]) as cache:
                seq, frames = cache["image_xy"].copy(), cache["frame_numbers"]
                if "observed" in cache:
                    seq[~cache["observed"].astype(bool)] = np.nan
            ids = np.flatnonzero((frames >= core[0]) & (frames <= core[1]))
            if len(ids) < 3:
                continue
            start, end = int(ids[0]), int(ids[-1])
            x = features(seq, start, end)
            if x is None:
                continue
            row = dict(
                id=draft["id"],
                person=draft["person"],
                source=draft["video_sha256"],
                target=grade,
                x=x,
                seq=seq,
                start=start,
                end=end,
                annotation_origin=str(audit.OUT / f"{stem}_direct_review.json"),
            )
            if row["id"] in rows:
                raise ValueError("Duplicate W draft ID")
            rows[row["id"]] = row
    return list(rows.values())


def metrics(rows: list[dict]) -> dict:
    """같은 직접 초안·입력 구간에서 방법별 등급 오차와 평가 수를 계산한다.

    Args:
        rows: target, ai, old_rule, component_rule가 있는 예측 목록.

    Returns:
        dict: 공통 평가 수와 방법별 평균 절대 등급 오차, 정확 일치 수.
        원본 그룹·독립성은 상위 보고서에 따로 기록한다.
    """
    result = dict(records=len(rows))
    for method in ["ai", "old_rule", "component_rule"]:
        pairs = [r for r in rows if r.get(method) is not None]
        result[method] = dict(
            records=len(pairs),
            MAE_grade=(
                float(np.mean([abs(r[method] - r["target"]) for r in pairs]))
                if pairs
                else None
            ),
            rounded_exact=sum(
                int(np.floor(r[method] + 0.5)) == r["target"] for r in pairs
            ),
        )
    return result


def run() -> dict:
    """기존 학습 초안의 참여자 제외 진단 후 새 구간을 한 번 비교한다.

    Returns:
        dict: 조건부 W 모델·규칙 비교와 새 영상의 자동 구간 겹침 진단.
        새 검토 구간·참여자를 해당 모델의 학습·모델 선택에서 제외한다.
        공통 모델·규칙·라벨을 교체하지 않는다.

    Raises:
        OSError: 입력·모델·결과 접근 실패.
        ValueError: 라벨·원본·참여자 중복 또는 캐시 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    new = audit.read_json(audit.OUT / "followup_W_direct.json")
    held_people = {
        r["person"] for r in new["reviews"] if r["W_grade_draft"] is not None
    }
    old = load_old_rows()
    training = [r for r in old if r["person"] not in held_people]
    development = []
    for person in sorted({r["person"] for r in training}):
        check_resources()
        testing = [r for r in training if r["person"] == person]
        sources = {r["source"] for r in testing}
        fitting = [
            r
            for r in training
            if r["person"] != person and r["source"] not in sources
        ]
        model = fit(fitting)
        for row in testing:
            development.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target=row["target"],
                    ai=float(model.predict([row["x"]])[0]),
                    old_rule=w_morphology.evaluate(
                        row["seq"], row["start"], row["end"]
                    )["grade"],
                    component_rule=w_component_trial.evaluate(
                        row["seq"], row["start"], row["end"]
                    )["grade"],
                )
            )
    model = fit(training)
    folder = audit.OUT / "followup_W_model"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        dict(
            model=model,
            names=NAMES,
            train_ids=[r["id"] for r in training],
            train_people=sorted({r["person"] for r in training}),
            output="W_visible_shape_grade_not_total_score",
            expert_validated=False,
        ),
        folder / "model.joblib",
    )
    extractions = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_extraction.json")[
            "W_records"
        ]
    }
    validation = []
    import ytw_phases

    for draft in new["reviews"]:
        if draft["W_grade_draft"] is None:
            continue
        extracted = extractions[draft["id"]]
        with np.load(extracted["cache"]) as cache:
            seq, frames = cache["image_xy"].copy(), cache["frame_numbers"]
        a, b = draft["direct_W_core_frames"]
        ids = np.flatnonzero((frames >= a) & (frames <= b))
        row = dict(
            id=draft["id"],
            person=draft["person"],
            target=draft["W_grade_draft"],
            ai=None,
            old_rule=None,
            component_rule=None,
            automatic=None,
        )
        if len(ids) >= 3:
            start, end = int(ids[0]), int(ids[-1])
            x = features(seq, start, end)
            if x is not None:
                row.update(
                    ai=float(model.predict([x])[0]),
                    old_rule=w_morphology.evaluate(seq, start, end)["grade"],
                    component_rule=w_component_trial.evaluate(seq, start, end)[
                        "grade"
                    ],
                )
        phase = ytw_phases.detect(seq)
        window = phase["phases"].get("W_attempt")
        row["phase_detection"] = phase
        row["same_attempt_core_overlaps"] = bool(
            window
            and frames[window["start_index"]] <= b
            and frames[window["end_index"]] >= a
        )
        if window:
            x = features(seq, window["start_index"], window["end_index"])
            if x is not None:
                row["automatic"] = dict(
                    ai=float(model.predict([x])[0]),
                    old_rule=w_morphology.evaluate(
                        seq, window["start_index"], window["end_index"]
                    )["grade"],
                    component_rule=w_component_trial.evaluate(
                        seq, window["start_index"], window["end_index"]
                    )["grade"],
                )
        validation.append(row)
    report = dict(
        training_records=len(training),
        training_people=len({r["person"] for r in training}),
        train_grade_counts=dict(
            collections.Counter(r["target"] for r in training)
        ),
        excluded_training_records_for_validation_people=len(old)
        - len(training),
        training_ids=[r["id"] for r in training],
        development_LOPO=development,
        development_metrics=metrics(development),
        new_development_validation=validation,
        new_conditional_metrics=metrics(validation),
        rule_diagnostic_designed_after_prior_errors=True,
        new_sources_used_for_fitting=False,
        independent_final_test=False,
        expert_validated=False,
        common_policy_modified=False,
        model_sha256=audit.digest(folder / "model.joblib"),
    )
    audit.save_json(audit.OUT / "followup_W_learning_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["new_conditional_metrics"])
