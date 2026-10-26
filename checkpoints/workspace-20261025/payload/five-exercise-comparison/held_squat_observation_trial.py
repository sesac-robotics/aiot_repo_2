"""부분 복귀의 발 관측을 복귀 점수와 분리한 별도 스쿼트 개발 실험이다."""

import json

import joblib
from sklearn.model_selection import GroupKFold

import audit
import recovery_train
from allfive_round1_features import load
from allfive_round1_report import paired_summary
from allfive_round1_train import manifest, predict
from held_feature_update import fit, update_existing
from held_round2 import OUT as ROUND2
from held_round2_train import added_rows
from resources import check_resources

OUT = ROUND2.parent / "squat_partial_top_trial"


def run() -> dict:
    """기존 승인 라벨·분할을 유지하며 발 특징 관측 조건만 바꿔 비교한다.

    Returns:
        dict: 공통 검증 총점·항목 오차, 평가 가능 수, 연결 그룹 교차검증.
        철회한 뒤꿈치 오류 가설 4구간은 입력하지 않는다. 라벨 값은 바꾸지
        않으며 부분 복귀를 정상 복귀·완결 반복으로 바꾸지 않는다.

    Raises:
        ValueError: 원본·참여자 분할 또는 승인 캐시 계약 위반.
        OSError: 입력·모델·개발 비교 결과 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    before, _, _ = load()
    old = [r for r in before if r["exercise"] == "squats"]
    rows, changes = update_existing(old, allow_partial_top=True)
    rows += [r for r in added_rows() if r["exercise"] == "squats"]
    recovery_train.component_groups(rows)
    previous_inputs = {
        r["id"]: r
        for r in audit.read_json(ROUND2 / "candidate_input_manifest.json")
    }
    changed = [
        dict(
            id=r["id"],
            split=r["split"],
            levels=r["levels"],
            old_eligible=previous_inputs[r["id"]]["items"][2]["eligible"],
            new_eligible=r["notes"][2]["eligible"],
            holds=r["notes"][2].get("holds", []),
        )
        for r in rows
        if r["notes"][2]["eligible"]
        != previous_inputs[r["id"]]["items"][2]["eligible"]
    ]
    audit.save_json(
        OUT / "observation_changes.json",
        dict(
            records=changed,
            feature_changes=changes,
            direct_labels_changed=False,
            rejected_heel_hypothesis_inputs_used=False,
        ),
    )
    audit.save_json(OUT / "candidate_input_manifest.json", manifest(rows))
    fitting = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    check_resources()
    model = fit(fitting)
    joblib.dump(model, OUT / "squats_candidate.joblib")
    fixed = [predict(model, r) for r in validation]
    previous = next(
        r
        for r in audit.read_json(ROUND2 / "comparison.json")["exercises"]
        if r["exercise"] == "squats"
    )
    by_id = {r["id"]: r for r in previous["fixed_predictions"]}
    paired = paired_summary([dict(r, baseline=by_id[r["id"]]) for r in fixed])
    oof = []
    for fitting_ids, held in GroupKFold(3).split(
        fitting, groups=recovery_train.component_groups(fitting)
    ):
        check_resources()
        fold = fit([fitting[int(i)] for i in fitting_ids])
        oof.extend(predict(fold, fitting[int(i)]) for i in held)
    result = dict(
        completed=True,
        paired=paired,
        fixed=recovery_train.summarize(fixed),
        fixed_predictions=fixed,
        group_cv=recovery_train.summarize(oof),
        group_cv_predictions=oof,
        observation_changes=changed,
        independent_final_test=False,
        labels_changed=False,
        rejected_heel_hypotheses_used=False,
        promoted=False,
        expert_reference=False,
        model_sha256=audit.digest(OUT / "squats_candidate.joblib"),
    )
    audit.save_json(OUT / "comparison.json", result)
    return dict(fixed=result["fixed"], paired=paired, changes=changed)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
