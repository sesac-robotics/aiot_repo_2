"""정면 플랭크의 보이는 상지 변경 사건을 총점과 분리하여 비교한다."""

import collections
import json
import pathlib

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

import audit
import upper_support_event_model as events
from allfive_round1 import OUT
from multiview_squat_batch2_train import mask_reviewed_frames
from plank_partial_item_trial import load_rows
from recovery_train import component_groups
from resources import check_resources


def counts(predictions: list[dict], method: str) -> dict:
    """이진 사건 예측의 혼동 수를 반환한다.

    Args:
        predictions: 직접 사건 라벨과 예측이 있는 목록.
        method: ai_event 또는 rule_event.

    Returns:
        dict: TP·FN·FP·TN. 정상 자세·총점 정확도로 환산하지 않는다.
    """
    return dict(
        TP=sum(r[method] and r["target"] == 1 for r in predictions),
        FN=sum(not r[method] and r["target"] == 1 for r in predictions),
        FP=sum(r[method] and r["target"] == 0 for r in predictions),
        TN=sum(not r[method] and r["target"] == 0 for r in predictions),
    )


def run() -> dict:
    """기존 사건과 새 QEVD 직접 표본을 연결 그룹 단위로 제외 검증한다.

    Returns:
        dict: 관측 가능 수·사건 혼동 수·새 모델 지문. 고정 사건 AI 경계
        0.75·규칙 변위 경계 0.20은 이번 검증 결과에 맞춰 바꾸지 않는다.

    Raises:
        ValueError: 관절 캐시·직접 검토·기존 사건 지문·분할 계약 불일치.
        OSError: 원본·모델·결과 읽기 또는 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    old_model_path = audit.OUT / "upper_support_event_trial/model.joblib"
    prior_sha = audit.digest(old_model_path)
    data = []
    unavailable = []
    for row in load_rows(True):
        if row["split"] != "train":
            continue
        target = (
            0
            if row["grades"][3] == 2
            else (
                1
                if row["id"]
                in (
                    "error_review_colossus_forward_core",
                    "support_pannell_stop",
                )
                else None
            )
        )
        if target is None:
            continue
        x, note = events.event_features(row["seq"])
        if x is not None:
            data.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    source=row["source"],
                    sources={row["source"]},
                    split="train",
                    x=x,
                    target=target,
                    observation=note,
                    new=False,
                )
            )
    direct_path = audit.OUT / "qevd_upper_event_direct.json"
    direct = audit.read_json(direct_path)
    extraction = audit.read_json(audit.OUT / "qevd_upper_event_extraction.json")
    if extraction["direct_sha256"] != audit.digest(direct_path):
        raise ValueError("Prior direct event labels changed")
    paths = {r["id"]: r["extraction"]["cache"] for r in extraction["records"]}
    for row in direct["reviews"]:
        with np.load(paths[row["id"]]) as cache:
            x, note = events.event_features(cache["image_xy"])
        if x is not None:
            data.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    source=row["video_sha256"],
                    sources={row["video_sha256"]},
                    split="train",
                    x=x,
                    target=row["visible_upper_support_change"],
                    observation=note,
                    new=False,
                )
            )
    labels = audit.read_json(OUT / "manual_decisions.json")["events"]
    plan = {
        r["source_id"]: r
        for r in audit.read_json(OUT / "extraction_plan.json")["records"]
    }
    receipts = {
        r["id"]: r["extraction"]
        for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    reviews = {
        r["id"]: r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    for row in labels:
        check_resources()
        p = plan[row["id"]]
        rec = receipts[p["id"]]
        review = reviews[p["id"]]
        if (
            review["fingerprint"] != rec["fingerprint"]
            or not review["direct_temporal_sample_review"]
        ):
            raise ValueError("Tracking review changed")
        with np.load(rec["cache"]) as cache:
            seq = mask_reviewed_frames(
                cache["image_xy"],
                cache["frame_numbers"],
                review.get("unreliable_frame_intervals", []),
            )
        x, note = events.event_features(seq)
        if x is None or not review["event_tracking_permitted"]:
            unavailable.append(
                dict(
                    id=row["id"],
                    reason="required_upper_arm_not_observed_or_direct_tracking_not_permitted",
                    observation=note,
                )
            )
            continue
        data.append(
            dict(
                id=p["id"],
                person=p["person"],
                source=p["video_sha256"],
                sources={p["video_sha256"]},
                split="train",
                x=x,
                target=row["visible_upper_support_interruption"],
                observation=note,
                new=True,
            )
        )
    groups = component_groups(data)
    predictions = []
    for train, held in GroupKFold(min(3, len(set(groups)))).split(
        data, groups=groups
    ):
        model = events.fit([data[int(i)] for i in train])
        classes = list(model.classes_)
        for i in held:
            row = data[int(i)]
            prob = (
                float(model.predict_proba([row["x"]])[0][classes.index(1)])
                if 1 in classes
                else 0.0
            )
            predictions.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    target=row["target"],
                    new=row["new"],
                    ai_probability=prob,
                    ai_event=prob >= 0.75,
                    rule_event=row["observation"]["relative_displacement"]
                    >= 0.20,
                )
            )
    model = events.fit(data)
    joblib.dump(
        dict(
            model=model,
            threshold=0.75,
            output="visible_upper_support_change_probability_not_posture_score",
        ),
        OUT / "upper_event_candidate.joblib",
    )
    if audit.digest(old_model_path) != prior_sha:
        raise ValueError("Existing event model changed")
    report = dict(
        training_records=len(data),
        new_eligible_events=sum(r["new"] for r in data),
        new_event_targets=dict(
            collections.Counter(str(r["target"]) for r in data if r["new"])
        ),
        metadata_groups=len(set(groups)),
        unavailable=unavailable,
        group_cv_predictions=predictions,
        group_cv_metrics={
            m: counts(predictions, m) for m in ("ai_event", "rule_event")
        },
        same_new_group_cv_metrics={
            m: counts([p for p in predictions if p["new"]], m)
            for m in ("ai_event", "rule_event")
        },
        model_sha256=audit.digest(OUT / "upper_event_candidate.joblib"),
        preserved_model_sha256=prior_sha,
        absence_does_not_mean_normal_grade=True,
        posture_total=None,
        independent_final_test=False,
        expert_reference=False,
        code_sha256=audit.digest(pathlib.Path(__file__)),
        direct_sha256=audit.digest(OUT / "manual_decisions.json"),
    )
    audit.save_json(OUT / "upper_event_comparison.json", report)
    return dict(
        training_records=len(data),
        new_eligible_events=report["new_eligible_events"],
        metrics=report["group_cv_metrics"],
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
