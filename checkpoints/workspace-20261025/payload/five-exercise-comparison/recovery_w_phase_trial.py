"""W 복귀 오선택 보완의 기하 규칙과 별도 AI 항목 회귀를 비교한다."""

import collections
import json
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import audit
import recovery_w_phase
import w_morphology
from recovery_prepare import ROOT
from recovery_train import component_groups
from resources import check_resources


def run() -> dict:
    """현재 평가 가능 W 표본에서 시점 보완·동일 방식 AI 재학습을 비교한다.

    Returns:
        dict: 시점 이동 수·직접 초안 대비 규칙 오차·개발 AI 항목 오차.
        개발 실패 사례를 보고 설계한 진단 실험이며 새 독립 시험이 아니다.
        공통 모델·규칙·기존 라벨을 수정하거나 총점을 생성하지 않는다.

    Raises:
        OSError: 캐시·학습 기록·출력 접근 실패.
        ValueError: 원본 구간·W 관측·분할 계약 위반.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    comparison = audit.read_json(ROOT / "comparison.json")
    inputs = audit.read_json(ROOT / "fit_input_manifest.json")
    plan = {
        r["id"]: r for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    caches = {
        r["id"]: r["extraction"]
        for r in audit.read_json(ROOT / "extraction.json")["records"]
    }
    fitted = next(r for r in comparison["exercises"] if r["exercise"] == "ytw")
    old_fixed = {r["id"]: r for r in fitted["fixed_predictions"]}
    ready, excluded = [], []
    for row in inputs:
        if (
            row["exercise"] != "ytw"
            or row["levels"][2] is None
            or not row["items"][2]["eligible"]
            or row["items"][2]["feature"] is None
        ):
            continue
        check_resources()
        source, receipt = plan[row["id"]], caches[row["id"]]
        if (
            receipt["config"]["source_sha256"] != source["video_sha256"]
            or receipt["config"]["segment"] != source["segment"]
        ):
            raise ValueError("W phase cache contract differs")
        with np.load(receipt["cache"]) as data:
            seq, frames = data["image_xy"].copy(), data["frame_numbers"].copy()
        # 현재 AI 입력에 적용한 관측 경계를 그대로 복원한다. NPZ의 픽셀 크기
        # 대신 저장된 이전 7개 특징과의 비교로 원본 W 경로를 대조한다.
        import cv2
        import three_items

        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(
            source["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        width, height = cap.get(3), cap.get(4)
        cap.release()
        seq = three_items.mask_image_border(seq, width, height)
        result = recovery_w_phase.detect(seq)
        window = result["phases"].get("W_attempt")
        if not window:
            raise ValueError("Previously observed W phase disappeared")
        measured = w_morphology.evaluate(
            seq, window["start_index"], window["end_index"]
        )
        if measured["grade"] is None:
            excluded.append(row["id"])
            continue
        feature = [
            measured["arms"][s][k]
            for s in ("left", "right")
            for k in (
                "projected_bend_deg",
                "hand_above_elbow_torso_ratio",
                "elbow_up_upper_arm_ratio",
            )
        ] + [measured["bilateral_bend_difference_deg"]]
        old_window = result.get("original_W_attempt", window)
        old_result = w_morphology.evaluate(
            seq, old_window["start_index"], old_window["end_index"]
        )
        old_feature = [
            old_result["arms"][s][k]
            for s in ("left", "right")
            for k in (
                "projected_bend_deg",
                "hand_above_elbow_torso_ratio",
                "elbow_up_upper_arm_ratio",
            )
        ] + [old_result["bilateral_bend_difference_deg"]]
        if not np.allclose(old_feature, row["items"][2]["feature"]):
            raise ValueError("Original W features differ from fitted inputs")
        ready.append(
            dict(
                id=row["id"],
                person=row["person_metadata_group"],
                sources=set(row["sources"]),
                recording_groups=row["recording_groups"],
                split=row["split"],
                level=row["levels"][2],
                feature=feature,
                old_rule=old_result["grade"],
                new_rule=measured["grade"],
                refined=result["w_refined"],
                before_frames=[
                    int(frames[old_window[k]])
                    for k in ("start_index", "end_index")
                ],
                after_frames=[
                    int(frames[window[k]]) for k in ("start_index", "end_index")
                ],
            )
        )
    component_groups(ready)
    train = [r for r in ready if r["split"] == "train"]
    val = [r for r in ready if r["split"] == "validation"]
    counts = collections.Counter((r["level"], r["person"]) for r in train)
    people = collections.Counter(g for g, p in counts)
    weights = [
        1 / (people[r["level"]] * counts[(r["level"], r["person"])])
        for r in train
    ]
    model = ExtraTreesRegressor(
        n_estimators=128, min_samples_leaf=2, random_state=221022, n_jobs=1
    )
    model.fit(
        [r["feature"] for r in train],
        [r["level"] for r in train],
        sample_weight=weights,
    )
    predictions = model.predict([r["feature"] for r in val]) if val else []

    def error(rows, key):
        return (
            float(np.mean([abs(r[key] - r["level"]) for r in rows]))
            if rows
            else None
        )

    report = dict(
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        phase_code_sha256=audit.digest(pathlib.Path(recovery_w_phase.__file__)),
        fit_input_manifest_sha256=audit.digest(
            ROOT / "fit_input_manifest.json"
        ),
        training_records=len(train),
        fixed_development_records=len(val),
        refined_training_records=sum(r["refined"] for r in train),
        refined_development_records=sum(r["refined"] for r in val),
        training_rule_before_MAE_grade=error(train, "old_rule"),
        training_rule_after_MAE_grade=error(train, "new_rule"),
        fixed_rule_before_MAE_grade=error(val, "old_rule"),
        fixed_rule_after_MAE_grade=error(val, "new_rule"),
        fixed_ai_before_MAE_grade=(
            float(
                np.mean(
                    [
                        abs(old_fixed[r["id"]]["ai_levels"][2] - r["level"])
                        for r in val
                    ]
                )
            )
            if val
            else None
        ),
        fixed_ai_after_MAE_grade=(
            float(
                np.mean(
                    [
                        abs(float(p) - r["level"])
                        for r, p in zip(val, predictions)
                    ]
                )
            )
            if val
            else None
        ),
        development_failure_cases_informed_phase_design=True,
        independent_final_test=False,
        expert_ground_truth=False,
        common_model_promoted=False,
        excluded=excluded,
        records=[
            {k: v for k, v in r.items() if k not in ("sources",)} for r in ready
        ],
    )
    joblib.dump(model, ROOT / "W_phase_trial_model.joblib")
    audit.save_json(ROOT / "W_phase_trial.json", report)
    return {k: v for k, v in report.items() if k != "records"}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
