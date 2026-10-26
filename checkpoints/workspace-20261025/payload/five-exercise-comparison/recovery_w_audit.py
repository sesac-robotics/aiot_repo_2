"""자동 W 구간 오류와 W 모양 규칙 오류를 같은 기존 초안에서 분리한다."""

import json

import numpy as np

import audit
import w_morphology
import ytw_partial_trial
from recovery_prepare import ROOT


def run() -> dict:
    """직접 표시한 기존 구간을 진단 대조로만 사용해 W 실패 유형을 저장한다.

    Returns:
        dict: 자동 구간·직접 표시 구간의 W 규칙과 동일 직접 등급 초안.
        직접 구간 대조는 자동 인식 성능이나 독립 시험으로 합산하지 않는다.

    Raises:
        OSError: 기존 구간·라벨·관절 접근 실패.
        ValueError: 원본 해시 또는 프레임 연결 불일치.
    """
    plan = {
        r["id"]: r for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    cached = {
        r["id"]: r["extraction"]
        for r in audit.read_json(ROOT / "extraction.json")["records"]
    }
    manual = {
        r["id"]: dict(
            source_sha=r["video_sha256"], window=r["phase_windows"]["W_attempt"]
        )
        for r in audit.read_json(audit.OUT / "ytw_phase_annotations.json")[
            "records"
        ]
        if "W_attempt" in r["phase_windows"]
    }
    for row in audit.read_json(
        audit.BASE
        / "output/existing-data-audit-20261022/score_annotations.json"
    ):
        if row.get("stage_windows"):
            a, b = row["stage_windows"]["W"]
            manual[row["id"]] = dict(
                source_sha=row["video_sha256"],
                window=dict(start_frame=a, end_frame=b),
            )
    results = []
    for identifier, entry in manual.items():
        if identifier not in plan:
            continue
        row = plan[identifier]
        if row["video_sha256"] != entry["source_sha"]:
            raise ValueError("Phase source changed")
        with np.load(cached[identifier]["cache"]) as cache:
            seq, frames = cache["image_xy"], cache["frame_numbers"]
        phase = ytw_partial_trial.detect(seq)["phases"].get("W_attempt")
        auto = (
            w_morphology.evaluate(seq, phase["start_index"], phase["end_index"])
            if phase
            else dict(grade=None, hold_reason="no_automatic_W")
        )
        w = entry["window"]
        indices = np.flatnonzero(
            (frames >= w["start_frame"]) & (frames <= w["end_frame"])
        )
        direct = (
            w_morphology.evaluate(seq, int(indices[0]), int(indices[-1]))
            if len(indices) >= 3
            else dict(grade=None, hold_reason="direct_window_too_few_samples")
        )
        target = row["criterion_levels"][2]
        results.append(
            dict(
                id=identifier,
                person=row["person"],
                split=row["split"],
                direct_draft=target,
                manual_source_window=w,
                automatic=auto,
                direct_window_diagnostic=direct,
                diagnosis=(
                    "phase_selection_difference"
                    if auto["grade"] != target
                    and direct["grade"] == target
                    and target is not None
                    else (
                        "shape_rule_or_reference_disagreement"
                        if direct["grade"] is not None
                        and direct["grade"] != target
                        else (
                            "insufficient_observed_evidence"
                            if direct["grade"] is None
                            else "agreement"
                        )
                    )
                ),
            )
        )
    report = dict(
        records=results,
        manual_windows_used_as_model_input=False,
        direct_window_results_are_oracle_diagnostic_not_automatic_accuracy=True,
    )
    audit.save_json(ROOT / "W_failure_audit.json", report)
    return dict(
        cases=len(results),
        diagnoses={
            k: sum(r["diagnosis"] == k for r in results)
            for k in {r["diagnosis"] for r in results}
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
