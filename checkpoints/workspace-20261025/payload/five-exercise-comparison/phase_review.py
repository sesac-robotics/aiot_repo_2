"""직접 표시한 시도 구간과 자동 단계 인식을 개발 자료에서 대조한다.

직접 표시는 점수 정답이 아닌 표본 기반 단계 초안이다. 독립 시험이 아니다.
"""

import importlib.util
import numpy as np

import audit
import ytw_phases


def run() -> dict:
    """보존 규칙과 새 단계 인식을 같은 관절 입력으로 비교해 저장한다.

    Returns:
        dict: 직접 구간·자동 구간·대표점 일치·구간 중첩 진단 결과.
        프레임 경계 불확실성이 있어 정밀한 경계 정확도를 뜻하지 않는다.

    Raises:
        OSError: 직접 검토 JSON·관절 캐시·이전 규칙을 읽을 수 없는 경우.
        ValueError: 원본 해시 또는 입력 계약이 일치하지 않는 경우.
    """
    annotations = audit.read_json(audit.OUT / "ytw_phase_annotations.json")
    spec = importlib.util.spec_from_file_location(
        "preserved_rules", audit.OUT / "snapshots/pre_phase_v2/rule_score.py"
    )
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    targets = {
        row["id"]: row for row in audit.read_json(audit.OUT / "labels.json")
    }
    rows = []
    for item in annotations["records"]:
        path = audit.OUT / "heavy" / item["id"] / "primary.npz"
        if not path.exists():
            path = audit.OUT / "phase_candidates" / item["id"] / "primary.npz"
        if not path.exists():
            rows.append(
                dict(id=item["id"], status="no_eligible_cache_not_forced")
            )
            continue
        metadata = audit.read_json(path.with_suffix(".json"))
        if (
            metadata["config"]["source_sha256"] != item["video_sha256"]
            or metadata["config"]["segment"] != item["segment"]
        ):
            raise ValueError("Phase annotation source/segment mismatch")
        with np.load(path) as data:
            detected = ytw_phases.detect(data["image_xy"])
            indices = data["frame_numbers"]
            old = previous.evaluate(
                "ytw",
                data["image_xy"],
                None,
                targets[item["id"]]["criterion_ids"],
                targets[item["id"]]["criterion_weights"],
            )
        comparisons = []
        for name, phase in detected["phases"].items():
            actual = [
                int(indices[phase[key]])
                for key in ("start_index", "anchor_index", "end_index")
            ]
            band = item["phase_windows"].get(name)
            if band:
                low, high = band["start_frame"], band["end_frame"]
                intersection = max(
                    0, min(high, actual[2]) - max(low, actual[0]) + 1
                )
                union = max(high, actual[2]) - min(low, actual[0]) + 1
                comparisons.append(
                    dict(
                        phase=name,
                        detected_frames=actual,
                        direct_core_frames=[low, high],
                        anchor_inside_core=low <= actual[1] <= high,
                        anchor_within_uncertainty=low
                        - item.get("boundary_uncertainty_frames", 5)
                        <= actual[1]
                        <= high + item.get("boundary_uncertainty_frames", 5),
                        sample_based_interval_IoU=intersection / union,
                    )
                )
        old_frames = [
            int(indices[i]) for i in old["evidence"].get("phase_indices", [])
        ]
        old_matches = []
        for name, frame in zip(
            ("Y_attempt", "T_attempt", "W_attempt"), old_frames
        ):
            band = item["phase_windows"].get(name)
            if band:
                old_matches.append(
                    dict(
                        phase=name,
                        frame=frame,
                        anchor_inside_core=band["start_frame"]
                        <= frame
                        <= band["end_frame"],
                    )
                )
        rows.append(
            dict(
                previous_anchor_matches=old_matches,
                id=item["id"],
                role=item["evaluation_role"],
                person=item["person"],
                detected=detected,
                phase_comparisons=comparisons,
                previous_anchor_frames=[
                    int(indices[i])
                    for i in old["evidence"].get("phase_indices", [])
                ],
                previous_complete_sequence=bool(
                    old["evidence"].get("phase_indices")
                ),
            )
        )
    summaries = {}
    for role in ("design_training", "exposed_development_validation"):
        selected = [r for r in rows if r.get("role") == role]
        comparisons = [c for r in selected for c in r["phase_comparisons"]]
        summaries[role] = dict(
            records=len(selected),
            previous_complete=sum(
                r["previous_complete_sequence"] for r in selected
            ),
            new_complete=sum(
                r["detected"]["complete_attempt_sequence"] for r in selected
            ),
            previous_anchors_inside_core=sum(
                c["anchor_inside_core"]
                for r in selected
                for c in r.get("previous_anchor_matches", [])
            ),
            phase_anchors=len(comparisons),
            anchors_inside_direct_core=sum(
                c["anchor_inside_core"] for c in comparisons
            ),
            anchors_within_uncertainty=sum(
                c["anchor_within_uncertainty"] for c in comparisons
            ),
            mean_sample_interval_IoU=(
                float(
                    np.mean(
                        [c["sample_based_interval_IoU"] for c in comparisons]
                    )
                )
                if comparisons
                else None
            ),
        )
    report = dict(
        summary_by_role=summaries,
        rows=rows,
        independent_final_test=False,
        expert_validated=False,
        score_labels_modified=False,
        annotation_sha256=audit.digest(
            audit.OUT / "ytw_phase_annotations.json"
        ),
        detector_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/ytw_phases.py"
        ),
        scope_ko="이미 노출된 개발 자료. 단계 시도 인식의 회귀 진단이며 자세 점수 정확도 또는 새로운 독립 시험이 아님.",
    )
    audit.save_json(audit.OUT / "ytw_phase_comparison.json", report)
    return report


if __name__ == "__main__":
    print(run())
