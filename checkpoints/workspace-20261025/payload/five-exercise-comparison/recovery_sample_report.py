"""저장된 학습 기록으로 후보의 실제 입력 수와 항목별 부족을 집계한다."""

import json

import audit
from recovery_prepare import ROOT


def run() -> dict:
    """학습 결과에 기록된 건수를 70개 목표와 대조한다.

    Returns:
        dict: 운동별 실제 입력·참여자·항목별 건수와 미학습 추가 반복.
        원본 후보·프레임·검증 자료는 학습 수에 더하지 않는다.

    Raises:
        OSError: 저장된 비교 결과·추가 주석의 읽기 또는 저장 실패.
        KeyError: 학습 기록에 항목별 건수가 없는 경우.
    """
    source = ROOT / "comparison.json"
    fitted = audit.read_json(source)
    inputs = audit.read_json(ROOT / "fit_input_manifest.json")
    grade_groups = {}
    for row in inputs:
        if row["split"] != "train":
            continue
        for level, item in zip(row["levels"], row["items"]):
            if level is None or not item["eligible"] or item["feature"] is None:
                continue
            key = (row["exercise"], item["item"], item["bank"])
            grade_groups.setdefault(key, {}).setdefault(str(level), set()).add(
                row["person_metadata_group"]
            )
    rows = []
    for exercise in fitted["exercises"]:
        heads = exercise["heads"]
        rows.append(
            dict(
                exercise=exercise["exercise"],
                fitted_effective_input_segments=exercise[
                    "training_effective_segments"
                ],
                fitted_people=exercise["training_people"],
                fitted_metadata_groups=exercise["training_people"],
                fitted_connected_groups=exercise.get(
                    "training_connected_groups"
                ),
                metadata_groups_are_verified_people=False,
                fitted_four_item_observed_segments=exercise.get(
                    "training_four_item_observed_segments"
                ),
                fitted_source_videos=exercise.get(
                    "training_effective_source_videos"
                ),
                minimum_additional_effective_segments_to_70=max(
                    0, 70 - exercise["training_effective_segments"]
                ),
                minimum_additional_four_item_segments_to_70=max(
                    0,
                    70 - exercise["training_four_item_observed_segments"],
                ),
                item_view_counts=[
                    dict(
                        item=h["key"][1],
                        view=h["key"][2],
                        records=h["records"],
                        people=h["people"],
                        grade_counts=h["grade_counts"],
                        grade_metadata_groups={
                            g: len(
                                grade_groups.get(tuple(h["key"]), {}).get(
                                    g, set()
                                )
                            )
                            for g in ("0", "1", "2")
                        },
                        single_grade_model=h["single_grade_model"],
                    )
                    for h in heads
                ],
                all_item_view_heads_have_70=all(
                    h["records"] >= 70 for h in heads
                ),
                completed=False,
            )
        )
    reviewed = [
        r
        for path in ROOT.glob("full_repetition_score_annotations_*.json")
        for r in audit.read_json(path)
    ]
    planned = {
        r["id"] for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    fit_matches_plan = fitted.get("input_plan_sha256") == audit.digest(
        ROOT / "input_plan.json"
    )
    variant_path = ROOT / "variant_policy_exclusions.json"
    variant_excluded = (
        {r["id"] for r in audit.read_json(variant_path)["records"]}
        if variant_path.exists()
        else set()
    )
    pending = [
        r
        for r in reviewed
        if r["id"] not in variant_excluded
        and (r["id"] not in planned or not fit_matches_plan)
    ]
    report = dict(
        comparison_sha256=audit.digest(source),
        exercises=rows,
        target_interpretation_ko="70개는 최소 개발 목표이며 정확도 보장 기준이 아니다. 운동별 유효 입력 70개와 각 채점 항목별 70개는 다르다. 동작 구간·참여자·원본·구도별로 별도 집계한다.",
        pending_directly_reviewed_repetitions=len(pending),
        pending_full_score_drafts=sum(
            r["eligible_for_total_score_training"] for r in pending
        ),
        additional_directly_reviewed_repetitions=len(reviewed),
        excluded_variant_direct_repetitions=len(variant_excluded),
        fit_matches_current_input_plan=fit_matches_plan,
        pending_reason=(
            "추출·학습 계획에 아직 연결되지 않은 새 직접 초안"
            if pending
            else None
        ),
        ac_required=False,
        data_exhausted=False,
        local_remaining_ko="MultiView의 미라벨 반복과 REHAB24의 미라벨 반복 후보가 남아 있다. 후보는 직접 점수 검토와 관측 검증을 통과해야 학습 수로 셀 수 있다.",
        expert_ground_truth=False,
        independent_final_test_complete=False,
    )
    static_path = audit.OUT / "plank_static_review/comparison.json"
    if static_path.exists():
        static = audit.read_json(static_path)
        report["preserved_auxiliary_plank_static_experiment"] = {
            k: static[k]
            for k in [
                "source_images",
                "directly_drafted_trunk_labels",
                "observable_images",
                "common_evaluated_images",
                "metrics",
            ]
        }
        report["preserved_auxiliary_plank_static_experiment"][
            "not_added_to_video_training_count"
        ] = True
    audit.save_json(ROOT / "sample_count_review.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
