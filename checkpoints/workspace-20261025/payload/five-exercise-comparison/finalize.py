"""감사·개발 비교·웹 생성·파일 검사 순서로 완료된 결과를 갱신한다.

Git 저장소나 팀 서비스는 수정하지 않는다. 독립 최종 시험 완료를 뜻하지 않는다.
"""

import logging

import audit
import analysis_render
import w_diagnostics
import w_policy_sensitivity
import benchmark
import item_error_audit
import cache_parity
import render
import quality_diagnostics
import phase_review
import phase_render
import verify


def run() -> dict:
    """현재 라벨·캐시로 비교를 갱신하고 검사가 끝난 상태를 기록한다.

    Returns:
        dict: 검사 결과와 개발 모델 상태. 점수 단위는 프로젝트 배점이다.

    Raises:
        OSError: 필수 파일·영상·로컬 서버 접근에 실패한 경우.
        ValueError: 자료 분할·캐시 버전·원본 또는 형식 검사가 실패한 경우.
        RuntimeError: 영상 처리의 자원 보호 조건을 위반한 경우.
    """
    checked = audit.audit()
    if not checked["passed"]:
        raise ValueError("Structural audit failed")
    logging.info("Audit passed: %d records", checked["records"])
    phase_comparison = phase_review.run()
    comparison = benchmark.run()
    item_error_audit.run()
    logging.info("Development comparison saved")
    quality_diagnostics.run()
    render.render()
    phase_render.run()
    w_diagnostics.run()
    w_policy_sensitivity.run()
    analysis_render.run()
    logging.info("Dashboard and overlays saved")
    verified = verify.run()
    parity = cache_parity.run()
    progress = audit.read_json(audit.OUT / "progress.json")
    progress["work_in_progress"] = None
    progress["latest_completed_stage"] = (
        "W-morphology-v2-phase-hysteresis-v3-development"
    )
    progress["new_direct_reviews"].update(
        external_plank_temporal_records=4,
        total_score_drafts_added=16,
        external_candidates_direct_screened=17,
    )
    progress["latest_counts"] = dict(
        records=checked["records"],
        supplemental_records=checked["supplemental_records"],
        source_video_paths_checked=checked["source_video_paths_checked"],
        Heavy_view_caches=verified["view_caches"],
        development_models=sum(
            r["model_status"] == "unvalidated_development_model"
            for r in comparison["results"].values()
        ),
        direct_total_score_drafts_added=16,
        independent_final_test=0,
    )
    progress["completed"].extend(
        [
            "sequential_v2_migration_and_model_input_cache_checks_passed",
            "training_participant_holdout_diagnostic_executed",
            "feature_inclusive_11_repeat_median_timing_executed",
            f"{verified['unit_test_count']}_boundary_tests_passed",
            "all_generated_overlay_frames_decoded",
            "previous_and_current_cache_frame_comparison_executed",
        ]
    )
    progress["pending"] = [
        "W_morphology_rule_requires_more_grade_zero_cases_and_independent_calibration",
        "plank_error_cases_and_more_people_before_model_fit",
        "standing_shoulder_error_cases_and_more_people",
        "YTW_hinge_observation_and_phase_rule_validation",
        "squat_multiview_frame_synchronization_independent_recheck",
        "prospective_unexposed_person_source_group_final_test",
        "all_five_reliable_models_and_full_paired_test",
        "real_recapture_consistency_and_controlled_end_to_end_latency",
        "independent_label_review_and_expert_validation",
    ]
    progress["ytw_phase_summary"] = phase_comparison["summary_by_role"]
    progress["new_direct_reviews"]["ytw_phase_annotation_records"] = 7
    progress["new_direct_reviews"][
        "external_plank_new_usable_training_person"
    ] = "dvids_gysgt_gates"
    progress["new_external_auto_screen_candidates"] = 2
    progress["external_screen_delegate_check"] = str(
        audit.OUT / "screen_delegate_check.json"
    )
    progress["targeted_acquisition"] = audit.read_json(
        audit.OUT / "targeted_acquisition_review.json"
    )["counts"]
    progress["W_development_validation"] = dict(
        item_draft_labels_added=18,
        item_unknown_reviews=6,
        new_integrated_total_score_labels=1,
        candidate_identity_groups_suspected=3,
        identity_verified_new_people=None,
        first_set=audit.read_json(audit.OUT / "w_challenge_results.json")[
            "summary"
        ],
        second_set=audit.read_json(audit.OUT / "w_challenge_v2_results.json")[
            "summary"
        ],
        reused_training_negative=audit.read_json(
            audit.OUT / "w_prior_negative_results.json"
        )["automatic"]["grade"],
        independent_test=False,
        expert_validated=False,
    )
    progress["targeted_W_error_review"] = audit.read_json(
        audit.OUT / "w_targeted_results.json"
    )["summary"]
    progress["new_direct_reviews"]["plank_QEVD_view_screened_new"] = 36
    progress["new_direct_reviews"]["navy773312_direct_score_drafts"] = 2
    progress["new_direct_reviews"]["navy773312_error_score_drafts"] = 1
    progress["W_contract_fix"] = dict(
        prior_90_degree_proxy_retired_globally=True,
        score_labels_modified=False,
        historical_failure_cases_preserved=3,
        accuracy_improvement_claimed=False,
    )
    progress["latest_counts"]["unit_tests"] = verified["unit_test_count"]
    progress["latest_counts"]["overlays"] = len(verified["overlay_checks"])
    progress["cache_regression_view_pairs"] = parity["view_pairs"]
    audit.save_json(audit.OUT / "progress.json", progress)
    return verified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run())
