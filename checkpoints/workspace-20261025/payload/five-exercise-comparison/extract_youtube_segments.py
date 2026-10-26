"""분리한 YouTube 개발 구간에 Heavy를 적용해 항목별 관측을 확인한다."""

import numpy as np

import audit
import benchmark
from extract import extract_view
import rule_score


def run() -> dict:
    """같은 프레임·관절 계약으로 9개 구간의 관측과 규칙 미리보기를 저장한다.

    Returns:
        dict: 공통 관절 캐시·관측 보류 사유·규칙 미리보기. 기존 AI 모델의
        학습 자료나 독립 시험으로 자동 편입하지 않으며 변형은 채점하지 않는다.

    Raises:
        ValueError: 원본·구간·캐시 입력 계약 불일치.
        OSError: 원본·결과 접근 실패.
        RuntimeError: 자원 보호 또는 Heavy 추출 실패.
    """
    exports = audit.read_json(audit.OUT / "youtube_segment_exports.json")
    results = []
    for source in exports["records"]:
        exercise = "ytw" if source["id"].startswith("youtube_paul") else "plank"
        view = "primary" if exercise == "ytw" else "side"
        row = dict(
            id=source["id"],
            exercise=exercise,
            video_sha256=source["video_sha256"],
            required_views=[
                (
                    "side"
                    if view == "side"
                    else "single_phase_single_view_no_total"
                )
            ],
            segment=dict(
                start_frame=0, end_frame=source["frames"] - 1, fps=source["fps"]
            ),
        )
        extraction = extract_view(row, view, source["video_path"])
        seq, _, reasons = benchmark.load_input(row)
        preview = None
        if source["kind"] == "modified_kneeling_plank_not_error":
            reasons.append("different_exercise_variant_not_an_error_label")
        elif source["id"] == "youtube_mark_basic":
            reasons.append("oblique_elbow_sagittal_position_not_score_approved")
        elif exercise == "plank" and not reasons:
            preview = rule_score.evaluate(
                "plank",
                seq,
                None,
                [
                    "trunk_line",
                    "elbow_sagittal_position",
                    "leg_and_body_support",
                    "support_continuity",
                ],
                [40, 25, 20, 15],
            )
        if exercise == "ytw":
            reasons.append("single_phase_view_not_synchronized_complete_YTW")
        observed = np.isfinite(seq).all(2).mean(0)
        results.append(
            dict(
                id=row["id"],
                source_group=source["source_group"],
                person_group=source["person_group"],
                original_start_frame=source["start_frame"],
                original_end_frame_exclusive=source["end_frame_exclusive"],
                crop_xyxy=source["crop_xyxy"],
                extraction=extraction,
                per_joint_observed_fraction=observed.tolist(),
                hold_reasons=reasons,
                rule_preview=preview,
                AI_score=None,
                full_score_label=None,
                integrated_into_training=False,
                independent_final_test=False,
            )
        )
        audit.save_json(
            audit.OUT / "youtube_joint_results_partial.json",
            dict(results=results, complete=False),
        )
    report = dict(
        results=results,
        sources=3,
        segments=len(results),
        original_groups_kept_together=True,
        new_independent_test_videos=0,
        full_score_labels_added=0,
        common_AI_models_retrained=False,
    )
    audit.save_json(audit.OUT / "youtube_joint_results.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            {k: r[k] for k in ["id", "hold_reasons", "rule_preview"]}
            for r in run()["results"]
        ]
    )
