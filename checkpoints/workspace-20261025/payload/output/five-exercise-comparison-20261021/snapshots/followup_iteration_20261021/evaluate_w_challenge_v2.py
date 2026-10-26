"""고정 W 모양 규칙을 새 직접 표본 초안과 대조한다. AI 정답을 만들지 않는다."""

import numpy as np

import audit
from extract import extract_view
import w_morphology
import ytw_phases


def run() -> dict:
    """12개 새 개발 후보의 관절·단계·W 모양 결과를 별도로 저장한다.

    Returns:
        dict: 실제 자동 구간의 등급 비교·평가 가능률·오차와 직접 구간을
        사용한 진단을 구분한 결과. 독립 최종 시험·전문가 정확도가 아니다.

    Raises:
        ValueError: 원본·캐시·구간 불일치.
        RuntimeError: 관절 추출·자원 보호 실패.
        OSError: 원본·캐시·검토 라벨·결과 접근 실패.
    """
    direct = audit.read_json(audit.OUT / "w_challenge_v2_direct_review.json")
    if direct["policy_sha256"] != audit.digest(
        audit.BASE / "five-exercise-comparison/w_morphology.py"
    ):
        raise ValueError("Frozen v2 policy changed")
    results = []
    for row in direct["reviews"]:
        prepared = dict(
            id=row["id"],
            video_sha256=row["video_sha256"],
            segment=row["segment"],
            required_views=["W_projected_shape_challenge_only"],
        )
        extracted = extract_view(
            prepared,
            "primary",
            row["video_path"],
            audit.OUT / "w_challenge_v2_cache",
        )
        with np.load(extracted["cache"]) as cache:
            seq, frames = cache["image_xy"], cache["frame_numbers"]
        phases = ytw_phases.detect(seq)
        window = phases["phases"].get("W_attempt")
        auto = (
            w_morphology.evaluate(
                seq, window["start_index"], window["end_index"]
            )
            if window
            else dict(
                grade=None, hold_reason="automatic_W_attempt_not_detected"
            )
        )
        manual = None
        core = row["direct_W_core_frames"]
        if core:
            indices = np.flatnonzero((frames >= core[0]) & (frames <= core[1]))
            if len(indices):
                manual = w_morphology.evaluate(
                    seq, int(indices[0]), int(indices[-1])
                )
        overlap = bool(
            core
            and window
            and frames[window["start_index"]] <= core[1]
            and frames[window["end_index"]] >= core[0]
        )
        results.append(
            dict(
                id=row["id"],
                W_grade_draft=row["W_grade_draft"],
                same_attempt_core_overlaps=overlap,
                automatic=auto,
                manual_window_diagnostic=manual,
                phase_detection=phases,
                extraction=extracted,
                source_sha256=row["video_sha256"],
                source_split="train",
                evaluation_role=row["evaluation_role"],
                expert_validated=False,
            )
        )
        audit.save_json(
            audit.OUT / "w_challenge_v2_results_partial.json",
            dict(results=results, complete=False),
        )
    known = [r for r in results if r["W_grade_draft"] is not None]
    paired = [r for r in known if r["automatic"]["grade"] is not None]
    errors = [abs(r["automatic"]["grade"] - r["W_grade_draft"]) for r in paired]
    aligned = [r for r in paired if r["same_attempt_core_overlaps"]]
    aligned_errors = [
        abs(r["automatic"]["grade"] - r["W_grade_draft"]) for r in aligned
    ]
    report = dict(
        policy_version=w_morphology.VERSION,
        policy_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/w_morphology.py"
        ),
        labeling_policy_sha256=direct["policy_sha256"],
        evaluation_exposure="frozen_v2_before_dense_item_review_and_Heavy_results_prior_sparse_view_exposed",
        results=results,
        summary=dict(
            same_attempt_pairs=len(aligned),
            same_attempt_MAE_grade=(
                float(np.mean(aligned_errors)) if aligned_errors else None
            ),
            same_attempt_exact_matches=sum(e == 0 for e in aligned_errors),
            candidates=len(results),
            directly_graded=len(known),
            human_unknown=len(results) - len(known),
            automatic_assessed_known=len(paired),
            known_assessment_fraction=(
                len(paired) / len(known) if known else None
            ),
            MAE_grade=float(np.mean(errors)) if errors else None,
            exact_matches=sum(e == 0 for e in errors),
            automatic_grade_on_human_unknown=sum(
                r["W_grade_draft"] is None
                and r["automatic"]["grade"] is not None
                for r in results
            ),
            direct_grade_distribution={
                str(i): sum(r["W_grade_draft"] == i for r in results)
                for i in (0, 1, 2)
            },
        ),
        phase_version=ytw_phases.VERSION,
        phase_code_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/ytw_phases.py"
        ),
        phase_evaluation_exposure="reused_development_after_phase_v2_failure_review",
        independent_final_test=False,
        score_labels_modified=False,
        model_training_uses_rule_targets=False,
    )
    audit.save_json(audit.OUT / "w_challenge_v2_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["summary"])
