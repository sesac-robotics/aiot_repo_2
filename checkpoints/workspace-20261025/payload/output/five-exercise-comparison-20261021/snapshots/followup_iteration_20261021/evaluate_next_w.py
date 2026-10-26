"""새 직접 검토 후보에서 고정 W 규칙과 자동 단계 인식을 검증한다."""

import numpy as np

import audit
from extract import extract_view
import w_morphology
import ytw_phases


def run() -> dict:
    """3개 후보의 Heavy 관절·자동/직접 구간 W 판정을 별도로 저장한다.

    Returns:
        dict: 새 비전문가 초안과 고정 규칙의 개발 비교, 원본·코드 해시.
        수동 구간 결과는 자동 단계 성공률에 넣지 않으며 총점은 None이다.

    Raises:
        ValueError: 동결 코드·원본 해시 또는 영상 구간 오류.
        OSError: 원본·결과 접근 실패.
        RuntimeError: 자원 보호·관절 추출 실패.
    """
    direct = audit.read_json(audit.OUT / "next_w_direct_review.json")
    for field, filename in [
        ("rule_sha256", "w_morphology.py"),
        ("phase_sha256", "ytw_phases.py"),
    ]:
        if direct[field] != audit.digest(
            audit.BASE / "five-exercise-comparison" / filename
        ):
            raise ValueError("Frozen W rule or phase policy changed")
    rows = []
    for row in direct["reviews"]:
        item = dict(row, required_views=["W_projected_shape_development_only"])
        extraction = extract_view(
            item, "primary", row["video_path"], audit.OUT / "next_w_cache"
        )
        cache = np.load(extraction["cache"], allow_pickle=False)
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
        low, high = row["direct_W_core_frames"]
        selected = np.flatnonzero((frames >= low) & (frames <= high))
        diagnostic = (
            w_morphology.evaluate(seq, int(selected[0]), int(selected[-1]))
            if len(selected)
            else None
        )
        rows.append(
            dict(
                id=row["id"],
                draft_grade=row["W_grade_draft"],
                note_ko=row["note_ko"],
                automatic=auto,
                direct_window_diagnostic=diagnostic,
                automatic_W_source_frames=(
                    None
                    if not window
                    else [
                        int(frames[window["start_index"]]),
                        int(frames[window["end_index"]]),
                    ]
                ),
                direct_W_source_frames=[low, high],
                phases=phases,
                extraction=extraction,
                total_score=None,
            )
        )
    result = dict(
        results=rows,
        expert_validated=False,
        independent_final_test=False,
        labels_sha256=audit.digest(audit.OUT / "next_w_direct_review.json"),
        rule_sha256=direct["rule_sha256"],
        phase_sha256=direct["phase_sha256"],
        common_policy_modified=False,
        directly_graded=sum(r["draft_grade"] is not None for r in rows),
        total_labels_added=0,
    )
    audit.save_json(audit.OUT / "next_w_results.json", result)
    return result


if __name__ == "__main__":
    print(
        [
            {
                k: row[k]
                for k in [
                    "id",
                    "draft_grade",
                    "automatic_W_source_frames",
                    "direct_window_diagnostic",
                ]
            }
            for row in run()["results"]
        ]
    )
