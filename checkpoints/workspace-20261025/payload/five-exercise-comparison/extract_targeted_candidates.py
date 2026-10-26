"""직접 구도 선별한 플랭크 후보를 점수 학습과 분리해 Heavy로 추출한다."""

import audit
from extract import extract_view


def run() -> dict:
    """WRAIR 사선 후보 한 구간의 관절 관측률을 저장한다.

    Returns:
        dict: 3초 후보 캐시 위치·관측률. 점수·라벨을 생성하지 않는다.

    Raises:
        ValueError: 원본 해시·영상 구간·추출 입력 계약 불일치.
        RuntimeError: 관절 추출·자원 보호 실패.
        OSError: 원본·캐시·출력 접근 실패.
    """
    reviewed = audit.read_json(audit.OUT / "review/dvids_805453_hold.json")
    row = dict(
        id="dvids_805453_candidate_1129_1201",
        required_views=["side_oblique_candidate_not_score_approved"],
        video_sha256=reviewed["video_sha256"],
        segment=dict(start_frame=1129, end_frame=1201, fps=reviewed["fps"]),
    )
    result = extract_view(
        row,
        "primary",
        reviewed["video_path"],
        audit.OUT / "targeted_candidates",
    )
    result.update(
        purpose="view_and_joint_feasibility_only",
        score_training_eligible=False,
        expert_validated=False,
        source_group="dvids_805453_all_performers_same_video",
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "targeted_candidate_extraction.json", result)
    return result


if __name__ == "__main__":
    print(run())
