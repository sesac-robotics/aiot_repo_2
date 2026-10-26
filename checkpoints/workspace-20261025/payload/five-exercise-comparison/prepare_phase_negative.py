"""총점 보류 영상의 관절을 단계 누락 음성 사례 검사 전용으로 추출한다.

점수 모델 학습·검증의 적격 자료 수에는 포함하지 않는다.
"""

import audit
from extract import extract_view


def run() -> dict:
    """직접 검토한 단계 불명확 자료를 별도 폴더에 추출한다.

    Returns:
        dict: 별도 캐시 정보. 관절 단위는 픽셀·미터 추정치이며 점수 없음.

    Raises:
        ValueError: 원본 해시·영상 구간·추출 입력 계약 오류.
        RuntimeError: 관절 추출 또는 자원 보호 검사 실패.
        OSError: 원본·캐시 파일 접근 실패.
    """
    rows = audit.read_json(audit.OUT / "development_manifest.json")
    row = next(row for row in rows if row["id"] == "qevd_00105254")
    result = extract_view(
        row, "primary", row["video_path"], audit.OUT / "phase_candidates"
    )
    result.update(
        purpose="phase_missing_negative_case_only",
        score_training_eligible=False,
    )
    audit.save_json(audit.OUT / "phase_negative_extraction.json", result)
    return result


if __name__ == "__main__":
    print(run())
