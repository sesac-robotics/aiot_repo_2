"""직접 표시한 QEVD 상지 변경 사건의 관절을 원본 구간에서 추출한다."""

import audit
from extract import extract_view


def run() -> dict:
    """세 새 참여자 그룹의 직접 사건 초안과 Heavy 캐시를 연결한다.

    Returns:
        dict: 추출 계보와 초안 해시. 총점·하체 라벨을 생성하지 않는다.

    Raises:
        OSError: 원본·초안·캐시 접근 실패.
        ValueError: 원본·초안 해시 불일치.
        RuntimeError: GPU 추출·자원 보호 실패.
    """
    path = audit.OUT / "qevd_upper_event_direct.json"
    frozen = audit.digest(path)
    results = []
    for draft in audit.read_json(path)["reviews"]:
        row = dict(
            draft, exercise="plank", required_views=["upper_support_event_only"]
        )
        extraction = extract_view(
            row, "primary", row["video_path"], audit.OUT / "upper_support_cache"
        )
        results.append(dict(id=row["id"], extraction=extraction))
    if frozen != audit.digest(path):
        raise ValueError("Direct event drafts changed")
    report = dict(
        records=results,
        direct_sha256=frozen,
        full_score_training_records_added=0,
    )
    audit.save_json(audit.OUT / "qevd_upper_event_extraction.json", report)
    return report


if __name__ == "__main__":
    print(run())
