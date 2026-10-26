"""직접 검토 초안과 분리하여 FAO 후보의 공통 Heavy 입력을 추출한다."""

import audit
from extract import extract_view


def run() -> dict:
    """FAO 측면 3초 관절을 공통 입력 설정으로 추출한다.

    Returns:
        dict: 캐시 경로·관측률. 점수는 계산하지 않는다.

    Raises:
        ValueError: 원본·캐시·구간 계약 불일치.
        OSError: 입력·캐시 저장 실패.
        RuntimeError: 자원 보호·관절 추출 실패.
    """
    checked = audit.audit()
    if not checked["passed"]:
        raise ValueError("Audit failed")
    row = next(
        x
        for x in audit.read_json(audit.OUT / "development_manifest.json")
        if x["id"] == "dvids_fao947417_background_5430_5520"
    )
    result = extract_view(row, "side", row["video_path"], audit.OUT / "heavy")
    audit.save_json(audit.OUT / "fao_common_extraction.json", result)
    return result


if __name__ == "__main__":
    print(run())
