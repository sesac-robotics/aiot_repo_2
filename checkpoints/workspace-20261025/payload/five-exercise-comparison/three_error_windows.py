"""원본 개요에서 직접 고른 오류 후보 구간을 상세 시간순 검토판으로 준비한다."""

import audit
import external_review
import three_protocol
from resources import check_resources


def run() -> dict:
    """직접 고른 네 시간대에서 각각 48장을 준비하고 점수는 만들지 않는다.

    Returns:
        dict: 원본·구간·표본 프레임과 검토판 경로. 직접 등급은 미포함.

    Raises:
        OSError: 원본·검토판 접근 실패.
        ValueError: 지정 구간·영상 디코딩 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    records = []
    for id, start, end in [
        ("DVGi7h0lL3A", 10, 17),
        ("DVGi7h0lL3A", 39, 51),
        ("DVGi7h0lL3A", 72, 87),
        ("-K7cAxO4-pQ", 124, 143),
    ]:
        check_resources()
        sheet = external_review.contact_sheet(
            root / ("sources/youtube_" + id + ".mp4"),
            root / "review" / f"error_window_{id}_{start}_{end}.jpg",
            start,
            end,
            48,
        )
        records.append(
            dict(
                id=id,
                start_seconds=start,
                end_seconds=end,
                sheet=sheet,
                score_generated=False,
            )
        )
    result = dict(records=records, directly_reviewed=False)
    audit.save_json(root / "error_window_preparation.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
