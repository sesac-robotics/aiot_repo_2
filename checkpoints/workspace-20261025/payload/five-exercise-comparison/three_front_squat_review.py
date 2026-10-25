"""정면 스쿼트 후보의 몸통 항목을 직접 검토할 상세 장면을 준비한다."""

import audit
import external_review
import three_protocol

if __name__ == "__main__":
    root = three_protocol.OUT
    records = []
    for id, start, end in [
        ("g_U121IYD1Y", 73, 90),
        ("-K7cAxO4-pQ", 48, 60),
    ]:
        records.append(
            dict(
                id=id,
                sheet=external_review.contact_sheet(
                    root / f"sources/youtube_{id}.mp4",
                    root / f"review/front_squat_{id}_{start}_{end}.jpg",
                    start,
                    end,
                    48,
                ),
            )
        )
    audit.save_json(
        root / "front_squat_review_preparation.json",
        dict(records=records, directly_reviewed=False),
    )
