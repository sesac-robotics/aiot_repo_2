"""플랭크의 기존 보유 측면 원본에서 겹치지 않는 후속 검토 창을 준비한다."""

import json
import pathlib

import audit
import external_review
from allfive_round1 import OUT


def run() -> dict:
    """이미 학습 역할인 보유 해군 원본의 두 창과 수정 경계를 준비한다.

    Returns:
        dict: 시간순 자동 표본 수. 같은 원본·참여자의 추가 구간이므로
        독립 자료로 세지 않으며 직접 등급은 생성하지 않는다.

    Raises:
        OSError: 원본·근거 저장 실패.
        ValueError: 원본 시간 구간·분할 계약 불일치.
    """
    original = audit.read_json(audit.OUT / "navy773312_direct_drafts.json")[
        "labels"
    ][0]
    if original["split"] != "train":
        raise ValueError("Cannot review reserved source")
    records = []
    for first, last in ((3100, 3200), (3800, 3900)):
        identifier = f"held_navy773312_{first}_{last}"
        receipt = external_review.contact_sheet(
            pathlib.Path(original["video_path"]),
            OUT / "review" / (identifier + ".jpg"),
            first / original["segment"]["fps"],
            last / original["segment"]["fps"],
            24,
        )
        records.append(
            dict(
                id=identifier,
                exercise="plank",
                path=original["video_path"],
                person=original["person"],
                split="train",
                original=original,
                preparation=receipt,
            )
        )
    audit.save_json(
        OUT / "extra_preparation.json",
        dict(records=records, direct_labels_created=0),
    )
    row = next(
        r
        for r in audit.read_json(OUT / "selected_windows.json")["records"]
        if r["id"].startswith("subject_019")
    )
    receipt = external_review.contact_sheet(
        pathlib.Path(row["path"]),
        OUT / "review" / (row["id"] + "_boundary.jpg"),
        230 / 15,
        255 / 15,
        24,
    )
    audit.save_json(
        OUT / "shoulder_boundary.json", dict(id=row["id"], preparation=receipt)
    )
    return dict(held_plank_windows=len(records), boundary_windows=1)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
