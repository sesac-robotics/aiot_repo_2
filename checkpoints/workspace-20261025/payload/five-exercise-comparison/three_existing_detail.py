"""보유 영상에서 직접 선택한 네 구간을 조밀한 검토판으로 준비한다."""

import json
import pathlib

import audit
import external_review
from three_existing_review import ROOT


def run() -> dict:
    """선택한 프레임 구간의 시간순 36장 검토판을 저장한다.

    Returns:
        dict: 원본 해시·FPS·프레임 번호. 등급은 생성하지 않는다.

    Raises:
        OSError: 입력·출력 접근 실패.
        ValueError: 구간·디코딩 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    import cv2

    rows = audit.read_json(ROOT / "expanded_preparation.json")["records"]
    ranges = {
        "qevd_00104489": (0, 108),
        "qevd_00209671": (0, 106),
        "qevd_00159081": (0, 68),
        "qevd_00246145": (0, 146),
    }
    results = []
    for row in rows:
        if row["id"] not in ranges:
            continue
        cap = cv2.VideoCapture(row["path"])
        fps = cap.get(5)
        cap.release()
        first, last = ranges[row["id"]]
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + "_detail.jpg"),
            first / fps,
            last / fps,
            36,
        )
        results.append(dict(id=row["id"], preparation=receipt))
    audit.save_json(
        ROOT / "detail_preparation.json",
        dict(records=results, direct_review_performed_by_script=False),
    )
    return dict(prepared=len(results))


if __name__ == "__main__":
    print(json.dumps(run()))
