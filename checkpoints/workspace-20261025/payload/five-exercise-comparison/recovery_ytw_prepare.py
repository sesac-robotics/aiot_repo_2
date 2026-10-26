"""보유 QEVD YTW의 아직 라벨 없는 개발 후보를 시간순 검토판으로 준비한다."""

import json
import pathlib

import audit
import external_review
from recovery_prepare import ROOT


def run() -> dict:
    """기존 고정 대기열의 미검토 YTW만 준비하고 이전 근거를 보존한다.

    Returns:
        dict: 자동 준비한 새 검토판 수. 직접 검토·점수 라벨을 만들지 않는다.

    Raises:
        OSError: 기존 대기열·원본·출력 접근 실패.
        ValueError: 영상 디코딩·시간 구간 오류.
        RuntimeError: 온도·메모리 보호 조건 위반.
    """
    path = ROOT / "seventy_batch_preparation.json"
    previous = audit.read_json(path)["records"]
    known = {r["id"] for r in previous}
    count = 0
    for row in audit.read_json(ROOT / "seventy_plan.json")["candidates"]:
        if row["exercise"] != "ytw" or row["id"] in known:
            continue
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + ".jpg"),
            0,
            None,
            24,
        )
        previous.append(dict(row, preparation=receipt))
        count += 1
        audit.save_json(
            path,
            dict(records=previous, direct_review_performed_by_script=False),
        )
    return dict(new_prepared=count, directly_reviewed=False)


if __name__ == "__main__":
    print(json.dumps(run()))
