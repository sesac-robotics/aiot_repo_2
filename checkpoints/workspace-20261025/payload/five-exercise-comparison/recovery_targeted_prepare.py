"""보유 후속 반복 중 기존 구간과 겹치지 않는 오류 후보 검토판을 준비한다."""

import json
import pathlib

import audit
import external_review
from recovery_prepare import ROOT


def run() -> dict:
    """기존 역할을 유지한 푸시업·숄더 후보를 종목별 최대 6개 준비한다.

    Returns:
        dict: 자동 준비 수와 선택 ID. 제공자 bad 문자열은 검토 우선순위로만
        사용하며 직접 점수·정상 오류 판정·학습 수로 변환하지 않는다.

    Raises:
        OSError: 원본·기존 주석·검토판 접근 실패.
        ValueError: 프레임 또는 디코딩 계약 위반.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    if (ROOT / "full_repetition_manual_decisions_24.json").exists():
        existing = audit.read_json(ROOT / "full_repetition_batch_24.json")[
            "records"
        ]
        return dict(
            prepared=len(existing),
            cached_reviewed_batch=True,
            ids=[r["id"] for r in existing],
        )
    prior = []
    for folder, pattern in [
        (ROOT, "*score_annotations*.json"),
        (
            audit.BASE / "output/three-exercise-comparison-20261022",
            "*score_annotations*.json",
        ),
    ]:
        for path in folder.glob(pattern):
            prior.extend(audit.read_json(path))
    queue = audit.read_json(ROOT / "full_repetition_plan.json")["records"]
    prepared = []
    for exercise in ("pushups", "shoulder"):
        people = set()
        for row in queue:
            if row["exercise"] != exercise or "_bad_" not in row["id"]:
                continue
            if row["person"] in people:
                continue
            first, last = row["window_seconds"]
            if any(
                r["video_sha256"] == row["video_sha256"]
                and not (
                    last < r["segment"]["start_frame"] / r["segment"]["fps"]
                    or first > r["segment"]["end_frame"] / r["segment"]["fps"]
                )
                for r in prior
            ):
                continue
            receipt = external_review.contact_sheet(
                pathlib.Path(row["video_path"]),
                ROOT / "review" / (row["id"] + "_targeted.jpg"),
                first,
                last,
                24,
            )
            prepared.append(dict(row, sheets=dict(primary=receipt)))
            people.add(row["person"])
            audit.save_json(
                ROOT / "full_repetition_batch_24.json",
                dict(records=prepared, directly_reviewed_by_script=False),
            )
            if len(people) >= 6:
                break
    return dict(prepared=len(prepared), ids=[r["id"] for r in prepared])


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
