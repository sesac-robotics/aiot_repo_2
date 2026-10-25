"""보유 측면 스쿼트의 기존 학습 참여자에서 발 오류 검토판을 준비한다."""

import json

import audit
import external_review
from held_round2 import OUT

DEST = OUT.parent / "held_squat_errors"


def prepare() -> dict:
    """미사용 원본의 시간순 16표본을 준비하며 등급은 생성하지 않는다.

    Returns:
        dict: 후보·근거 경로. 제공자 bad는 후보 탐색에만 쓰고 점수로
        변환하지 않는다. 기존 train 참여자만 선택하고 기존 원본은 제외한다.

    Raises:
        ValueError: 분할·원본 해시 또는 디코딩 실패.
        OSError: 보유 원본·검토판 저장 실패.
    """
    rows = audit.read_json(OUT / "candidate_input_manifest.json")
    roles = {r["person"]: r["split"] for r in rows}
    known = {s for r in rows for s in r["sources"]}
    root = (
        audit.BASE
        / "data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise/squat/bad/side"
    )
    DEST.mkdir(exist_ok=True)
    records = []
    for path in sorted(root.glob("*.mp4")):
        subject = path.stem.split("_")[1]
        person = "shoulder_" + subject
        if roles.get(person) != "train":
            continue
        digest = audit.digest(path)
        if digest in known:
            continue
        evidence = external_review.contact_sheet(
            path, DEST / (path.stem + ".jpg"), 0, None, 16
        )
        records.append(
            dict(
                id=path.stem,
                person=person,
                path=str(path),
                sha256=digest,
                split="train",
                evidence=evidence,
                direct_labels_created=False,
            )
        )
        known.add(digest)
        audit.save_json(
            DEST / "screening.json",
            dict(records=records, completed=False, external_discovery=False),
        )
    report = dict(
        records=records,
        completed=True,
        external_discovery=False,
        provider_strings_are_not_scores=True,
    )
    audit.save_json(DEST / "screening.json", report)
    return dict(prepared=len(records), direct_labels_created=False)


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
