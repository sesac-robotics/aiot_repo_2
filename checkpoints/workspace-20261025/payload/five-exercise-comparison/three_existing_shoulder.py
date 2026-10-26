"""정면 도달 관측이 부족했던 보유 숄더 프레스의 사선 입력을 준비한다."""

import json
import pathlib

import audit
import external_review
from three_existing_review import ROOT


def run(detail: bool = False) -> dict:
    """기존 네 개발 사례의 사선 영상을 시간순 36장으로 준비한다.

    Args:
        detail: True이면 처음 4초의 조밀한 구간 표본, False이면 전체 개요.

    Returns:
        dict: 실제 사선 원본·표본·해시·기존 정면의 참여자 분할 연결.
        사선의 등급을 정면으로 복사하거나 원본 동기화를 가정하지 않는다.

    Raises:
        OSError: 원본·라벨 접근 실패.
        ValueError: 기존 개발 분할 불일치 또는 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    target = [
        "subject_010_shoulder_good",
        "subject_013_shoulder_bad",
        "subject_018_shoulder_bad",
        "subject_026_shoulder_bad",
    ]
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    previous = []
    for name in ("score_annotations.json", "additional_score_annotations.json"):
        previous.extend(
            audit.read_json(
                audit.BASE / "output/three-exercise-comparison-20261022" / name
            )
        )
    known = {x["id"]: x for x in previous}
    rows = []
    for stem in target:
        source = next(x for x in pool if x["id"] == stem + "_diagonal")
        original = known[stem + "_front"]
        if original["split"] == "test":
            raise ValueError("Development only")
        receipt = external_review.contact_sheet(
            pathlib.Path(source["path"]),
            ROOT
            / "review"
            / (source["id"] + ("_detail" if detail else "") + ".jpg"),
            0,
            4.0 if detail else None,
            36,
        )
        rows.append(
            dict(
                source,
                person=original["person"],
                split=original["split"],
                preparation=receipt,
                matched_front_id=original["id"],
                same_person_group_does_not_imply_synchronized_attempt=True,
            )
        )
    audit.save_json(
        ROOT
        / (
            "shoulder_detail_preparation.json"
            if detail
            else "shoulder_preparation.json"
        ),
        dict(records=rows, direct_review_performed_by_script=False),
    )
    return dict(prepared=len(rows))


if __name__ == "__main__":
    import sys

    print(json.dumps(run(detail="detail" in sys.argv)))
