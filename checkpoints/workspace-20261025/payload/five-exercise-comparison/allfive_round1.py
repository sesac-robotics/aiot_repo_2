"""보유 5종의 미검토 후보를 분할·기존 결과를 보존하여 별도 검토한다."""

import collections
import hashlib
import json
import pathlib
import re
import sys

import cv2

import audit
import external_review
from recovery_prepare import ROOT

OUT = ROOT / "allfive_round1"


def known_ids() -> set[str]:
    """기존 직접 결정·점수 주석의 ID를 수집한다.

    Returns:
        set[str]: 중복 검토를 피할 기존 ID. 자동 선별 전체 대기열은 포함하지
        않으며 직접 검토 수 자체를 이 집합 크기로 주장하지 않는다.

    Raises:
        OSError: 기존 결정 파일에 접근하지 못한 경우.
    """
    known = set()
    for folder in (
        audit.OUT,
        ROOT,
        audit.BASE / "output/three-exercise-comparison-20261022",
        audit.BASE / "output/existing-data-audit-20261022",
    ):
        for path in folder.rglob("*.json"):
            if any(
                p
                in ("snapshots", "runs", "heavy", "previews", "allfive_round1")
                for p in path.relative_to(folder).parts
            ):
                continue
            name = path.name
            if not (
                name == "labels.json"
                or "direct" in name
                or "manual_decisions" in name
                or "score_annotations" in name
            ):
                continue
            known.update(re.findall(r"qevd_\d{8}", path.read_text()))
    return known


def prepare() -> dict:
    """종목별 부족 항목 우선 후보의 시간순 표본을 자동 준비한다.

    Returns:
        dict: 자동 준비한 원본 수. 제공자 문자열은 검토 순서에만 사용하며
        직접 등급·점수·학습 가능 수를 만들지 않는다. test는 열지 않는다.

    Raises:
        OSError: 보유 후보·원본·검토판 접근 실패.
        ValueError: 기존 역할·원본·시간 구간 불일치.
        RuntimeError: 자원 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    (OUT / "review").mkdir(exist_ok=True)
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    known = known_ids()
    plan = []
    terms = {
        "pushups": [
            "early stopping",
            "legs on the floor",
            "butt sinking",
            "deep",
        ],
        "squats": [
            "insufficient",
            "back not straight",
            "shallow",
            "no obvious issue",
        ],
        "plank": [
            "high hips",
            "butt sinking",
            "on knees",
            "legs and hips on the floor",
            "stopping early",
            "average",
            "high hips",
            "average",
        ],
        "ytw": [
            "torso parallel",
            "arms too low",
            "not low enough",
            "torso parallel",
        ],
    }
    for exercise in terms:
        used = set()
        candidates = [
            r
            for r in pool
            if r["exercise"] == exercise
            and r["id"].startswith("qevd_")
            and r["split"] == "train"
            and r["id"] not in known
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(
                ("allfive-round1" + r["id"]).encode()
            ).hexdigest()
        )
        for term in terms[exercise]:
            r = next(
                (
                    r
                    for r in candidates
                    if r["person"] not in used
                    and any(term in t for t in r["source_labels"])
                ),
                None,
            )
            if r is None:
                continue
            used.add(r["person"])
            plan.append(dict(r, selection_term=term))
    # 숄더는 QEVD 후보가 없으므로 보유 MultiView 전면의 미검토 후반 창을 사용한다.
    shoulder = [
        r
        for r in pool
        if r["exercise"] == "shoulder"
        and r["split"] == "train"
        and r["id"].endswith("_front")
        and "_bad_" in r["id"]
    ]
    shoulder.sort(
        key=lambda r: hashlib.sha256(
            ("allfive-shoulder" + r["id"]).encode()
        ).hexdigest()
    )
    plan.extend(
        dict(
            r,
            id=r["id"] + "_round1_late",
            source_id=r["id"],
            selection_term="held_later_window_not_new_participant",
        )
        for r in shoulder[:4]
    )
    existing = (
        audit.read_json(OUT / "preparation.json")["records"]
        if (OUT / "preparation.json").exists()
        else []
    )
    ready = {r["id"]: r for r in existing}
    for source in plan:
        if source["id"] in ready:
            continue
        path = pathlib.Path(source["path"])
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(5)
        count = int(cap.get(7))
        cap.release()
        if source["exercise"] == "shoulder":
            start = max(0, (count / fps) - 7)
            end = (count - 1) / fps
        else:
            start = 0
            end = None
        receipt = external_review.contact_sheet(
            path, OUT / "review" / (source["id"] + ".jpg"), start, end, 24
        )
        ready[source["id"]] = dict(
            source,
            preparation=receipt,
            automatically_prepared_not_directly_reviewed=True,
        )
        audit.save_json(
            OUT / "preparation.json",
            dict(records=list(ready.values()), completed=False),
        )
    report = dict(
        records=list(ready.values()),
        completed=True,
        automatic_preparation_not_direct_review=True,
        exercise_counts=dict(
            collections.Counter(r["exercise"] for r in ready.values())
        ),
        known_direct_record_id_exclusions=len(known),
        external_discovery=False,
    )
    audit.save_json(OUT / "preparation.json", report)
    return dict(
        prepared=len(ready),
        exercise_counts=report["exercise_counts"],
        direct_labels_created=0,
    )


def dense() -> dict:
    """직접 선택한 원본 구간의 확대 검토판을 준비한다.

    Returns:
        dict: 구간별 자동 표본 준비 수. 점수·관절 추출·직접 검토는 아니다.

    Raises:
        ValueError: 원본 구간이나 디코딩이 유효하지 않은 경우.
        OSError: 입력 또는 근거 저장 실패.
    """
    records = []
    for row in audit.read_json(OUT / "selected_windows.json")["records"]:
        receipt = row["preparation"]
        result = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            OUT / "review" / (row["id"] + "_dense.jpg"),
            row["start_frame"] / receipt["fps"],
            row["end_frame"] / receipt["fps"],
            24,
        )
        records.append(dict(id=row["id"], preparation=result))
        audit.save_json(OUT / "dense_preparation.json", dict(records=records))
    return dict(prepared=len(records), direct_labels_created=0)


if __name__ == "__main__":
    print(
        json.dumps(
            dense() if "dense" in sys.argv else prepare(), ensure_ascii=False
        )
    )
