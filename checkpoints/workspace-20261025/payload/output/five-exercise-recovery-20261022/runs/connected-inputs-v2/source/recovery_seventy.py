"""종목별 70개 학습 구간 목표와 보유 자료의 다음 검토 대상을 분리 관리한다.

후보 수를 학습 수로 세지 않는다. 한 사람·원본의 반복 구간은 같은
분할로 유지하며 여러 카메라는 독립 수행자로 세지 않는다.
"""

import collections
import hashlib
import json
import pathlib

import cv2

import audit
import external_review
from recovery_prepare import ROOT


def plan() -> dict:
    """현재 실제 학습 수와 종목별 보강 후보를 사전 역할로 고정한다.

    Returns:
        dict: 목표·실제 학습 수·참여자·미달 수와 보유 개발 후보 목록.
        숄더 프레스 구간은 복수 반복의 검토 창이며 아직 단일 시도 라벨이 아니다.

    Raises:
        OSError: 보유 목록·모델 비교·원본 접근 실패.
        ValueError: 개발 분할 불일치 또는 영상 메타데이터 오류.
    """
    report = audit.read_json(ROOT / "comparison.json")
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    existing = audit.read_json(ROOT / "input_plan.json")["records"]
    existing.extend(audit.read_json(audit.OUT / "labels.json"))
    current = []
    for name in ["score_annotations.json", "additional_score_annotations.json"]:
        current.extend(
            audit.read_json(
                audit.BASE / "output/three-exercise-comparison-20261022" / name
            )
        )
    existing_ids = {r["id"].removesuffix("_existing_review") for r in existing}
    known_people = {r["person"] for r in existing}
    terms = {
        "pushups": [
            "shallow",
            "early stopping",
            "legs on the floor",
            "elevated hips",
            "90 degrees",
            "deep",
        ],
        "squats": [
            "shallow",
            "back not straight",
            "insufficient",
            "no obvious issue",
            "90 degrees",
            "over 90 degrees",
        ],
        "plank": [
            "high hips",
            "butt sinking",
            "on knees",
            "legs and hips on the floor",
            "average",
            "stopping early",
        ],
        "ytw": [
            "arms too low",
            "not low enough",
            "torso parallel",
            "too fast",
            "bent legs",
            "too slow",
        ],
    }
    selected = []
    for exercise, target in [
        ("pushups", 24),
        ("squats", 24),
        ("plank", 80),
        ("ytw", 40),
    ]:
        used = set()
        candidates = [
            r
            for r in pool
            if r["exercise"] == exercise
            and r["id"].startswith("qevd_")
            and r["split"] == "train"
            and r["id"] not in existing_ids
        ]
        candidates.sort(
            key=lambda r: (
                r["person"] in known_people,
                hashlib.sha256(
                    ("seventy-existing-first" + r["id"]).encode()
                ).hexdigest(),
            )
        )
        for i in range(target):
            available = [r for r in candidates if r["person"] not in used]
            if not available:
                break
            term = terms[exercise][i % len(terms[exercise])]
            source = next(
                (
                    r
                    for r in available
                    if any(term in s for s in r["source_labels"])
                ),
                available[0],
            )
            used.add(source["person"])
            selected.append(
                dict(
                    source,
                    selection_term=term,
                    review_status="not_reviewed_this_step",
                    score_label_created=False,
                )
            )
    front = {
        r["id"]: r
        for r in current
        if r["exercise"] == "shoulder"
        and r["id"].endswith("_front")
        and r["split"] == "train"
    }
    files = [
        r for r in pool if r["exercise"] == "shoulder" and r["id"] in front
    ]
    files.sort(
        key=lambda r: hashlib.sha256(
            ("seventy-shoulder" + r["id"]).encode()
        ).hexdigest()
    )
    for source in files[:20]:
        canonical = front[source["id"]]
        cap = cv2.VideoCapture(source["path"])
        fps = cap.get(5)
        count = int(cap.get(7))
        cap.release()
        for first_second in [8, 16]:
            if count / fps <= first_second + 4:
                continue
            selected.append(
                dict(
                    source,
                    person=canonical["person"],
                    split=canonical["split"],
                    id=source["id"] + f"_window_{first_second}",
                    source_id=source["id"],
                    window_seconds=[first_second, first_second + 4],
                    review_status="window_requires_direct_single_attempt_selection",
                    score_label_created=False,
                )
            )
    summary = [
        dict(
            exercise=r["exercise"],
            target_annotated_training_segments=70,
            actually_used_annotated_input_segments=r[
                "training_effective_segments"
            ],
            training_participant_groups=r["training_people"],
            additional_needed=max(0, 70 - r["training_effective_segments"]),
            all_four_items_have_70_labels=False,
            model_quality_complete=False,
        )
        for r in report["exercises"]
    ]
    result = dict(
        target_unit="한 항목 이상 실제 학습에 사용된 검토 구간. 카메라·같은 사람 반복·원본 수는 별도 표시해야 하며 독립 시도 70개 또는 항목별 70개를 뜻하지 않는다.",
        target_is_engineering_milestone_not_accuracy_guarantee=True,
        summary=summary,
        candidates=selected,
        candidate_counts=dict(
            collections.Counter(x["exercise"] for x in selected)
        ),
        external_search_paused=True,
        test_sources_decoded=False,
    )
    audit.save_json(ROOT / "seventy_plan.json", result)
    return result


def prepare_batch() -> dict:
    """보유 자료의 첫 오류 후보 묶음만 시간순 검토판으로 준비한다.

    Returns:
        dict: 종목별 최대 6개 개요 검토판. 직접 등급·학습 수는 만들지 않는다.

    Raises:
        OSError: 원본·출력 접근 실패.
        ValueError: 구간 또는 디코딩 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    queue = audit.read_json(ROOT / "seventy_plan.json")["candidates"]
    (ROOT / "review").mkdir(exist_ok=True)
    prepared = []
    for exercise in ["pushups", "squats", "shoulder", "ytw"]:
        for row in [x for x in queue if x["exercise"] == exercise][:6]:
            first, last = row.get("window_seconds", [0, None])
            receipt = external_review.contact_sheet(
                pathlib.Path(row["path"]),
                ROOT / "review" / (row["id"] + ".jpg"),
                first,
                last,
                24,
            )
            prepared.append(dict(row, preparation=receipt))
            audit.save_json(
                ROOT / "seventy_batch_preparation.json",
                dict(records=prepared, direct_review_performed_by_script=False),
            )
    return dict(prepared=len(prepared))


if __name__ == "__main__":
    import sys

    print(
        json.dumps(
            (
                prepare_batch()
                if "prepare" in sys.argv
                else {k: v for k, v in plan().items() if k != "candidates"}
            ),
            ensure_ascii=False,
        )
    )
