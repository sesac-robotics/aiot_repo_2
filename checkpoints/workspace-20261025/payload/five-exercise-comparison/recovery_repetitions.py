"""보유 원본의 미라벨 후속 반복을 찾아 완전 채점 후보를 준비한다.

반복은 같은 참여자·원본 분할을 유지한다. 검토 창이나 제공자 반복 번호를
독립 참여자·직접 점수·학습 완료 수로 바꾸지 않는다.
"""

import copy
import csv
import hashlib
import json
import pathlib

import cv2

import audit
import external_review
from recovery_prepare import ROOT


def plan() -> dict:
    """푸시업·숄더 원본의 후속 창과 기존 스쿼트 미라벨 반복을 선정한다.

    Returns:
        dict: 학습 분할을 보존한 보유 후보 목록·기존 스쿼트 반복 수.
        정면·측면 카메라를 서로 다른 시도로 세지 않는다.

    Raises:
        OSError: 기존 목록·원본·분할 접근 실패.
        ValueError: 역할·원본 영상 메타데이터 오류.
    """
    root = audit.BASE / "output/three-exercise-comparison-20261022"
    previous = []
    for name in ["score_annotations.json", "additional_score_annotations.json"]:
        previous.extend(audit.read_json(root / name))
    known = {r["id"] for r in previous}
    interval_labels = list(previous)
    if (ROOT / "additional_score_annotations.json").exists():
        interval_labels.extend(
            audit.read_json(ROOT / "additional_score_annotations.json")
        )
    selected = []
    for exercise, suffix in [("pushups", "_side"), ("shoulder", "_front")]:
        candidates = [
            r
            for r in previous
            if r["exercise"] == exercise
            and r["id"].startswith("subject_")
            and r["id"].endswith(suffix)
            and r["split"] == "train"
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(
                ("whole-reps-70" + r["id"]).encode()
            ).hexdigest()
        )
        hashes = set()
        for source in candidates:
            sha = source["video_sha256"]
            if sha in hashes:
                continue
            hashes.add(sha)
            cap = cv2.VideoCapture(source["video_path"])
            fps = cap.get(5)
            count = int(cap.get(7))
            cap.release()
            for first_second in [8, 16, 24]:
                if count / fps <= first_second + 4:
                    continue
                a, b = round(first_second * fps), round(
                    (first_second + 4) * fps
                )
                prior_range = source["segment"]
                if not (
                    b < prior_range["start_frame"]
                    or a > prior_range["end_frame"]
                ):
                    continue
                if any(
                    r["video_sha256"] == sha
                    and not (
                        b < r["segment"]["start_frame"]
                        or a > r["segment"]["end_frame"]
                    )
                    for r in interval_labels
                ):
                    continue
                row = copy.deepcopy(source)
                row.update(
                    id=source["id"] + f"_recovery_window_{first_second}",
                    source_id=source["id"],
                    window_seconds=[first_second, first_second + 4],
                    candidate_only=True,
                )
                for key in ["assessment", "score", "criterion_levels"]:
                    row.pop(key, None)
                selected.append(row)
    with (root / "Segmentation.csv").open() as file:
        annotations = [
            r
            for r in csv.DictReader(file, delimiter=";")
            if r["exercise_id"] == "6" and r["cam17_orientation"] == "front"
        ]
    bases = {
        r["person"]: r
        for r in previous
        if r["exercise"] == "squats"
        and r.get("provenance", {}).get("dataset") == "REHAB24"
    }
    for annotation in annotations:
        identifier = f"rehab24_{annotation['video_id']}_rep{annotation['repetition_number']}"
        if identifier in known:
            continue
        row = copy.deepcopy(bases["rehab24_" + annotation["person_id"]])
        if row["split"] != "train":
            raise ValueError("Existing cohort role changed")
        row.update(
            id=identifier,
            source_id=annotation["video_id"],
            segment=dict(
                start_frame=int(annotation["first_frame"]),
                end_frame=int(annotation["last_frame"]),
                fps=30.0,
            ),
            provider_annotation=annotation,
            candidate_only=True,
        )
        for key in ["assessment", "score", "criterion_levels"]:
            row.pop(key, None)
        selected.append(row)
    # 여러 참여자의 미라벨 정상/오류 후보를 번갈아 검토하도록 순서를 고정한다.
    selected.sort(
        key=lambda r: hashlib.sha256(
            ("review-next-rep" + r["id"]).encode()
        ).hexdigest()
    )
    report = dict(
        records=selected,
        rehab24_available_segment_annotations=len(annotations),
        rehab24_previously_labeled=sum(
            r["id"].startswith("rehab24_") for r in previous
        ),
        candidates_are_not_new_labels=True,
        independent_participants_not_increased=True,
    )
    audit.save_json(ROOT / "full_repetition_plan.json", report)
    return report


def prepare(
    offset: int = 0,
    limit: int = 6,
    exercises: tuple = ("pushups", "squats", "shoulder"),
) -> dict:
    """각 종목의 후속 검토 창과 동기화 스쿼트 구간 검토판을 준비한다.

    Args:
        offset: 종목별 후보 시작 위치, 0 이상 정수.
        limit: 종목별 준비 상한, 양수 정수.
        exercises: 준비할 운동 코드의 튜플. 기본값은 3종이다.

    Returns:
        dict: 자동 준비한 구도별 시간순 24장 검토판과 원본 연결.
        직접 검토·구간 확정·등급 생성은 수행하지 않는다.

    Raises:
        OSError: 원본·검토판 출력 실패.
        ValueError: 인수 또는 프레임 범위·디코딩 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    if offset < 0 or limit <= 0:
        raise ValueError("Nonnegative offset and positive limit required")
    queue = audit.read_json(ROOT / "full_repetition_plan.json")["records"]
    (ROOT / "review").mkdir(exist_ok=True)
    prepared = []
    for exercise in exercises:
        for row in [r for r in queue if r["exercise"] == exercise][
            offset : offset + limit
        ]:
            if "window_seconds" in row:
                first, last = row["window_seconds"]
            else:
                first, last = (
                    row["segment"]["start_frame"] / row["segment"]["fps"],
                    row["segment"]["end_frame"] / row["segment"]["fps"],
                )
            sheets = {}
            for view, path in [("primary", row["video_path"])] + (
                [("front", row["front_path"])] if exercise == "squats" else []
            ):
                sheets[view] = external_review.contact_sheet(
                    pathlib.Path(path),
                    ROOT / "review" / (row["id"] + "_" + view + ".jpg"),
                    first,
                    last,
                    24,
                )
            prepared.append(dict(row, sheets=sheets))
            audit.save_json(
                ROOT / f"full_repetition_batch_{offset}.json",
                dict(records=prepared, directly_reviewed_by_script=False),
            )
    return dict(prepared=len(prepared))


if __name__ == "__main__":
    import sys

    print(
        json.dumps(
            (
                prepare(
                    int(sys.argv[2]) if len(sys.argv) > 2 else 0,
                    int(sys.argv[3]) if len(sys.argv) > 3 else 6,
                    (
                        tuple(sys.argv[4:])
                        if len(sys.argv) > 4
                        else ("pushups", "squats", "shoulder")
                    ),
                )
                if "prepare" in sys.argv
                else {k: v for k, v in plan().items() if k != "records"}
            ),
            ensure_ascii=False,
        )
    )
