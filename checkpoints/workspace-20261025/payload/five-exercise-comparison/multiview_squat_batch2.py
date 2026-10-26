"""보유 MultiView 스쿼트 12묶음의 두 번째 직접 검토 실험을 준비한다.

기존 5종 결과와 첫 MultiView 실험을 변경하지 않는다. 자동 검토판·광류
시차는 직접 점수 라벨이나 같은 수행의 승인으로 간주하지 않는다.
"""

import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import external_review
from multiview_squat_prepare import OUT as PREVIOUS
from multiview_squat_sync import lag_scores, motion_trace
from resources import check_resources

OUT = PREVIOUS / "batch2"


def boards() -> dict:
    """선택 시차 후보의 한 반복을 두 시점·8장씩 나란히 준비한다.

    Returns:
        dict: 자동 준비 검토판 수. 점수 라벨·동기화 승인은 생성하지 않는다.

    Raises:
        ValueError: 승인 전 구간 후보 또는 원본 프레임 디코딩 실패.
        OSError: 선택 구간·검토판 접근 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    rows = {
        r["id"]: r for r in audit.read_json(OUT / "preparation.json")["records"]
    }
    receipts = []
    previous_path = OUT / "interval_review_preparation.json"
    previous = (
        {r["id"]: r for r in audit.read_json(previous_path)["records"]}
        if previous_path.exists()
        else {}
    )
    for interval in audit.read_json(OUT / "interval_candidates.json")[
        "records"
    ]:
        old = previous.get(interval["id"])
        if (
            old
            and all(old.get(k) == v for k, v in interval.items())
            and pathlib.Path(old["evidence_image"]).exists()
        ):
            receipts.append(old)
            continue
        row = rows[interval["source_id"]]
        indices = np.rint(
            np.linspace(interval["start_frame"], interval["end_frame"], 8)
        ).astype(int)
        board = Image.new("RGB", (1920, 960), "white")
        draw = ImageDraw.Draw(board)
        for v, view in enumerate(("side", "front")):
            check_resources()
            cap = cv2.VideoCapture(
                row["views"][view]["video_path"],
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_N_THREADS, 1],
            )
            frames = indices + (interval["lag_frames"] if v else 0)
            if min(frames) < 0:
                raise ValueError("Negative paired source frame")
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
                decoded = int(frames[0]) - 1
                for k, index in enumerate(frames):
                    while decoded < index:
                        if not cap.grab():
                            raise ValueError("Paired sample missing")
                        decoded += 1
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError("Paired sample decode failed")
                    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    im.thumbnail((238, 450))
                    board.paste(im, (k * 240, v * 480 + 25))
                    draw.text(
                        (k * 240 + 3, v * 480 + 4),
                        f"{view}: {index}",
                        fill="black",
                    )
            finally:
                cap.release()
        target = OUT / "review" / (interval["id"] + "_paired.jpg")
        board.save(target, quality=94)
        receipts.append(
            dict(
                interval,
                evidence_image=str(target),
                side_frames=indices.tolist(),
                front_frames=(indices + interval["lag_frames"]).tolist(),
                automatic_preparation_not_direct_review=True,
            )
        )
        audit.save_json(
            OUT / "interval_review_preparation.json",
            dict(records=receipts, completed=False),
        )
    audit.save_json(
        OUT / "interval_review_preparation.json",
        dict(records=receipts, completed=True),
    )
    return dict(boards=len(receipts), score_labels_created=0)


def prepare() -> dict:
    """기존 역할을 유지한 12묶음·24시점 검토판과 자동 시차를 저장한다.

    Returns:
        dict: 자동 준비된 묶음·시점 수. 직접 검토 수·점수 생성 수는 0이다.

    Raises:
        ValueError: 기존 역할·원본 해시·FPS 불일치 또는 시험 원본 포함.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 원본·검토판·자동 결과 접근 실패.
    """
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    (OUT / "review").mkdir(exist_ok=True)
    canonical = {}
    annotations = audit.read_json(
        audit.BASE
        / "output/three-exercise-comparison-20261022/score_annotations.json"
    )
    tests = set()
    for file in (audit.OUT / "labels.json",):
        annotations += audit.read_json(file)
    for row in annotations:
        if row.get("split") == "test":
            tests.update(
                row.get(k)
                for k in ("video_sha256", "front_sha256")
                if row.get(k)
            )
        if row["id"].startswith("subject_") and row.get(
            "person", ""
        ).startswith("shoulder_"):
            subject = row["id"].split("_")[1]
            if subject in canonical and (
                canonical[subject]["person"],
                canonical[subject]["split"],
            ) != (row["person"], row["split"]):
                raise ValueError("Existing participant role differs")
            canonical.setdefault(subject, row)
    records = []
    old_path = OUT / "preparation.json"
    old = (
        {r["id"]: r for r in audit.read_json(old_path)["records"]}
        if old_path.exists()
        else {}
    )
    for candidate in audit.read_json(PREVIOUS / "next_held_candidates.json")[
        "records"
    ]:
        identifier = candidate["id"]
        parent = canonical[identifier.split("_")[1]]
        if parent["split"] != candidate["existing_role"] or parent[
            "split"
        ] not in ("train", "validation"):
            raise ValueError("Cannot decode test or unresolved candidates")
        row = dict(
            id=identifier,
            exercise="squats",
            person=parent["person"],
            worker_id=parent["worker_id"],
            split=parent["split"],
            provider_quality=identifier.split("_")[-1],
            views={},
        )
        for view in ("side", "front"):
            source = pathlib.Path(candidate["views"][view]["path"])
            sha = audit.digest(source)
            if sha in tests:
                raise ValueError(
                    "Previously reserved test source cannot be decoded"
                )
            receipt = old.get(identifier, {}).get("views", {}).get(view)
            if receipt is None:
                receipt = external_review.contact_sheet(
                    source,
                    OUT / "review" / (identifier + "_" + view + ".jpg"),
                    0,
                    8,
                    24,
                )
            if receipt["video_sha256"] != sha:
                raise ValueError("Temporal sheet source changed")
            row["views"][view] = receipt
        if row["views"]["side"]["fps"] != row["views"]["front"]["fps"]:
            raise ValueError("Different native frame rates")
        alignment = old.get(identifier, {}).get("automatic_alignment")
        if alignment is None:
            traces = {
                v: motion_trace(row["views"][v]["video_path"], 120)
                for v in ("side", "front")
            }

            def best(a, b, limit):
                candidates = lag_scores(a, b, limit)
                return candidates[0] if candidates else None

            alignment = dict(
                best=best(traces["side"], traces["front"], 36),
                first_half=best(traces["side"][:60], traces["front"][:60], 24),
                second_half=best(traces["side"][60:], traces["front"][60:], 24),
                trace=traces,
                same_performance_verified=False,
            )
        row["automatic_alignment"] = alignment
        records.append(row)
        audit.save_json(
            old_path,
            dict(
                records=records,
                completed=False,
                automatic_preparation_not_direct_review=True,
                score_labels_created=0,
            ),
        )
    audit.save_json(
        old_path,
        dict(
            records=records,
            completed=True,
            automatic_preparation_not_direct_review=True,
            score_labels_created=0,
        ),
    )
    return dict(
        candidates=len(records),
        view_sheets=sum(len(r["views"]) for r in records),
        score_labels_created=0,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False))
    elif sys.argv[1] == "boards":
        print(json.dumps(boards(), ensure_ascii=False))
