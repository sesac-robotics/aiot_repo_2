"""직접 검토 결정을 QEVD 기본 JSON·구간 초안·Heavy 입력으로 연결한다."""

import copy
import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import extract
import render
from allfive_round1 import OUT
from resources import check_resources

DEFINITIONS = {
    "pushups": (
        ["trunk_line", "descent_range", "return_extension", "support"],
        [40, 30, 20, 10],
    ),
    "squats": (
        ["descent_range", "return_extension", "knee_foot", "trunk_balance"],
        [30, 20, 30, 20],
    ),
    "shoulder": (
        [
            "overhead_reach",
            "lower_return",
            "bilateral_timing",
            "frontal_trunk_hold",
        ],
        [30, 25, 25, 20],
    ),
    "plank": (
        [
            "trunk_line",
            "elbow_sagittal_position",
            "leg_and_body_support",
            "support_continuity",
        ],
        [40, 25, 20, 15],
    ),
    "ytw": (
        ["y_reach", "t_reach", "w_shape", "hinge_and_trunk_hold"],
        [30, 25, 25, 20],
    ),
}


def build() -> dict:
    """명시적인 직접 결정만 검증·변환하며 새 자동 채점 정답을 만들지 않는다.

    Returns:
        dict: 항목 초안·원본 기본 행·사건 별도 주석 수. 기본 네 필드는
        video_path·labels·labels_descriptive·split이며 worker는 별도 연결한다.

    Raises:
        ValueError: 원본 지문·구간·기존 역할·주석 중복·관측 계약 불일치.
        OSError: 결정·원본·출력 저장 실패.
    """
    prepared = {
        r["id"]: r
        for r in audit.read_json(OUT / "selected_windows.json")["records"]
    }
    extras = audit.read_json(OUT / "extra_preparation.json")["records"]
    for r in extras:
        prepared[r["id"]] = dict(
            r,
            start_frame=r["preparation"]["start_frame"],
            end_frame=r["preparation"]["end_frame"],
        )
    decisions = audit.read_json(OUT / "manual_decisions.json")
    originals = {
        pathlib.Path(r["video_path"]).stem: r
        for r in audit.read_json(
            audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
        )
    }
    workers = {
        pathlib.Path(r["video_path"]).stem: r["worker_id"]
        for r in audit.read_json(
            audit.BASE
            / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
        )
    }
    registry = audit.read_json(
        audit.BASE
        / "output/three-exercise-comparison-20261022/external_worker_registry.json"
    )
    if extras[0]["person"] not in registry:
        registry[extras[0]["person"]] = max(registry.values()) + 1
    historical = []
    for folder in (
        audit.BASE / "output/three-exercise-comparison-20261022",
        OUT.parent,
    ):
        for path in folder.glob("*score_annotations*.json"):
            historical.extend(audit.read_json(path))
    historical += audit.read_json(audit.OUT / "labels.json")
    rows = []
    basic = {}
    participants = {}
    plans = {}
    for decision in decisions["records"]:
        source = prepared[decision["id"]]
        qevd = source["id"].startswith("qevd_")
        e = source["exercise"]
        rec = source["preparation"]
        first, last = source["start_frame"], source["end_frame"]
        sha = rec["video_sha256"]
        if (
            sha != audit.digest(pathlib.Path(source["path"]))
            or source["split"] != "train"
        ):
            raise ValueError("Source or role changed")
        for previous in historical:
            if previous.get("video_sha256") != sha:
                continue
            if (
                previous["person"] != source["person"]
                or previous["split"] != "train"
            ):
                raise ValueError("Existing source role differs")
            interval = previous["segment"]
            if not (
                last < interval["start_frame"] or first > interval["end_frame"]
            ):
                raise ValueError("Reviewed source interval already labeled")
        if qevd:
            native = copy.deepcopy(originals[pathlib.Path(source["path"]).stem])
            worker = workers[pathlib.Path(source["path"]).stem]
            if (
                source["person"] != "qevd_" + str(worker)
                or native["split"] == "test"
            ):
                raise ValueError("Native source role mismatch")
        else:
            native = dict(
                video_path="./external/"
                + ("DVIDS" if e == "plank" else "MultiView")
                + "/"
                + sha
                + ".mp4",
                labels=source.get(
                    "source_labels",
                    source.get("original", {}).get("labels", []),
                ),
                labels_descriptive=[],
                split="train",
            )
            worker = registry[source["person"]]
        levels = decision["levels"]
        ids, weights = DEFINITIONS[e]
        if len(levels) != 4 or any(
            type(g) is not int or g not in (0, 1, 2)
            for g in levels
            if g is not None
        ):
            raise ValueError("Invalid direct grades")
        complete = decision["attempt_complete"]
        score = (
            sum(w * g / 2 for w, g in zip(weights, levels))
            if complete and None not in levels
            else None
        )
        row = dict(
            id=source["id"] + "_round1_score",
            source_id=source["id"],
            exercise=e,
            video_path=source["path"],
            video_sha256=sha,
            qevd_video_path=native["video_path"],
            person=source["person"],
            worker_id=worker,
            split="train",
            labels=native["labels"],
            labels_descriptive=native["labels_descriptive"],
            segment=dict(start_frame=first, end_frame=last, fps=rec["fps"]),
            required_input_views=(
                ["side"] if source["id"].startswith("held_navy") else ["front"]
            ),
            assessment=dict(
                criterion_ids=ids,
                criterion_weights=weights,
                criterion_levels=levels,
                score=score,
                rubric_version=(
                    "project-plank-side-core-v2"
                    if e == "plank"
                    else (
                        "project-core-form-v1"
                        if e in ("pushups", "squats")
                        else "project-other-three-core-v1"
                    )
                ),
                expert_validated=False,
                label_source="assistant_direct_temporal_sample_review",
                note_ko=decision["note_ko"],
            ),
            attempt_complete=complete,
            eligible_for_total_score_training=score is not None,
            evidence_images=[rec["evidence_image"]],
            provenance=dict(
                dataset=(
                    "QEVD" if qevd else "DVIDS" if e == "plank" else "MultiView"
                ),
                provider_strings_not_score_targets=True,
            ),
            continuous_full_video_review=False,
        )
        dense = OUT / "review" / (source["id"] + "_dense.jpg")
        if dense.exists() and e != "plank":
            row["evidence_images"].append(str(dense))
        if source["id"].startswith("subject_019"):
            row["evidence_images"].append(
                str(OUT / "review" / (source["id"] + "_boundary.jpg"))
            )
        rows.append(row)
        basic[native["video_path"]] = native
        participants[native["video_path"]] = dict(
            video_path=native["video_path"], worker_id=worker
        )
        plans[source["id"]] = row
    for event in decisions["events"]:
        source = prepared[event["id"]]
        if source["id"] not in plans:
            plans[source["id"]] = dict(
                id=source["id"] + "_round1_event",
                source_id=source["id"],
                exercise="plank",
                video_path=source["path"],
                video_sha256=source["preparation"]["video_sha256"],
                person=source["person"],
                split="train",
                segment=dict(
                    start_frame=source["start_frame"],
                    end_frame=source["end_frame"],
                    fps=source["preparation"]["fps"],
                ),
                required_input_views=["front"],
                event_only=True,
            )
    audit.save_json(OUT / "score_annotations.json", rows)
    audit.save_json(OUT / "fine_grained_labels.json", list(basic.values()))
    audit.save_json(
        OUT / "fine_grained_labels_with_worker_ids.json",
        list(participants.values()),
    )
    audit.save_json(OUT / "external_worker_registry.json", registry)
    audit.save_json(
        OUT / "extraction_plan.json",
        dict(
            records=list(plans.values()),
            score_labels=len(rows),
            event_annotations=len(decisions["events"]),
        ),
    )
    return dict(
        score_drafts=len(rows),
        original_videos=len(basic),
        extract_windows=len(plans),
        events=len(decisions["events"]),
    )


def process() -> dict:
    """허용 원본 구간을 Heavy로 추출하고 관절 표시 표본을 자동 준비한다.

    Returns:
        dict: 추출·표시판 수. 직접 관절 검토·추적 승인·점수는 생성하지 않는다.

    Raises:
        ValueError: 캐시 지문·프레임·원본 계약 불일치.
        OSError: 영상·관절·표시판 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cv2.setNumThreads(1)
    receipts = []
    boards = []
    for row in audit.read_json(OUT / "extraction_plan.json")["records"]:
        check_resources()
        rec = extract.extract_view(
            dict(row, required_views=row["required_input_views"]),
            "primary",
            row["video_path"],
            OUT / "heavy",
        )
        receipts.append(dict(id=row["id"], extraction=rec))
        with np.load(rec["cache"]) as data:
            frames = data["frame_numbers"]
            points = data["image_xy"]
            if str(data["fingerprint"]) != rec["fingerprint"]:
                raise ValueError("Cache fingerprint differs")
        chosen = np.unique(
            np.rint(np.linspace(0, len(frames) - 1, 12)).astype(int)
        )
        canvas = Image.new("RGB", (1024, 1380), "white")
        draw = ImageDraw.Draw(canvas)
        cap = cv2.VideoCapture(row["video_path"])
        cap.set(1, int(frames[0]))
        decoded = int(frames[0]) - 1
        try:
            for k, idx in enumerate(chosen):
                while decoded < int(frames[idx]):
                    if not cap.grab():
                        raise ValueError("Joint sheet source ended")
                    decoded += 1
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Joint sheet frame missing")
                for a, b in render.CONNECTIONS:
                    if np.isfinite(points[idx, [a, b]]).all():
                        pa, pb = np.rint(points[idx, [a, b]]).astype(int)
                        cv2.line(frame, tuple(pa), tuple(pb), (0, 255, 0), 3)
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im.thumbnail((254, 430))
                x, y = (k % 4) * 256, (k // 4) * 460
                canvas.paste(im, (x, y + 24))
                draw.text((x + 2, y + 3), str(frames[idx]), fill="black")
        finally:
            cap.release()
        path = OUT / "review" / (row["id"] + "_joints.jpg")
        canvas.save(path, quality=94)
        boards.append(
            dict(
                id=row["id"],
                fingerprint=rec["fingerprint"],
                evidence_image=str(path),
                frames=frames[chosen].tolist(),
                automatically_prepared_not_directly_reviewed=True,
            )
        )
        audit.save_json(
            OUT / "extraction.json", dict(records=receipts, completed=False)
        )
        audit.save_json(
            OUT / "joint_review_preparation.json", dict(records=boards)
        )
    audit.save_json(
        OUT / "extraction.json", dict(records=receipts, completed=True)
    )
    return dict(
        extracted=len(receipts),
        prepared_joint_sheets=len(boards),
        directly_reviewed=False,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            process() if "extract" in sys.argv else build(), ensure_ascii=False
        )
    )
