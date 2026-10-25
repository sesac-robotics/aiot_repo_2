"""직접 검토한 보유 뒤꿈치 오류를 별도 스쿼트 후보로 학습·검증한다."""

import json
import pathlib
import sys

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupKFold

import audit
import extract
import recovery_train
import render
import three_items
from allfive_round1_features import load
from allfive_round1_train import manifest, predict
from held_feature_update import fit, update_existing, update_note
from held_round2 import OUT as ROUND2
from held_round2_train import added_rows
from held_squat_error_screen import DEST
from multiview_squat_batch2_train import load_batch, mask_reviewed_frames
from multiview_squat_trial import validate_alignment
from resources import check_resources


def extract_inputs() -> dict:
    """직접 확정된 두 시점 구간의 Heavy와 점수 없는 관절 검토판을 만든다.

    Returns:
        dict: 자동 추출 구간·시점 수. 직접 관절 승인·등급은 만들지 않는다.

    Raises:
        ValueError: 학습 역할·원본 해시·관측 정렬 또는 디코딩 실패.
        OSError: 주석·캐시·검토판 접근 실패.
        RuntimeError: 자원 보호 기준 위반.
    """
    receipts = []
    for ann in audit.read_json(DEST / "score_annotations.json"):
        if ann["split"] != "train":
            raise ValueError("Only existing training participants permitted")
        record = dict(id=ann["id"])
        board = Image.new("RGB", (1920, 960), "white")
        draw = ImageDraw.Draw(board)
        for j, view in enumerate(("side", "front")):
            path = ann["video_path"] if view == "side" else ann["front_path"]
            sha = ann["video_sha256"] if view == "side" else ann["front_sha256"]
            if audit.digest(pathlib.Path(path)) != sha:
                raise ValueError("Frozen video changed")
            row = dict(ann, required_views=["side", "synchronized_front"])
            if view == "front":
                row["segment"] = ann["front_segment"]
            receipt = extract.extract_view(row, view, path, DEST / "heavy")
            record[view] = receipt
            with np.load(receipt["cache"]) as cache:
                sequence = cache["image_xy"]
                frames = cache["frame_numbers"]
            cap = cv2.VideoCapture(
                path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            for k, index in enumerate(
                np.rint(np.linspace(0, len(frames) - 1, 8)).astype(int)
            ):
                cap.set(1, int(frames[index]))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Joint evidence frame missing")
                for a, b in render.CONNECTIONS:
                    if np.isfinite(sequence[index, [a, b]]).all():
                        aa, bb = sequence[index, [a, b]].round().astype(int)
                        cv2.line(frame, tuple(aa), tuple(bb), (70, 250, 90), 2)
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im.thumbnail((238, 450))
                board.paste(im, (k * 240, j * 480 + 25))
                draw.text(
                    (k * 240 + 3, j * 480 + 4),
                    f"{view}: {frames[index]}",
                    fill="black",
                )
            cap.release()
        with np.load(record["side"]["cache"]) as side, np.load(
            record["front"]["cache"]
        ) as front:
            validate_alignment(
                ann["alignment"], side["frame_numbers"], front["frame_numbers"]
            )
        target = DEST / (ann["id"] + "_joints.jpg")
        board.save(target, quality=94)
        record["joint_evidence"] = str(target)
        receipts.append(record)
        audit.save_json(
            DEST / "extraction.json", dict(records=receipts, completed=False)
        )
    audit.save_json(
        DEST / "extraction.json", dict(records=receipts, completed=True)
    )
    return dict(attempts=len(receipts), views=len(receipts) * 2)


def train() -> dict:
    """동일 특징의 기존 학습에 직접 승인 오류를 더해 고정 개발·그룹 비교한다.

    Returns:
        dict: 전후 공통 고정 검증, 연결 그룹 교차검증과 원본 분할 감사.
        검증 라벨은 특징·경계 학습에 사용하지 않는다. 이전 후보는 보존한다.

    Raises:
        ValueError: 원본·참여자·구간 중복·관절 직접 승인 불일치.
        OSError: 기존 입력·캐시·모델·결과 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    confirmed = audit.read_json(DEST / "score_annotations.json")
    error_people = {
        r["person"]
        for r in confirmed
        if r["assessment"]["criterion_levels"][2] == 0
        and not r.get("heel_error_hypothesis_rejected_before_training", False)
    }
    if len(error_people) < 2:
        raise ValueError(
            "Two directly confirmed heel-error groups required; rejected hypotheses cannot train"
        )
    before, _, _ = load()
    old, _ = update_existing([r for r in before if r["exercise"] == "squats"])
    old += [r for r in added_rows() if r["exercise"] == "squats"]
    new = load_batch(DEST)
    annotations = {
        r["id"]: r for r in audit.read_json(DEST / "score_annotations.json")
    }
    receipts = {
        r["id"]: r for r in audit.read_json(DEST / "extraction.json")["records"]
    }
    reviews = {
        (r["id"], r["view"]): r
        for r in audit.read_json(DEST / "subject_tracking_review.json")[
            "records"
        ]
    }
    existing_sources = {s for r in old for s in r["sources"]}
    for row in new:
        ann = annotations[row["id"]]
        if existing_sources.intersection(row["sources"]):
            raise ValueError(
                "New trial source already present in previous squat inputs"
            )
        with np.load(receipts[row["id"]]["side"]["cache"]) as data:
            seq = mask_reviewed_frames(
                data["image_xy"],
                data["frame_numbers"],
                reviews[row["id"], "side"].get(
                    "unreliable_frame_intervals", []
                ),
            )
        cap = cv2.VideoCapture(ann["video_path"])
        seq = three_items.mask_image_border(seq, cap.get(3), cap.get(4))
        cap.release()
        update_note(row, seq)
    rows = old + new
    recovery_train.component_groups(rows)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate annotation IDs")
    fitting = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    audit.save_json(DEST / "candidate_input_manifest.json", manifest(rows))
    check_resources()
    model = fit(fitting)
    joblib.dump(model, DEST / "squats_candidate.joblib")
    fixed = [predict(model, r) for r in validation]
    oof = []
    for fit_indices, held in GroupKFold(3).split(
        fitting, groups=recovery_train.component_groups(fitting)
    ):
        check_resources()
        fold = fit([fitting[int(i)] for i in fit_indices])
        oof.extend(predict(fold, fitting[int(i)]) for i in held)
    previous = next(
        r
        for r in audit.read_json(ROUND2 / "comparison.json")["exercises"]
        if r["exercise"] == "squats"
    )
    result = dict(
        completed=True,
        added_direct_segments=len(new),
        effective_training_segments=sum(
            any(
                n["eligible"] and g is not None
                for n, g in zip(r["notes"], r["levels"])
            )
            for r in fitting
        ),
        fixed=recovery_train.summarize(fixed),
        fixed_predictions=fixed,
        group_cv=recovery_train.summarize(oof),
        group_cv_predictions=oof,
        baseline=previous["fixed"],
        baseline_predictions=previous["fixed_predictions"],
        model_sha256=audit.digest(DEST / "squats_candidate.joblib"),
        new_inputs=manifest(new),
        expert_reference=False,
        independent_final_test=False,
        promoted=False,
        external_discovery=False,
    )
    audit.save_json(DEST / "comparison.json", result)
    return result["fixed"]


if __name__ == "__main__":
    print(
        json.dumps(
            extract_inputs() if "extract" in sys.argv else train(),
            ensure_ascii=False,
        )
    )
