"""보유 미검토 후보의 직접 라벨과 Heavy 관절을 별도 실험에 연결한다."""

import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import external_review
import extract
import render
from held_round2 import OUT


def dense() -> dict:
    """미검토 운동 후보와 플랭크 중단 후보의 확대 표본을 준비한다.

    Returns:
        dict: 자동 준비 수. 직접 등급·학습 적격 판정은 생성하지 않는다.

    Raises:
        ValueError: 원본·시간 범위·디코딩 계약 위반.
        OSError: 후보 입력 또는 근거 이미지 저장 실패.
    """
    screen = audit.read_json(OUT / "screening_manifest.json")
    if not screen["completed"]:
        raise ValueError("Finish screening before dense preparation")
    review = audit.read_json(OUT / "plank_screen_direct_review.json")
    selected = {
        r["id"]
        for r in review["records"]
        if r["upper_support_interruption_candidate"]
    }
    (OUT / "dense").mkdir(exist_ok=True)
    ready = (
        audit.read_json(OUT / "dense_preparation.json")["records"]
        if (OUT / "dense_preparation.json").exists()
        else []
    )
    known = {r["id"] for r in ready}
    for r in screen["records"]:
        if r["exercise"] == "plank" and r["id"] not in selected:
            continue
        if r["id"] in known:
            continue
        receipt = external_review.contact_sheet(
            pathlib.Path(r["path"]),
            OUT / "dense" / (r["id"] + ".jpg"),
            0,
            None,
            24,
        )
        ready.append(dict(r, preparation=receipt))
        audit.save_json(
            OUT / "dense_preparation.json",
            dict(records=ready, completed=False, direct_labels_created=0),
        )
    audit.save_json(
        OUT / "dense_preparation.json",
        dict(records=ready, completed=True, direct_labels_created=0),
    )
    return dict(prepared=len(ready), direct_labels_created=0)


def extract_selected() -> dict:
    """직접 동결한 구간을 Heavy로 추출하고 추적 검토판을 저장한다.

    Returns:
        dict: 캐시·모델 해시와 원본 프레임별 관절 표본 연결 목록. 관절
        자동 추출은 직접 추적 검토 또는 점수 검토를 뜻하지 않는다.

    Raises:
        ValueError: 직접 주석과 원본 해시·구간·학습 역할 불일치.
        OSError: 주석·캐시·이미지 접근 실패.
        RuntimeError: GPU·자원 보호 또는 추출 실패.
    """
    decisions = audit.read_json(OUT / "manual_decisions.json")
    previous = (
        {
            r["id"]: r
            for r in audit.read_json(OUT / "extraction.json")["records"]
        }
        if (OUT / "extraction.json").exists()
        else {}
    )
    sources = {
        r["id"]: r
        for r in audit.read_json(OUT / "dense_preparation.json")["records"]
    }
    if (OUT / "held_plank_preparation.json").exists():
        sources.update(
            {
                r["id"]: r
                for r in audit.read_json(OUT / "held_plank_preparation.json")[
                    "records"
                ]
            }
        )
    records = []
    (OUT / "joints").mkdir(exist_ok=True)
    for d in decisions["records"]:
        source = sources[d["id"]]
        if source["split"] != "train":
            raise ValueError("Only existing training source is permitted")
        row = dict(
            id=d["id"] + "_held2_score",
            exercise=source["exercise"],
            video_sha256=source["sha256"],
            required_views=d["views"],
            segment=dict(
                start_frame=d["frames"][0],
                end_frame=d["frames"][1],
                fps=source["fps"],
            ),
        )
        receipt = extract.extract_view(
            row, "primary", source["path"], destination=OUT / "heavy"
        )
        old = previous.get(d["id"])
        if (
            old
            and old["extraction"]["fingerprint"] == receipt["fingerprint"]
            and pathlib.Path(old["joint_evidence"]).exists()
        ):
            records.append(dict(old, extraction=receipt))
            audit.save_json(
                OUT / "extraction.json", dict(records=records, completed=False)
            )
            continue
        with np.load(receipt["cache"]) as data:
            seq, frames = data["image_xy"], data["frame_numbers"]
        cap = cv2.VideoCapture(
            source["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        indices = np.unique(
            np.linspace(0, len(frames) - 1, 12).round().astype(int)
        )
        sheet = Image.new("RGB", (1536, 500), "#14212a")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (5, 5), row["id"] + " | AUTOMATIC JOINT SAMPLES", fill="white"
        )
        for i, k in enumerate(indices):
            cap.set(1, int(frames[k]))
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Joint review frame could not be decoded")
            # 기존 표시 함수는 픽셀 관절만 그리며 점수·라벨은 입력하지 않는다.
            for a, b in render.CONNECTIONS:
                if np.isfinite(seq[k, [a, b]]).all():
                    pa, pb = seq[k, [a, b]].round().astype(int)
                    cv2.line(frame, tuple(pa), tuple(pb), (70, 250, 90), 2)
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((256, 200))
            x, y = i % 6 * 256, i // 6 * 230 + 30
            sheet.paste(im, (x + (256 - im.width) // 2, y))
            draw.text((x + 4, y + 202), f"f{frames[k]}", fill="white")
        cap.release()
        evidence = OUT / "joints" / (row["id"] + ".jpg")
        sheet.save(evidence, quality=95)
        records.append(
            dict(id=d["id"], extraction=receipt, joint_evidence=str(evidence))
        )
        audit.save_json(
            OUT / "extraction.json", dict(records=records, completed=False)
        )
    audit.save_json(
        OUT / "extraction.json", dict(records=records, completed=True)
    )
    return dict(extracted=len(records), direct_tracking_reviews_created=0)


if __name__ == "__main__":
    print(
        json.dumps(
            extract_selected() if "extract" in sys.argv else dense(),
            ensure_ascii=False,
        )
    )
