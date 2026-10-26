"""미검토 참여자의 Y–T–W 개발 검증 구도·시간순 장면을 준비한다.

기존 제공자 validation 배정은 유지하며 최종 시험 자료는 선정하지 않는다.
자동 등급·점수는 만들지 않고 직접 검토를 위한 36장 표본만 저장한다.
"""

import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit


def prepare(
    exercise: str = "ytw", limit: int = 4, split: str = "validation"
) -> list[dict]:
    """선택 운동의 개발 검증 클립 첫 12초에서 장면 검토판을 저장한다.

    Args:
        exercise: ytw 또는 shoulder 영어 코드.
        limit: 준비할 최대 클립 수. 숄더 프레스는 같은 검증 참여자의
            서로 다른 수행도 같은 validation 분할에 유지한다.
        split: 기존 train 또는 validation 배정. 서로 이동하지 않는다.

    Returns:
        list[dict]: 원본·구간·참여자·근거 이미지와 미검토 상태.

    Raises:
        OSError: 입력 읽기 또는 결과 저장 실패.
        ValueError: 영상 메타데이터 또는 디코딩 실패.
    """
    if (
        exercise not in ("ytw", "shoulder")
        or limit < 1
        or split not in ("train", "validation")
    ):
        raise ValueError("Invalid validation preparation request")
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    known = set()
    train_people = set()
    for path in audit.SOURCE.parent.rglob("*.json"):
        if path.name not in (
            "labels.json",
            "candidate_manifest.json",
            "screen_manifest.json",
        ):
            continue
        data = audit.read_json(path)
        if isinstance(data, list):
            known.update(
                r.get("person")
                for r in data
                if isinstance(r, dict)
                and (exercise == "ytw" or r.get("exercise") == exercise)
            )
            train_people.update(
                r.get("person")
                for r in data
                if isinstance(r, dict) and r.get("split") == "train"
            )
    choices = [
        r
        for r in pool
        if r["exercise"] == exercise
        and r["split"] == split
        and r["person"] not in known
        and (split != "validation" or r["person"] not in train_people)
        and (exercise != "shoulder" or r["id"].endswith("_front"))
    ]
    np.random.default_rng(212026).shuffle(choices)
    selected = []
    for row in choices:
        if row["person"] in known:
            continue
        if exercise == "ytw":
            known.add(row["person"])
        capture = cv2.VideoCapture(
            row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
        )
        canvas = Image.new("RGB", (1536, 6 * 218 + 30), "#14212a")
        draw = ImageDraw.Draw(canvas)
        number = len(selected) + 1
        draw.text(
            (8, 5),
            f"{exercise.upper()} DEV {split.upper()} {number}",
            fill="white",
        )
        try:
            fps = capture.get(5)
            count = int(capture.get(7))
            if fps <= 0 or count < 36:
                raise ValueError("Invalid validation clip metadata")
            last = min(count - 1, round(12 * fps))
            indices = np.linspace(0, last, 36).round().astype(int)
            for j, index in enumerate(indices):
                capture.set(1, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Validation clip decoding failed")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((256, 190))
                x, y = j % 6 * 256, 30 + j // 6 * 218
                canvas.paste(image, (x + (256 - image.width) // 2, y))
                draw.text(
                    (x + 3, y + 195), f"f{index} {index/fps:.2f}s", fill="white"
                )
        finally:
            capture.release()
        sheet = audit.OUT / "review" / f"{exercise}_{split}_{number:02d}.jpg"
        canvas.save(sheet, quality=96)
        selected.append(
            dict(
                id=row["id"],
                person=row["person"],
                exercise=exercise,
                video_path=row["path"],
                video_sha256=audit.digest(pathlib.Path(row["path"])),
                split=split,
                fps=fps,
                start_frame=0,
                end_frame=last,
                evidence_image=str(sheet),
                status="prepared_not_directly_reviewed",
                provider_label_not_used_as_score=True,
            )
        )
        if len(selected) == limit:
            break
    audit.save_json(audit.OUT / f"{exercise}_{split}_queue.json", selected)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exercise", choices=["ytw", "shoulder"], default="ytw"
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    args = parser.parse_args()
    print(len(prepare(args.exercise, args.limit, args.split)))
