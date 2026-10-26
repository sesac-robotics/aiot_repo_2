"""미검토 QEVD 학습 후보의 구도 검토판을 생성한다.

제공자 오류 문자열은 선정 층화에만 사용하며 점수·등급은 만들지 않는다.
출력 검토판은 직접 읽기 전까지 자동 준비 자료로만 집계한다.
"""

import json
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit


def prepare(limit: int = 16) -> list[dict]:
    """아직 검토 기록이 없는 참여자의 플랭크 구도 자료를 저장한다.

    Args:
        limit: 준비할 최대 영상 수. 각 영상은 다섯 시점으로 표본화한다.

    Returns:
        list[dict]: 원본·참여자·분할·검토판·미검토 상태 메타데이터.

    Raises:
        ValueError: limit가 양수가 아니거나 영상 디코딩에 실패한 경우.
        OSError: 목록 읽기 또는 결과 저장 실패.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    used_people = set()
    used_ids = set()
    parent = audit.SOURCE.parent
    # 지난 단계에서 보았던 사람도 새 미검토 참여자로 세지 않는다.
    for path in parent.rglob("*.json"):
        if path.name not in (
            "labels.json",
            "screen_manifest.json",
            "screening_decisions.json",
            "candidate_manifest.json",
        ):
            continue
        value = audit.read_json(path)
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    used_people.add(row.get("person"))
                    used_ids.add(row.get("id"))
    choices = [
        row
        for row in pool
        if (
            row["exercise"] == "plank"
            and row["split"] == "train"
            and row["person"] not in used_people
            and row["id"] not in used_ids
            and row["label_source"].startswith("QEVD")
        )
    ]
    rng = np.random.default_rng(21102026)
    rng.shuffle(choices)
    selected = []
    folder = audit.OUT / "review"
    folder.mkdir(parents=True, exist_ok=True)
    for row in choices:
        if row["person"] in used_people:
            continue
        used_people.add(row["person"])
        number = len(selected) + 1
        capture = cv2.VideoCapture(
            row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
        )
        images = []
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count < 5:
                raise ValueError("Invalid video metadata")
            indices = np.linspace(0, count - 1, 5).round().astype(int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Video frame decoding failed")
                images.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
        finally:
            capture.release()
        canvas = Image.new("RGB", (1600, 290), "#14212a")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"PLANK SCREEN {number:02d} | TRAIN", fill="white")
        for position, (index, image) in enumerate(zip(indices, images)):
            image.thumbnail((320, 240))
            x = 320 * position
            canvas.paste(image, (x + (320 - image.width) // 2, 25))
            draw.text((x + 5, 270), f"f{index} {index/fps:.2f}s", fill="white")
        sheet = folder / f"plank_{number:02d}.jpg"
        canvas.save(sheet, quality=94)
        selected.append(
            dict(
                id=row["id"],
                person=row["person"],
                exercise="plank",
                original_split=row["split"],
                video_path=row["path"],
                video_sha256=audit.digest(pathlib.Path(row["path"])),
                fps=fps,
                total_frames=count,
                sample_frames=[int(index) for index in indices],
                evidence_image=str(sheet),
                status="automatically_prepared_not_visually_reviewed",
                score=None,
                criterion_levels=None,
            )
        )
        if len(selected) == limit:
            break
    audit.save_json(audit.OUT / "new_screening_queue.json", selected)
    for start in range(0, len(selected), 4):
        canvas = Image.new("RGB", (1600, 1160), "#14212a")
        for offset, row in enumerate(selected[start : start + 4]):
            with Image.open(row["evidence_image"]) as image:
                canvas.paste(image, (0, offset * 290))
        canvas.save(folder / f"overview_{start//4+1:02d}.jpg", quality=94)
    return selected


if __name__ == "__main__":
    print(json.dumps({"prepared": len(prepare())}, ensure_ascii=False))
