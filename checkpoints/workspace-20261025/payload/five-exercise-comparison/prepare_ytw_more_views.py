"""새 후보 ID의 YTW 구도 선별판을 만든다. 등급·구도 판정은 생성하지 않는다."""

import hashlib
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def run() -> None:
    """학습 분할 후보 ID 최대 48개의 5장 표본과 출처를 보존한다.

    Returns:
        None: 선별판 2장과 원본 SHA·표본 프레임 manifest를 저장한다.

    Raises:
        ValueError: 원본 메타데이터·디코딩 실패.
        RuntimeError: 자원 보호 조건 충족.
        OSError: 입력 읽기·출력 쓰기 실패.
    """
    candidates = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    labels = audit.read_json(audit.OUT / "labels.json")
    seen = {x["person"] for x in labels}
    previous = audit.read_json(audit.OUT / "ytw_new_view_candidates.json")
    seen.update(r["person"] for r in previous["records"])
    for name in (
        "w_challenge_direct_review.json",
        "w_challenge_v2_direct_review.json",
        "w_targeted_direct_review.json",
    ):
        seen.update(
            r["person"] for r in audit.read_json(audit.OUT / name)["reviews"]
        )
    rows = []
    for row in sorted(
        candidates, key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest()
    ):
        if (
            row["exercise"] != "ytw"
            or row["split"] != "train"
            or row["person"] in seen
        ):
            continue
        if not pathlib.Path(row["path"]).exists():
            continue
        seen.add(row["person"])
        rows.append(row)
        if len(rows) == 48:
            break
    records = []
    for page in range(4):
        canvas = Image.new("RGB", (1600, 12 * 190), "#18232c")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[page * 12 : (page + 1) * 12]):
            check_resources()
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            count, fps = int(cap.get(7)), cap.get(5)
            if count < 2 or fps <= 0:
                raise ValueError("Invalid candidate metadata")
            frames = np.linspace(0, count - 1, 7).round().astype(int)[1:-1]
            cap.set(1, int(frames[0]))
            decoded = int(frames[0]) - 1
            for k, frame in enumerate(frames):
                while decoded < frame:
                    if not cap.grab():
                        raise ValueError("Cannot decode candidate")
                    decoded += 1
                    if decoded % 90 == 0:
                        check_resources()
                ok, pixels = cap.retrieve()
                if not ok:
                    raise ValueError("Cannot retrieve candidate")
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image.thumbnail((280, 158))
                x, y = 200 + k * 280, j * 190
                canvas.paste(image, (x, y))
                draw.text(
                    (x, y + 160), f"f{frame} / {frame/fps:.2f}s", fill="white"
                )
            cap.release()
            draw.text(
                (4, j * 190 + 20),
                row["id"] + "\n" + row["person"],
                fill="white",
            )
            records.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    split=row["split"],
                    video_path=row["path"],
                    video_sha256=audit.digest(pathlib.Path(row["path"])),
                    sample_frames=frames.tolist(),
                    sheet=f"review/ytw_more_views_{page+1}.jpg",
                    direct_review=False,
                    score_label_created=False,
                    independent_final_test=False,
                )
            )
        canvas.save(
            audit.OUT / f"review/ytw_more_views_{page+1}.jpg", quality=95
        )
    audit.save_json(
        audit.OUT / "ytw_more_view_candidates.json",
        dict(
            records=records,
            provider_strings_used_for_selection=False,
            selection="deterministic SHA256 ID order, distinct provisional source participant IDs not in current labels; identities not independently verified",
        ),
    )


if __name__ == "__main__":
    run()
