"""오류 후보를 층화하여 시간순 원본 구도 검토판을 준비한다.

제공자 문자열은 후보 선별에만 쓰며 검토판에는 표시하지 않는다.
점수·등급·직접 검토 여부는 자동으로 만들지 않는다.
"""

import hashlib
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def run() -> dict:
    """플랭크 오류 후보 36개와 W 후보 12개의 5장 표본을 저장한다.

    Returns:
        dict: 후보 ID·원본 해시·분할·제공자 선정 층·정확한 표본 프레임.
        서로 다른 worker ID는 실제 다른 사람이라는 보장이 아니다.

    Raises:
        ValueError: 영상 메타데이터·디코딩 실패.
        OSError: 입력·검토판·선정 기록 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    used = {r["id"] for r in audit.read_json(audit.OUT / "labels.json")}
    for name in (
        "w_challenge_direct_review.json",
        "w_challenge_v2_direct_review.json",
    ):
        used.update(
            r["id"] for r in audit.read_json(audit.OUT / name)["reviews"]
        )
    selected = []
    people = set()
    strata = [
        ("plank", term, 9)
        for term in (
            "high hips",
            "butt sinking",
            "legs and hips on the floor",
            "on knees",
        )
    ] + [("ytw", "arms too low", 12)]
    for exercise, term, limit in strata:
        count = 0
        for row in sorted(
            pool,
            key=lambda r: hashlib.sha256(
                ("error-batch-v1" + r["id"]).encode()
            ).hexdigest(),
        ):
            if (
                row["exercise"] != exercise
                or row["split"] != "train"
                or row["id"] in used
                or row["person"] in people
            ):
                continue
            if (
                not any(term in s for s in row.get("source_labels", []))
                or not pathlib.Path(row["path"]).is_file()
            ):
                continue
            selected.append(dict(row, selection_stratum=term))
            used.add(row["id"])
            people.add(row["person"])
            count += 1
            if count >= limit:
                break
    records = []
    for page in range((len(selected) + 11) // 12):
        canvas = Image.new("RGB", (1600, 12 * 190), "#14212a")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(selected[page * 12 : (page + 1) * 12]):
            check_resources()
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            fps, count = cap.get(5), int(cap.get(7))
            if fps <= 0 or count < 5:
                raise ValueError("Invalid candidate video")
            frames = np.rint(
                (count - 1) * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            ).astype(int)
            cap.set(1, int(frames[0]))
            decoded = int(frames[0]) - 1
            for k, index in enumerate(frames):
                while decoded < int(index):
                    if not cap.grab():
                        raise ValueError("Candidate frame unavailable")
                    decoded += 1
                ok, pixels = cap.retrieve()
                if not ok:
                    raise ValueError("Candidate decode failed")
                im = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                im.thumbnail((280, 155))
                x, y = 200 + k * 280, j * 190
                canvas.paste(im, (x, y))
                draw.text(
                    (x + 3, y + 158), f"f{index} {index/fps:.3f}s", fill="white"
                )
            cap.release()
            draw.text(
                (5, j * 190 + 20),
                row["id"] + "\n" + row["exercise"],
                fill="white",
            )
            records.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    exercise=row["exercise"],
                    source_split="train",
                    selection_stratum=row["selection_stratum"],
                    video_path=row["path"],
                    video_sha256=audit.digest(pathlib.Path(row["path"])),
                    fps=fps,
                    total_frames=count,
                    sample_frames=frames.tolist(),
                    sheet=f"review/error_batch_{page+1}.jpg",
                    direct_review=False,
                    score_created=False,
                    identity_independently_verified=False,
                )
            )
        canvas.save(audit.OUT / f"review/error_batch_{page+1}.jpg", quality=96)
    result = dict(
        records=records,
        provider_strings_used_for_selection_only=True,
        provider_strings_displayed_on_sheets=False,
        independent_final_test=False,
        score_labels_created=0,
    )
    audit.save_json(audit.OUT / "error_batch_candidates.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
