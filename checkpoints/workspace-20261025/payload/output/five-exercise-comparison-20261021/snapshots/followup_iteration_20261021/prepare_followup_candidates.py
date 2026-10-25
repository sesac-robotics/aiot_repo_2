"""기존 참여자의 미검토 시행까지 포함해 오류·구도 후보를 추가 준비한다.

기존 자동 선별의 '새 참여자만' 제한을 자료 부족의 근거로 삼지 않는다.
학습 분할의 미검토 원본만 선택하며 정상·오류 문자열은 선정에만 쓴다.
"""

import hashlib
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def collect_ids(value: object) -> set[str]:
    """중첩 감사 JSON에서 영상 ID를 수집한다.

    Args:
        value: 사전·목록 또는 JSON 단일 값. 단위 없음.

    Returns:
        set[str]: 문자열 id 키의 집합. 참여자 ID와 혼합하지 않는다.
    """
    ids = set()
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            ids.add(value["id"])
        for item in value.values():
            ids.update(collect_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(collect_ids(item))
    return ids


def run() -> dict:
    """플랭크 24개·YTW 18개의 서로 다른 참여자 원본 표본을 준비한다.

    Returns:
        dict: 원본별 세 시점 표본·해시·기존 노출 여부와 검토판 경로.
        새 원본이어도 기존 참여자일 수 있다. 직접 검토·점수는 생성하지 않는다.

    Raises:
        OSError: 목록·영상·이미지 읽기 또는 쓰기 실패.
        ValueError: 디코딩 실패 또는 기존 평가 분할 참여자 중복.
        RuntimeError: 자원 보호 조건 위반.
    """
    target = audit.OUT / "followup_candidate_preparation.json"
    if target.exists():
        return audit.read_json(target)
    known = set()
    for path in audit.OUT.glob("*.json"):
        if (
            "candidate" in path.name
            or "direct" in path.name
            or path.name in ("labels.json", "new_screening_decisions.json")
        ):
            known.update(collect_ids(audit.read_json(path)))
    for path in audit.OUT.glob("plank_auto_screen*/results.json"):
        known.update(collect_ids(audit.read_json(path)))
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    held_people = {r["person"] for r in pool if r["split"] != "train"}
    previous_people = {r["person"] for r in pool if r["id"] in known}
    selected = []
    for exercise, limit, terms in [
        (
            "plank",
            24,
            [
                "legs and hips on the floor",
                "on knees",
                "butt sinking",
                "high hips",
                "stopping early",
                "shaking",
            ],
        ),
        (
            "ytw",
            18,
            [
                "arms too low",
                "hands flat",
                "not low enough",
                "thumbs pointed down",
                "too fast",
                "bent legs",
            ],
        ),
    ]:
        eligible = [
            r
            for r in pool
            if r["exercise"] == exercise
            and r["split"] == "train"
            and r["person"] not in held_people
            and r["id"] not in known
            and pathlib.Path(r["path"]).is_file()
        ]
        eligible.sort(
            key=lambda r: hashlib.sha256(
                ("followup-20261021" + r["id"]).encode()
            ).hexdigest()
        )
        people = set()
        for index in range(limit):
            term = terms[index % len(terms)]
            row = next(
                (
                    r
                    for r in eligible
                    if r["person"] not in people
                    and any(term in s for s in r["source_labels"])
                ),
                None,
            )
            if row is None:
                break
            people.add(row["person"])
            selected.append(row)
    records = []
    for page in range((len(selected) + 5) // 6):
        canvas = Image.new("RGB", (1320, 1260), "#14212c")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(selected[page * 6 : page * 6 + 6]):
            check_resources()
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            try:
                count, fps = int(cap.get(7)), cap.get(5)
                if count < 3 or fps <= 0:
                    raise ValueError("Invalid candidate video")
                indices = np.rint(
                    (count - 1) * np.array([0.2, 0.5, 0.8])
                ).astype(int)
                y = offset * 210
                draw.text(
                    (5, y + 5),
                    row["id"] + " " + row["person"] + " " + row["exercise"],
                    fill="white",
                )
                for k, frame in enumerate(indices):
                    cap.set(1, int(frame))
                    ok, pixels = cap.read()
                    if not ok:
                        raise ValueError("Candidate decoding failed")
                    image = Image.fromarray(
                        cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                    )
                    image.thumbnail((440, 160))
                    canvas.paste(image, (440 * k, y + 23))
                    draw.text(
                        (440 * k, y + 185),
                        f"f{frame} / {frame/fps:.2f}s",
                        fill="white",
                    )
            finally:
                cap.release()
            records.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    exercise=row["exercise"],
                    split="train",
                    video_path=row["path"],
                    video_sha256=audit.digest(pathlib.Path(row["path"])),
                    fps=fps,
                    total_frames=count,
                    sample_frames=indices.tolist(),
                    prior_person_seen=row["person"] in previous_people,
                    evidence_image=str(
                        audit.OUT / f"review/followup_candidates_{page+1}.jpg"
                    ),
                    direct_review=False,
                    score_labels_created=0,
                )
            )
        canvas.save(
            audit.OUT / f"review/followup_candidates_{page+1}.jpg", quality=95
        )
    report = dict(
        records=records,
        automatically_prepared=len(records),
        direct_reviews=0,
        labels_created=0,
        original_split_preserved=True,
        independent_final_test=False,
    )
    audit.save_json(target, report)
    return report


if __name__ == "__main__":
    print(run()["automatically_prepared"])
