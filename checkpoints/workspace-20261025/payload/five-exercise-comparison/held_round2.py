"""보유 원본의 미검토 대기열과 시간순 구도 검토판을 재개 가능하게 만든다.

제공자 문자열은 검토 순서에만 쓰며 점수 라벨을 생성하지 않는다.
외부 검색·다운로드와 기존 라벨·모델의 변경은 하지 않는다.
"""

import collections
import hashlib
import json
import pathlib
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from allfive_round1 import known_ids
from resources import check_resources

OUT = audit.BASE / "output/five-exercise-recovery-20261022/held_round2"


def prepare() -> dict:
    """기존 학습 역할에서 미검토 원본을 계층 선별하고 표본판을 저장한다.

    Returns:
        dict: 후보·메타 그룹·자동 준비 수. 학습 가능 수나 직접 검토 수가
        아니며 표본 장면은 각 원본의 10~90%에서 여섯 장이다.

    Raises:
        OSError: 보유 입력·결과 파일 접근 실패.
        ValueError: 원본 메타데이터 또는 프레임 디코딩 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    (OUT / "screen").mkdir(exist_ok=True)
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    known = known_ids()
    for folder in (
        OUT.parent / "allfive_round1",
        audit.BASE / "output/label-review-20261021/all-five-v2",
    ):
        for path in folder.glob("*.json"):
            if any(t in path.name for t in ("manual", "labels", "annotation")):
                known.update(re.findall(r"qevd_\d{8}", path.read_text()))
    prior = audit.read_json(OUT.parent / "allfive_round1/preparation.json")
    known.update(r["id"] for r in prior["records"])
    effective = audit.read_json(
        OUT.parent / "allfive_round1/candidate_input_manifest.json"
    )
    existing_roles = {r["person"]: r["split"] for r in effective}
    selections = []
    for exercise, limit in (
        ("plank", 64),
        ("pushups", 8),
        ("squats", 8),
        ("ytw", 8),
        ("shoulder", 8),
    ):
        candidates = [
            r
            for r in pool
            if r["exercise"] == exercise
            and r["split"] == "train"
            and r["id"] not in known
            and existing_roles.get(r["person"], "train") == "train"
            and (
                r["id"].startswith("qevd_")
                if exercise != "shoulder"
                else r["id"].endswith("_front")
            )
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(
                ("held-round2" + r["id"]).encode()
            ).hexdigest()
        )
        buckets = collections.defaultdict(list)
        for r in candidates:
            buckets[tuple(r["source_labels"])].append(r)
        used = set()
        chosen = []
        while len(chosen) < limit:
            changed = False
            for label in sorted(buckets):
                options = buckets[label]
                row = next(
                    (r for r in options if r["person"] not in used), None
                )
                if row is None:
                    continue
                options.remove(row)
                used.add(row["person"])
                chosen.append(row)
                changed = True
                if len(chosen) == limit:
                    break
            if not changed:
                break
        selections.extend(chosen)
    manifest_path = OUT / "screening_manifest.json"
    manifest = (
        audit.read_json(manifest_path)
        if manifest_path.exists()
        else {
            "records": [],
            "completed": False,
            "automatic_preparation_not_direct_review": True,
            "external_discovery": False,
        }
    )
    ready = {r["id"]: r for r in manifest["records"]}
    for row in selections:
        if row["id"] in ready:
            continue
        check_resources()
        cap = cv2.VideoCapture(
            row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        count, fps = int(cap.get(7)), cap.get(5)
        if count < 6 or fps <= 0:
            raise ValueError("Invalid held video metadata")
        frames = np.linspace(0, count - 1, 6).round().astype(int)
        canvas = Image.new("RGB", (1152, 190), "#14212a")
        draw = ImageDraw.Draw(canvas)
        for i, frame in enumerate(frames):
            cap.set(1, int(frame))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("Held video frame decoding failed")
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image.thumbnail((192, 144))
            canvas.paste(image, (i * 192 + (192 - image.width) // 2, 23))
            draw.text(
                (i * 192 + 2, 169), f"f{frame} {frame/fps:.2f}s", fill="white"
            )
        cap.release()
        draw.text((3, 3), row["id"] + " | " + row["person"], fill="white")
        evidence = OUT / "screen" / (row["id"] + ".jpg")
        canvas.save(evidence, quality=94)
        ready[row["id"]] = dict(
            row,
            frame_count=count,
            fps=fps,
            sample_frames=[int(x) for x in frames],
            evidence=str(evidence),
            sha256=audit.digest(pathlib.Path(row["path"])),
            direct_reviewed=False,
        )
        manifest["records"] = list(ready.values())
        audit.save_json(manifest_path, manifest)
    pages = []
    for exercise in ("plank", "pushups", "squats", "ytw", "shoulder"):
        rows = [r for r in ready.values() if r["exercise"] == exercise]
        for start in range(0, len(rows), 8):
            page = Image.new("RGB", (1152, 190 * len(rows[start : start + 8])))
            for i, r in enumerate(rows[start : start + 8]):
                page.paste(Image.open(r["evidence"]), (0, i * 190))
            path = OUT / f"{exercise}_screen_{start//8+1:02d}.jpg"
            page.save(path, quality=95)
            pages.append(
                dict(
                    path=str(path),
                    ids=[r["id"] for r in rows[start : start + 8]],
                )
            )
    manifest.update(completed=True, pages=pages)
    audit.save_json(manifest_path, manifest)
    return dict(prepared=len(ready), direct_labels_created=0, pages=pages)


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
