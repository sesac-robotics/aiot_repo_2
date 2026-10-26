"""보유된 학습 원본의 미사용 유지 구간을 중복 없이 검토 대기열로 만든다.

새 원본·참여자를 확보한 것으로 세지 않는다. 인접 유지 구간은 같은
참여자·원본 그룹에 남으며 이후 학습에서 그룹별 가중치를 적용한다.
"""

import json
import pathlib

import cv2
from PIL import Image, ImageDraw

import audit
import external_review
from held_round2 import OUT


def prepare() -> dict:
    """기존 원본 네 개의 3.2초 구간과 시간순 검토판을 준비한다.

    Returns:
        dict: 자동 준비한 구간·원본·메타 그룹 수. 직접 라벨·추적 승인·
        학습 적격 수는 생성하지 않으며 기존 시험·검증 원본은 사용하지 않는다.

    Raises:
        ValueError: 기존 학습 역할·원본·시간 범위 또는 중복 구간 위반.
        OSError: 입력 원본·기존 주석·검토판 접근 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    root = audit.OUT / "sources"
    specs = [
        (
            "navy773312",
            "dvids_773312.mp4",
            "dvids_773312_single_performer",
            [*range(43, 99, 4), 107, 111, 115, *range(131, 207, 4)],
        ),
        (
            "navy805472",
            "navy_805472.mp4",
            "dvids_navy805472_performer",
            list(range(26, 119, 6)),
        ),
        (
            "chris",
            "youtube_3UNXPzP0QaY.mp4",
            "youtube_chris_ellerby_hemmings",
            [11, 15, 19, 23, 33, 37, 41, 45],
        ),
        (
            "avid",
            "youtube_WEIcNQqcZqY.mp4",
            "youtube_channel_UC7_LFnyBTIbN9e6p98YgIgw",
            [16.5, 20.5, 24.5, 28.5, 32],
        ),
    ]
    existing = audit.read_json(
        OUT.parent / "allfive_round1/candidate_input_manifest.json"
    )
    occupied = {}
    for r in audit.read_json(audit.OUT / "development_manifest.json"):
        if r.get("exercise") == "plank":
            occupied.setdefault(r["video_sha256"], []).append(r["segment"])
    for r in audit.read_json(
        OUT.parent / "allfive_round1/score_annotations.json"
    ):
        if r["exercise"] == "plank":
            occupied.setdefault(r["video_sha256"], []).append(r["segment"])
    for r in audit.read_json(audit.OUT / "youtube_segment_plan.json")[
        "records"
    ]:
        if r["id"] == "youtube_chris_basic":
            occupied.setdefault(r["source_sha256"], []).append(
                dict(
                    start_frame=r["start_frame"],
                    end_frame=r["end_frame_exclusive"] - 1,
                )
            )
    for r in audit.read_json(audit.OUT / "plank_side_export_plan.json")[
        "records"
    ]:
        occupied.setdefault(r["source_sha256"], []).append(
            dict(
                start_frame=r["start_frame"],
                end_frame=r["end_frame_exclusive"] - 1,
            )
        )
    (OUT / "held_plank").mkdir(exist_ok=True)
    target = OUT / "held_plank_preparation.json"
    ready = audit.read_json(target)["records"] if target.exists() else []
    done = {r["id"] for r in ready}
    skipped = []
    for source_id, filename, person, starts in specs:
        path = root / filename
        sha = audit.digest(path)
        matching = [r for r in existing if sha in r["sources"]]
        if not matching or any(r["split"] != "train" for r in matching):
            raise ValueError("Held source is not exclusively existing train")
        cap = cv2.VideoCapture(str(path))
        fps, count = cap.get(5), int(cap.get(7))
        cap.release()
        added = []
        for seconds in starts:
            first, last = round(seconds * fps), round((seconds + 3.2) * fps)
            identifier = f"held2_{source_id}_{first}_{last}"
            if identifier in done:
                continue
            conflicts = occupied.get(sha, []) + added
            if any(
                not (last < s["start_frame"] or first > s["end_frame"])
                for s in conflicts
            ):
                skipped.append(
                    dict(id=identifier, reason="existing_interval_overlap")
                )
                continue
            if not 0 <= first < last < count:
                raise ValueError("Held source interval outside original")
            receipt = external_review.contact_sheet(
                path,
                OUT / "held_plank" / (identifier + ".jpg"),
                first / fps,
                last / fps,
                12,
            )
            segment = dict(start_frame=first, end_frame=last, fps=fps)
            added.append(segment)
            ready.append(
                dict(
                    id=identifier,
                    path=str(path),
                    sha256=sha,
                    person=person,
                    exercise="plank",
                    split="train",
                    fps=fps,
                    segment=segment,
                    preparation=receipt,
                    evidence=receipt["evidence_image"],
                    new_original=False,
                    new_participant=False,
                    direct_reviewed=False,
                )
            )
            audit.save_json(target, dict(records=ready, completed=False))
    pages = []
    for source_id, filename, person, starts in specs:
        rows = [r for r in ready if f"held2_{source_id}_" in r["id"]]
        for start in range(0, len(rows), 4):
            selected = rows[start : start + 4]
            canvas = Image.new("RGB", (1536, 492 * len(selected)), "#14212a")
            draw = ImageDraw.Draw(canvas)
            for i, r in enumerate(selected):
                im = Image.open(r["evidence"])
                canvas.paste(im, (0, i * 492 + 20))
                draw.text((3, i * 492 + 3), r["id"], fill="white")
            page = OUT / "held_plank" / f"{source_id}_page_{start//4+1:02d}.jpg"
            canvas.save(page, quality=95)
            pages.append(dict(path=str(page), ids=[r["id"] for r in selected]))
    audit.save_json(
        target,
        dict(
            records=ready,
            pages=pages,
            skipped=skipped,
            completed=True,
            automatic_preparation_not_direct_review=True,
            original_sources=4,
            new_original_sources=0,
            new_participants=0,
            near_duplicate_motion_audit_required=True,
            independent_final_test=False,
            external_discovery=False,
        ),
    )
    return dict(
        prepared=len(ready), skipped=len(skipped), direct_labels_created=0
    )


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
