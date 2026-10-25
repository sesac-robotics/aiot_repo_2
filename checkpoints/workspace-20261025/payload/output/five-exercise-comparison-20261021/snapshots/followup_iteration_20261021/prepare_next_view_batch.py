"""미검토 참여자 후보의 시간순 구도 표본을 준비한다.

제공자 문자열은 후보 선별에만 사용한다. 점수·구도·직접 검토 판정은
생성하지 않으며 기존 학습 분할만 사용한다.
"""

import hashlib
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def run() -> dict:
    """종목 내 미검토 토큰 최대 24개의 5장 구도 표본을 저장한다.

    Returns:
        dict: 종목별 최대 12개 후보의 원본, FPS, 프레임,
        기존 검토 manifest와 대조한 토큰 목록. 실제 새 인물 수는 미확정.

    Raises:
        OSError: 입력·출력 접근 실패.
        ValueError: 후보 부족 또는 원본 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    seen = set()
    seen_by_exercise = {"plank": set(), "ytw": set()}
    manifests = [
        "labels.json",
        "new_screening_decisions.json",
        "ytw_new_view_candidates.json",
        "ytw_more_view_candidates.json",
        "error_batch_candidates.json",
        "w_challenge_direct_review.json",
        "w_targeted_direct_review.json",
    ]

    def collect(value: object, context: str | None = None) -> None:
        """기존 manifest의 참여자 토큰을 재귀 수집한다."""
        if isinstance(value, dict):
            exercise = value.get("exercise", context)
            for key, item in value.items():
                if key in ("person", "person_token") and isinstance(item, str):
                    seen.add(item)
                    if exercise in seen_by_exercise:
                        seen_by_exercise[exercise].add(item)
                collect(item, exercise)
        elif isinstance(value, list):
            for item in value:
                collect(item, context)

    for name in manifests:
        path = audit.OUT / name
        if path.exists():
            context = "ytw" if name.startswith(("ytw_", "w_")) else None
            collect(audit.read_json(path), context)
    excluded = sorted(seen)
    candidates = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    selected = []
    selected_people = set()
    for exercise in ("plank", "ytw"):
        eligible = [
            r
            for r in candidates
            if r["exercise"] == exercise
            and r["split"] == "train"
            and r["person"] not in seen_by_exercise[exercise]
            and pathlib.Path(r["path"]).exists()
        ]
        # 오류 후보를 찾기 위한 메타데이터 필터이며 점수 라벨이 아니다.
        if exercise == "plank":
            eligible = [
                r
                for r in eligible
                if any(
                    "high hips" in s or "butt sinking" in s
                    for s in r["source_labels"]
                )
            ]
        eligible.sort(
            key=lambda r: hashlib.sha256(
                ("next-view-v1" + r["id"]).encode()
            ).hexdigest()
        )
        count = 0
        for row in eligible:
            if row["person"] in selected_people:
                continue
            selected_people.add(row["person"])
            selected.append(row)
            count += 1
            if count == 12:
                break
        if count == 0:
            raise ValueError("No unseen exercise participant tokens")
    records = []
    for page in range((len(selected) + 5) // 6):
        canvas = Image.new("RGB", (1500, 6 * 195 + 30), "#14212c")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8), "NEW VIEW CANDIDATES - NOT SCORE LABELS", fill="white"
        )
        for j, row in enumerate(selected[page * 6 : (page + 1) * 6]):
            check_resources()
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            count, fps = int(cap.get(7)), cap.get(5)
            if count < 2 or fps <= 0:
                raise ValueError("Invalid video metadata")
            frames = np.linspace(0, count - 1, 7).round().astype(int)[1:-1]
            for k, frame in enumerate(frames):
                cap.set(1, int(frame))
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError("Candidate decode failed")
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image.thumbnail((255, 155))
                x, y = 205 + k * 255, 30 + j * 195
                canvas.paste(image, (x, y))
                draw.text(
                    (x, y + 157), f"f{frame} {frame/fps:.2f}s", fill="white"
                )
            cap.release()
            draw.text(
                (5, 45 + j * 195),
                row["id"] + "\n" + row["person"] + "\n" + row["exercise"],
                fill="white",
            )
            records.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    exercise=row["exercise"],
                    split="train",
                    video_path=row["path"],
                    video_sha256=audit.digest(pathlib.Path(row["path"])),
                    fps=fps,
                    sample_frames=frames.tolist(),
                    sheet=f"review/next_views_{page+1}.jpg",
                    direct_review=False,
                    score_labels_created=0,
                    source_condition_used_for_selection_only=True,
                    independent_final_test=False,
                    person_seen_elsewhere_in_project=row["person"] in seen,
                )
            )
        canvas.save(audit.OUT / f"review/next_views_{page+1}.jpg", quality=95)
    result = dict(
        records=records,
        excluded_existing_person_tokens=excluded,
        excluded_same_exercise_person_tokens={
            k: sorted(v) for k, v in seen_by_exercise.items()
        },
        exclusion_manifests=manifests,
        actual_person_identity_verified=False,
        all_new_records_are_development_candidates=True,
    )
    audit.save_json(audit.OUT / "next_view_candidates.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
