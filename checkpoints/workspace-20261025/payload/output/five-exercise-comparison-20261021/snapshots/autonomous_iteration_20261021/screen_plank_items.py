"""미검토 QEVD 플랭크의 항목 관측 후보를 Heavy로 자동 선별한다.

세 시점의 관절 가시성과 화면상 수평 몸통 길이는 구도 후보 순위일 뿐,
직접 영상 검토·자세 점수·정상 오류 판정이 아니다. 학습·시험은 하지 않는다.
"""

import argparse
import collections
import pathlib

import cv2
import mediapipe as mp
import numpy as np

import audit
import extract


def screen(limit: int = 60, include_known_people: bool = False) -> list[dict]:
    """기존에 검토하지 않은 학습 참여자별 한 영상을 자동 선별한다.

    Args:
        limit: 최대 후보 수. 점수 라벨은 생성하지 않는다.
        include_known_people: 기존 학습 참여자의 미검토 클립도 허용한다.
            이 경우 새 독립 참여자라고 표시하지 않는다.

    Returns:
        list[dict]: 구도 대리값·가시성·원본 해시·자동 선별 상태.

    Raises:
        ValueError: 후보 수·영상 메타데이터·디코딩이 잘못된 경우.
        RuntimeError: Heavy GPU 또는 자원 보호 검사 실패.
        OSError: 메타데이터·원본 읽기 또는 결과 저장 실패.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    known = set()
    known_ids = set()
    for parent in (audit.SOURCE.parent, audit.OUT):
        for path in parent.rglob("*.json"):
            if path.name not in (
                "labels.json",
                "screen_manifest.json",
                "new_screening_decisions.json",
            ):
                continue
            data = audit.read_json(path)
            if isinstance(data, list):
                known.update(
                    r.get("person") for r in data if isinstance(r, dict)
                )
                known_ids.update(
                    r.get("id") for r in data if isinstance(r, dict)
                )
    previous_path = audit.OUT / "plank_auto_screen/results.json"
    if previous_path.exists():
        known_ids.update(r["id"] for r in audit.read_json(previous_path))
    for previous in audit.OUT.glob("plank_auto_screen*/results.json"):
        known_ids.update(r["id"] for r in audit.read_json(previous))
    own = audit.OUT / "plank_auto_screen_items/results.json"
    if own.exists():
        known_ids.update(r["id"] for r in audit.read_json(own))
    previous_results = audit.read_json(own) if own.exists() else []
    remaining = max(0, limit - len(previous_results))
    if not remaining:
        return previous_results
    candidates = [
        row
        for row in pool
        if row["exercise"] == "plank"
        and row["split"] == "train"
        and (include_known_people or row["person"] not in known)
        and row["id"] not in known_ids
        and row["label_source"].startswith("QEVD")
    ]
    np.random.default_rng(212127).shuffle(candidates)
    # 제공자 문자열은 후보의 다양성 선별에만 사용하며 점수로 변환하지 않는다.
    wanted = [
        "stopping early",
        "legs and hips on the floor",
        "butt sinking",
        "high hips",
        "average",
    ]
    candidates.sort(
        key=lambda r: min(
            [
                i
                for i, term in enumerate(wanted)
                if any(term in label for label in r["source_labels"])
            ]
            or [99]
        )
    )
    selected = []
    selected_people = {r["person"] for r in previous_results}
    for row in candidates:
        if row["person"] in selected_people:
            continue
        selected_people.add(row["person"])
        selected.append(row)
        if len(selected) == remaining:
            break
    folder = audit.OUT / "plank_auto_screen_items"
    folder.mkdir(exist_ok=True)
    audit.save_json(folder / "candidate_manifest.json", selected)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(extract.MODEL),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    results = previous_results
    extract.check_resources()
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as detector:
        for row in selected:
            extract.check_resources()
            capture = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
            )
            sequences = []
            try:
                count = int(capture.get(7))
                fps = capture.get(5)
                if count < 3 or fps <= 0:
                    raise ValueError("Invalid screening clip")
                indices = np.rint(
                    (count - 1) * np.array([0.2, 0.5, 0.8])
                ).astype(int)
                for index in indices:
                    capture.set(1, int(index))
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError("Screening decode failed")
                    height, width = frame.shape[:2]
                    scale = min(1, 640 / max(height, width))
                    resized = cv2.resize(
                        frame, (round(width * scale), round(height * scale))
                    )
                    result = detector.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(
                                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            ),
                        )
                    )
                    points = np.full((33, 2), np.nan)
                    if result.pose_landmarks:
                        for j, point in enumerate(result.pose_landmarks[0]):
                            if (
                                point.visibility >= 0.5
                                and point.presence >= 0.5
                                and 0 <= point.x <= 1
                                and 0 <= point.y <= 1
                            ):
                                points[j] = point.x * width, point.y * height
                    sequences.append(points)
            finally:
                capture.release()
            seq = np.asarray(sequences)
            sides = []
            for shoulder, hip, knee, ankle, toe in (
                (11, 23, 25, 27, 31),
                (12, 24, 26, 28, 32),
            ):
                joints = [shoulder, hip, knee, ankle, toe]
                observed = np.isfinite(seq[:, joints]).all(2).all(1)
                torso = np.linalg.norm(seq[:, shoulder] - seq[:, hip], axis=1)
                span = np.divide(
                    np.abs(seq[:, shoulder, 0] - seq[:, ankle, 0]),
                    torso,
                    out=np.full(3, np.nan),
                    where=torso > 10,
                )
                valid = span[observed & np.isfinite(span)]
                sides.append(
                    dict(
                        full_chain_frames=int(observed.sum()),
                        horizontal_span=(
                            float(np.median(valid)) if valid.size else None
                        ),
                    )
                )
            item_coverage = {}
            for item, joints in [
                ("trunk", [11, 23, 27]),
                ("elbow", [11, 13, 23]),
                ("support", [23, 25, 27]),
            ]:
                item_coverage[item] = max(
                    float(
                        np.isfinite(seq[:, np.asarray(joints) + offset])
                        .all((1, 2))
                        .mean()
                    )
                    for offset in [0, 1]
                )
            plausible = max(item_coverage.values()) >= 2 / 3
            old_full_chain_candidate = any(
                s["full_chain_frames"] >= 2
                and s["horizontal_span"] is not None
                and s["horizontal_span"] >= 1.5
                for s in sides
            )
            results.append(
                dict(
                    id=row["id"],
                    person=row["person"],
                    video_path=row["path"],
                    video_sha256=audit.digest(pathlib.Path(row["path"])),
                    split="train",
                    fps=fps,
                    sample_frames=indices.tolist(),
                    side_geometry_proxies=sides,
                    item_coverage=item_coverage,
                    old_full_chain_candidate=old_full_chain_candidate,
                    candidate_for_direct_review=plausible,
                    direct_visual_reviewed=False,
                    score=None,
                    status="automated_camera_screening_only_not_a_label",
                )
            )
            np.savez_compressed(
                folder / (row["id"] + ".npz"),
                image_xy=seq,
                frame_numbers=indices,
            )
            audit.save_json(folder / "results.json", results)
    audit.save_json(
        folder / "summary.json",
        dict(
            automatically_screened=len(results),
            direct_review_candidates=sum(
                r["candidate_for_direct_review"] for r in results
            ),
            direct_visual_reviews=0,
            score_labels_created=0,
            model_sha256=audit.digest(extract.MODEL),
            labels_by_provider_used_for_scores=False,
        ),
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--include-known-people", action="store_true")
    args = parser.parse_args()
    print(
        collections.Counter(
            r["candidate_for_direct_review"]
            for r in screen(args.limit, args.include_known_people)
        )
    )
