"""보유 플랭크의 미선별 원본을 단일 Heavy 장면으로 빠르게 구도 선별한다.

600개 상한은 후보 선별 수이며 학습·직접 검토 수가 아니다. 동일 Heavy
IMAGE 추출기를 순차 재사용하고 점수·제공자 상태를 입력하지 않는다.
"""

import hashlib
import json
import pathlib

import cv2
import mediapipe as mp
import numpy as np

import audit
import external_review
from extract import MODEL
from recovery_prepare import ROOT
from resources import check_resources


def run() -> dict:
    """기존 280개와 다른 개발 후보에서 측면·사선 검토 우선순위를 만든다.

    Returns:
        dict: 자동 단일 장면 선별 수·넓게 잡은 구도 후보·검토판 수.
        점수 라벨·실제 접점·전 구간 관측 또는 학습 가능 판정이 아니다.

    Raises:
        OSError: 보유 원본·모델·결과 접근 실패.
        ValueError: 영상·장면 디코딩·모델 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cv2.setNumThreads(1)
    known = set()
    for name in ["plank_screen.json", "plank_screen_expanded_200.json"]:
        known.update(r["id"] for r in audit.read_json(ROOT / name)["records"])
    pool = [
        r
        for r in audit.read_json(
            audit.BASE / "output/overnight-20261020/score_candidates.json"
        )
        if r["exercise"] == "plank"
        and r["id"].startswith("qevd_")
        and r["split"] == "train"
        and r["id"] not in known
    ]
    pool.sort(
        key=lambda r: hashlib.sha256(
            ("single-heavy-local-plank" + r["id"]).encode()
        ).hexdigest()
    )
    queue = []
    groups = set()
    for row in pool:
        if row["person"] not in groups:
            queue.append(row)
            groups.add(row["person"])
            if len(queue) == 600:
                break
    selected_ids = {r["id"] for r in queue}
    for row in pool:
        if len(queue) >= 600:
            break
        if row["id"] not in selected_ids:
            queue.append(row)
            selected_ids.add(row["id"])
    target = ROOT / "plank_single_screen_600.json"
    model_sha = audit.digest(MODEL)
    existing = audit.read_json(target) if target.exists() else dict(records=[])
    if existing.get("model_sha256", model_sha) != model_sha:
        raise ValueError("Screening model changed")
    ready = {r["id"]: r for r in existing["records"]}
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as detector:
        for row in queue:
            check_resources()
            if row["id"] in ready:
                continue
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            count = int(cap.get(7))
            fps = cap.get(5)
            index = max(0, count // 2)
            cap.set(1, index)
            ok, frame = cap.read()
            cap.release()
            if not ok or fps <= 0:
                raise ValueError("Screen frame unavailable")
            height, width = frame.shape[:2]
            scale = min(1, 640 / max(width, height))
            small = cv2.resize(
                frame, (round(width * scale), round(height * scale))
            )
            result = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                )
            )
            points = np.full((33, 2), np.nan)
            if result.pose_landmarks:
                for i, point in enumerate(result.pose_landmarks[0]):
                    if point.visibility >= 0.5 and point.presence >= 0.5:
                        points[i] = [point.x * width, point.y * height]
            choices = []
            for joints in [[11, 23, 25, 27], [12, 24, 26, 28]]:
                observed = bool(np.isfinite(points[joints]).all())
                torso = (
                    float(np.linalg.norm(points[joints[0]] - points[joints[1]]))
                    if observed
                    else 0
                )
                extent = (
                    float(
                        abs(points[joints[0], 0] - points[joints[-1], 0])
                        / max(1, torso)
                    )
                    if observed
                    else 0
                )
                choices.append((observed, extent))
            observed, extent = max(choices, key=lambda x: (x[0], x[1]))
            ready[row["id"]] = dict(
                row,
                frame=index,
                fps=fps,
                source_sha256=audit.digest(pathlib.Path(row["path"])),
                full_side_chain_observed=observed,
                horizontal_extent_torso_ratio=extent,
                broad_oblique_or_side_candidate=observed and extent >= 1.4,
                automatically_screened=True,
                directly_reviewed=False,
            )
            audit.save_json(
                target,
                dict(
                    records=list(ready.values()),
                    model_sha256=model_sha,
                    completed=False,
                    mode="single_IMAGE_frame_screen_only",
                    training_labels_created=0,
                ),
            )
    ranked = sorted(
        ready.values(),
        key=lambda r: (
            r["broad_oblique_or_side_candidate"],
            r["full_side_chain_observed"],
            r["horizontal_extent_torso_ratio"],
        ),
        reverse=True,
    )
    prepared = []
    for row in ranked[:12]:
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + "_single_screen_full.jpg"),
            0,
            None,
            24,
        )
        prepared.append(dict(row, preparation=receipt))
    report = dict(
        records=list(ready.values()),
        model_sha256=model_sha,
        completed=True,
        mode="single_IMAGE_frame_screen_only",
        prepared=prepared,
        training_labels_created=0,
        direct_visual_reviews=0,
        side_candidates=sum(
            r["broad_oblique_or_side_candidate"] for r in ready.values()
        ),
    )
    audit.save_json(target, report)
    return dict(
        screened=len(ready),
        broad_candidates=report["side_candidates"],
        prepared=len(prepared),
    )


if __name__ == "__main__":
    print(json.dumps(run()))
