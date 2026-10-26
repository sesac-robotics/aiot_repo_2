"""보유 YTW에서 부족한 팔 펴기 관측 후보만 자동 선별해 직접 검토를 준비한다."""

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


def _angle(a, b, c):
    u, v = a - b, c - b
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    return (
        float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))
        if denominator > 1
        else None
    )


def run() -> dict:
    """미라벨 개발 YTW 150개에서 Y 도달 후보를 선별하고 상위 12개를 준비한다.

    Returns:
        dict: 자동 선별 수·팔 펴기 구도 후보·검토판 수. 각도 단위는 도다.
        제공자 문구는 후보 선택에만 사용하며 정상 등급·점수로 변환하지 않는다.

    Raises:
        OSError: 보유 원본·모델·검토판 접근 실패.
        ValueError: 원본·장면 디코딩 오류.
        RuntimeError: 온도·메모리 보호 조건 위반.
    """
    cv2.setNumThreads(1)
    known = {
        "_".join(r["id"].split("_")[:2])
        for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    candidates = [
        r
        for r in audit.read_json(
            audit.BASE / "output/overnight-20261020/score_candidates.json"
        )
        if r["exercise"] == "ytw"
        and r["id"].startswith("qevd_")
        and r["split"] == "train"
        and r["id"] not in known
    ]
    candidates.sort(
        key=lambda r: (
            not any(
                "torso parallel to the ground, thumbs pointed up" in s
                for s in r["source_labels"]
            ),
            hashlib.sha256(
                ("existing-y-extension" + r["id"]).encode()
            ).hexdigest(),
        )
    )
    selected = []
    seen = set()
    for row in candidates:
        if row["person"] in seen:
            continue
        selected.append(row)
        seen.add(row["person"])
        if len(selected) >= 150:
            break
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    output = ROOT / "ytw_extension_screen.json"
    results = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as detector:
        for row in selected:
            check_resources()
            cap = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            count = int(cap.get(7))
            fps = cap.get(5)
            views = []
            for fraction in [0.18, 0.68]:
                index = round((count - 1) * fraction)
                cap.set(1, index)
                ok, frame = cap.read()
                if not ok or fps <= 0:
                    cap.release()
                    raise ValueError("Y screen frame unavailable")
                height, width = frame.shape[:2]
                scale = min(1, 640 / max(height, width))
                small = cv2.resize(
                    frame, (round(width * scale), round(height * scale))
                )
                result = detector.detect(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
                    )
                )
                xy = np.full((33, 2), np.nan)
                if result.pose_landmarks:
                    for i, point in enumerate(result.pose_landmarks[0]):
                        if (
                            (point.visibility or 0) >= 0.5
                            and (point.presence or 0) >= 0.5
                            and 0.02 < point.x < 0.98
                            and 0.02 < point.y < 0.98
                        ):
                            xy[i] = [point.x * width, point.y * height]
                elbows = []
                elevations = []
                if np.isfinite(xy[[11, 12, 13, 14, 15, 16, 23, 24]]).all():
                    for shoulder, elbow, wrist, hip in [
                        [11, 13, 15, 23],
                        [12, 14, 16, 24],
                    ]:
                        elbows.append(
                            _angle(xy[shoulder], xy[elbow], xy[wrist])
                        )
                        elevations.append(
                            _angle(xy[hip], xy[shoulder], xy[wrist])
                        )
                if elbows and None not in elbows + elevations:
                    views.append(
                        dict(
                            frame=index,
                            min_elbow_deg=min(elbows),
                            min_elevation_deg=min(elevations),
                        )
                    )
            cap.release()
            best = (
                max(
                    views,
                    key=lambda x: min(x["min_elbow_deg"], 150)
                    + min(x["min_elevation_deg"], 130),
                )
                if views
                else None
            )
            results.append(
                dict(
                    row,
                    automatic_geometric_candidate_only=True,
                    best=best,
                    extension_candidate=best is not None
                    and best["min_elbow_deg"] >= 150
                    and best["min_elevation_deg"] >= 110,
                    direct_reviewed=False,
                    score_label_created=False,
                )
            )
            audit.save_json(
                output,
                dict(
                    records=results,
                    completed=False,
                    model_sha256=audit.digest(MODEL),
                ),
            )
    ranked = sorted(
        results,
        key=lambda x: (
            x["extension_candidate"],
            x["best"]["min_elevation_deg"] if x["best"] else 0,
            x["best"]["min_elbow_deg"] if x["best"] else 0,
        ),
        reverse=True,
    )
    prepared = []
    existing_path = ROOT / "seventy_batch_preparation.json"
    previous = audit.read_json(existing_path)["records"]
    ids = {x["id"] for x in previous}
    for row in ranked[:12]:
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + "_extension_candidate.jpg"),
            0,
            None,
            24,
        )
        record = dict(row, preparation=receipt)
        prepared.append(record)
        if row["id"] not in ids:
            previous.append(record)
            ids.add(row["id"])
    audit.save_json(
        existing_path,
        dict(records=previous, direct_review_performed_by_script=False),
    )
    report = dict(
        records=results,
        completed=True,
        prepared=prepared,
        extension_candidates=sum(r["extension_candidate"] for r in results),
        model_sha256=audit.digest(MODEL),
        training_labels_created=0,
    )
    audit.save_json(output, report)
    return dict(
        screened=len(results),
        extension_candidates=report["extension_candidates"],
        prepared=len(prepared),
    )


if __name__ == "__main__":
    print(json.dumps(run()))
