"""기존 스쿼트 원본 라벨과 분할을 보존하며 관절을 추출하던 연구 도구입니다. 이번 통합에서는 실행·재학습하지 않으며 자원 정리와 서식만 검토합니다.

Extract full squat subset at 6 fps. Resumable per-video cache; source test split kept.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / "qevd-selected"


def extract(task):
    import cv2
    import mediapipe as mp
    import numpy as np
    from features import SAMPLE_FPS, features, resample

    vid, path, cache = task
    cv2.setNumThreads(1)
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or total < 2:
            raise ValueError("invalid video metadata")
        step = max(1, round(fps / SAMPLE_FPS))
        points = []
        times = []
        aspect = 1.0
        # Full bundled model avoids network downloads. Reset tracking per video.
        with mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            i = 0
            while cap.grab():
                if i % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    h, w = frame.shape[:2]
                    aspect = w / h
                    if max(h, w) > 480:
                        frame = cv2.resize(
                            frame,
                            (
                                round(w * 480 / max(h, w)),
                                round(h * 480 / max(h, w)),
                            ),
                        )
                    result = pose.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    p = np.full((33, 4), np.nan, dtype=np.float32)
                    if result.pose_landmarks:
                        p = np.array(
                            [
                                [l.x, l.y, l.z, l.visibility]
                                for l in result.pose_landmarks.landmark
                            ],
                            dtype=np.float32,
                        )
                        if np.min(p[[11, 12, 23, 24, 25, 26, 27, 28], 3]) < 0.3:
                            p[:] = np.nan
                    points.append(p)
                    times.append(i / fps)
                i += 1
        cap.release()
        raw = np.asarray(points)
        valid = np.isfinite(raw).all(axis=(1, 2))
        if len(raw) < 8 or valid.mean() < 0.6 or valid.sum() < 8:
            return vid, dict(
                status="excluded_pose_quality",
                valid_ratio=float(valid.mean()) if len(raw) else 0,
            )
        flat = raw.reshape(len(raw), -1)
        for j in range(flat.shape[1]):
            flat[:, j] = np.interp(
                np.arange(len(flat)), np.flatnonzero(valid), flat[valid, j]
            )
        f = features(raw, aspect)
        np.savez_compressed(
            cache,
            X=resample(f),
            landmarks=raw,
            times=times,
            aspect=aspect,
            valid=valid,
        )
        return vid, dict(
            status="ok",
            valid_ratio=float(valid.mean()),
            sampled_frames=len(raw),
        )
    except Exception as e:
        return vid, dict(status="failed", error=str(e))
    finally:
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    (BASE / "cache").mkdir(exist_ok=True)
    (BASE / "data").mkdir(exist_ok=True)
    with (ROOT / "manifests/short_clips.csv").open() as f:
        manifest = {r["video_id"]: r for r in csv.DictReader(f)}
    rows = []
    excluded = []
    for r in json.loads(
        (ROOT / "raw/coach/fine_grained_labels.json").read_text()
    ):
        if not all(l.startswith("squats - ") for l in r["labels"]):
            continue
        labels = {l.split(" - ")[1] for l in r["labels"]}
        targets = labels & {"shallow", "back not straight"}
        if labels & {
            "not visible",
            "insufficient",
            "knees over toes",
            "plie",
        } or not (
            targets
            or (
                "no obvious issue" in labels
                and labels
                <= {"no obvious issue", "90 degrees", "shoulder-width", "hold"}
            )
        ):
            excluded.append(r)
            continue
        vid = Path(r["video_path"]).stem
        m = manifest[vid]
        rows.append(
            dict(
                video_id=vid,
                video_path=str(ROOT / m["video_path"]),
                worker_id=int(m["worker_id"]),
                source_split=r["split"],
                labels=r["labels"],
                target=[
                    int(t in labels) for t in ("shallow", "back not straight")
                ],
            )
        )
    (BASE / "data/selected.json").write_text(json.dumps(rows, indent=2))
    (BASE / "data/excluded_source_labels.json").write_text(
        json.dumps(excluded, indent=2)
    )
    quality = {}
    pending = []
    for r in rows:
        cache = BASE / "cache" / f'{r["video_id"]}.npz'
        if cache.exists():
            quality[r["video_id"]] = dict(status="ok_cached")
        else:
            pending.append((r["video_id"], r["video_path"], str(cache)))
    print(
        f"Selected {len(rows)}, source exclusions {len(excluded)}, cache {len(quality)}, pending {len(pending)}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, t) for t in pending]
        for n, future in enumerate(as_completed(futures), 1):
            vid, result = future.result()
            quality[vid] = result
            if n % 25 == 0 or n == len(futures):
                print(
                    f'Extracted {n}/{len(futures)}; ok={sum(v["status"].startswith("ok") for v in quality.values())}',
                    flush=True,
                )
                (BASE / "data/pose_quality.json").write_text(
                    json.dumps(quality, indent=2)
                )
    (BASE / "data/pose_quality.json").write_text(json.dumps(quality, indent=2))
    import numpy as np

    usable = [
        r for r in rows if (BASE / "cache" / f'{r["video_id"]}.npz').exists()
    ]
    np.savez_compressed(
        BASE / "data/squat.npz",
        X=np.stack(
            [
                np.load(BASE / "cache" / f'{r["video_id"]}.npz')["X"]
                for r in usable
            ]
        ),
        y=np.array([r["target"] for r in usable], dtype=np.float32),
        groups=np.array([r["worker_id"] for r in usable]),
        source_split=np.array([r["source_split"] for r in usable]),
        ids=np.array([r["video_id"] for r in usable]),
    )
    print(f"Pose dataset saved: {len(usable)} videos", flush=True)


if __name__ == "__main__":
    main()
