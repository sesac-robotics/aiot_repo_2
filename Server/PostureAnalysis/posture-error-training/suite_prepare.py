"""기존 제공자 Train/Test 구분을 유지하며 선택 영상의 관절을 추출합니다. 프로세스별 MediaPipe 어댑터를 재사용하고 추적 상태는 영상 사이에서 초기화합니다.

Full selected-source extraction for four exercises. Resume per-video caches."""

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from suite_config import BASE, CONFIG_VERSION, SUITE, TASKS


class _WorkerResources:
    """프로세스 풀의 한 워커가 소유하는 추출 어댑터입니다. 다른 프로세스와 공유하지 않습니다."""

    def __init__(self):
        self.pose = None

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None


_worker_resources = (
    _WorkerResources()
)  # Process-local bootstrap, not domain state.


def init_worker():
    from multiprocessing.util import Finalize

    import cv2
    import mediapipe as mp

    # multiprocessing workers use their own finalizer path, including fork exits.
    Finalize(None, _worker_resources.close, exitpriority=10)
    cv2.setNumThreads(1)
    _worker_resources.pose = mp.solutions.pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract(task):
    import cv2
    import numpy as np
    from suite_config import SAMPLE_FPS
    from suite_features import prepare_landmarks

    exercise, vid, path, cache = task
    cap = None
    try:
        if _worker_resources.pose is None:
            raise RuntimeError("추출 워커가 초기화되지 않았습니다.")
        _worker_resources.pose.reset()
        cap = cv2.VideoCapture(
            path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Invalid video FPS")
        step = max(1, round(fps / SAMPLE_FPS))
        points = []
        times = []
        aspect = 1.0
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
                r = _worker_resources.pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                points.append(
                    np.array(
                        [
                            [l.x, l.y, l.z, l.visibility]
                            for l in r.pose_landmarks.landmark
                        ],
                        dtype=np.float32,
                    )
                    if r.pose_landmarks
                    else np.zeros((33, 4), dtype=np.float32)
                )
                times.append(i / fps)
            i += 1
        if not points:
            return vid, dict(status="failed", error="No decoded frames")
        raw = np.array(points)
        x, ratio = prepare_landmarks(raw, aspect, TASKS[exercise]["required"])
        if x is None:
            return vid, dict(status="excluded_pose_quality", valid_ratio=ratio)
        temporary = cache + "." + str(os.getpid()) + ".tmp.npz"
        np.savez_compressed(
            temporary,
            X=x,
            landmarks=raw,
            times=times,
            aspect=aspect,
            config_version=CONFIG_VERSION,
        )
        os.replace(temporary, cache)
        return vid, dict(
            status="ok", valid_ratio=ratio, sampled_frames=len(raw)
        )
    except Exception as e:
        return vid, dict(status="failed", error=str(e))
    finally:
        if cap is not None:
            cap.release()


def collect():
    allrows = []
    counts = {}
    mapping = json.loads(
        (
            BASE.parent
            / "qevd-selected/derived/manifests/feedback_label_mapping.json"
        ).read_text()
    )
    feedback = {r["label"]: r["feedback_en"] for r in mapping}
    for exercise, config in TASKS.items():
        root = BASE.parent / (
            "shoulder-qevd"
            if config["dataset"] == "shoulder"
            else "qevd-selected"
        )
        with (root / "manifests/short_clips.csv").open() as f:
            manifest = {r["video_id"]: r for r in csv.DictReader(f)}
        if config["dataset"] == "shoulder":
            feedback.update(
                {
                    r["label"]: r["feedback_en"]
                    for r in json.loads(
                        (root / "manifests/label_definitions.json").read_text()
                    )
                }
            )
        selected = []
        excluded = 0
        for row in json.loads(
            (root / "raw/coach/fine_grained_labels.json").read_text()
        ):
            first = row["labels"][0].split(" - ")[0]
            belongs = (
                first.startswith("standing ")
                if exercise == "ytw"
                else first
                == {
                    "shoulder_press": "shoulder press",
                    "elbow_plank": "elbow plank",
                }.get(exercise, exercise)
            )
            if not belongs:
                continue
            labels = {l.split(" - ", 1)[1] for l in row["labels"]}
            if labels & set(config["excluded"]):
                excluded += 1
                continue
            # Combined source plank labels explicitly entail each named component.
            if exercise == "elbow_plank":
                for l in list(labels):
                    if l.startswith("arms facing inward and"):
                        labels.add("arms facing inward")
                    if "legs and hips on the floor" in l:
                        labels.add("legs and hips on the floor")
            vid = Path(row["video_path"]).stem
            meta = manifest[vid]
            target = [int(t in labels) for t in config["targets"]]
            selected.append(
                dict(
                    video_id=vid,
                    exercise=exercise,
                    dataset=config["dataset"],
                    video_path=str((root / meta["video_path"]).resolve()),
                    source_split=row["split"],
                    worker_id=int(meta["worker_id"]),
                    labels=row["labels"],
                    target=target,
                )
            )
        directory = SUITE / exercise
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "selected.json").write_text(json.dumps(selected, indent=2))
        counts[exercise] = dict(
            selected=len(selected),
            excluded_source_labels=excluded,
            source_train=sum(r["source_split"] == "train" for r in selected),
            source_test=sum(r["source_split"] == "test" for r in selected),
            target_positives=dict(
                zip(
                    config["targets"],
                    [
                        sum(r["target"][j] for r in selected)
                        for j in range(len(config["targets"]))
                    ],
                )
            ),
        )
        allrows += selected
    (SUITE / "selection_report.json").write_text(json.dumps(counts, indent=2))
    (SUITE / "source_feedback.json").write_text(json.dumps(feedback, indent=2))
    return allrows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--only", choices=list(TASKS))
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-excluded", action="store_true")
    a = p.parse_args()
    SUITE.mkdir(exist_ok=True)
    (SUITE / "cache").mkdir(exist_ok=True)
    rows = collect()
    if a.only:
        rows = [r for r in rows if r["exercise"] == a.only]
    if a.limit:
        rows = rows[: a.limit]
    quality_path = SUITE / "pose_quality.json"
    quality = (
        json.loads(quality_path.read_text()) if quality_path.exists() else {}
    )
    tasks = []
    for r in rows:
        cache = SUITE / "cache" / f'{r["video_id"]}.npz'
        if cache.exists():
            import numpy as np

            try:
                with np.load(cache) as saved:
                    assert str(saved["config_version"]) == CONFIG_VERSION
                    assert (
                        saved["X"].shape == (32, 109)
                        and np.isfinite(saved["X"]).all()
                    )
                continue
            except (OSError, ValueError, EOFError, AssertionError, KeyError):
                archive = SUITE / "invalid_cache"
                archive.mkdir(exist_ok=True)
                cache.rename(archive / cache.name)
        if (
            not a.retry_excluded
            and quality.get(r["video_id"], {}).get("status")
            == "excluded_pose_quality"
        ):
            continue
        tasks.append(
            (r["exercise"], r["video_id"], r["video_path"], str(cache))
        )
    tasks.sort(
        key=lambda t: {
            "shoulder_press": 0,
            "ytw": 1,
            "elbow_plank": 2,
            "pushups": 3,
        }[t[0]]
    )
    print(
        f"Pending {len(tasks)} of {len(rows)} selected videos; workers={a.workers}",
        flush=True,
    )
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=a.workers, initializer=init_worker
    ) as pool:
        jobs = [pool.submit(extract, t) for t in tasks]
        for n, future in enumerate(as_completed(jobs), 1):
            vid, result = future.result()
            quality[vid] = result
            if n % 50 == 0 or n == len(jobs):
                quality_path.write_text(json.dumps(quality, indent=2))
                print(
                    f'Extracted {n}/{len(tasks)} elapsed={time.time()-started:.0f}s statuses={dict(Counter(v["status"] for v in quality.values()))}',
                    flush=True,
                )
    quality_path.write_text(json.dumps(quality, indent=2))
    import numpy as np

    for exercise in TASKS:
        selected = [
            r
            for r in rows
            if r["exercise"] == exercise
            and (SUITE / "cache" / f'{r["video_id"]}.npz').exists()
        ]
        if not selected:
            continue
        xs = []
        for r in selected:
            with np.load(SUITE / "cache" / f'{r["video_id"]}.npz') as cache:
                assert str(cache["config_version"]) == CONFIG_VERSION
                xs.append(cache["X"])
        np.savez_compressed(
            SUITE / exercise / "data.npz",
            X=np.stack(xs),
            y=np.array([r["target"] for r in selected], dtype=np.float32),
            groups=np.array([r["worker_id"] for r in selected]),
            ids=np.array([r["video_id"] for r in selected]),
            source_split=np.array([r["source_split"] for r in selected]),
        )
        print(f"{exercise}: saved {len(selected)} videos", flush=True)


if __name__ == "__main__":
    main()
