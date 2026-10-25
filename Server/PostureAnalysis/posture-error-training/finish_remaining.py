"""기존 중간 학습 산출물에서 남은 보고 단계 실행을 조정하는 연구 진입점입니다. 실행 시 산출물을 변경할 수 있으므로 보존본에 자동 실행하지 않습니다.

Finish each exercise as soon as its full extraction completes (internal runner).
"""

import gc
import hashlib
import json
import subprocess
import sys
import time

from suite_config import BASE, SUITE, TASKS


def main():
    done = set()
    last_progress = 0
    while len(done) < len(TASKS):
        try:
            quality = json.loads((SUITE / "pose_quality.json").read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(5)
            continue
        progress = {}
        for exercise in ["shoulder_press", "ytw", "elbow_plank", "pushups"]:
            rows = json.loads((SUITE / exercise / "selected.json").read_text())
            complete = sum(
                (SUITE / "cache" / f'{r["video_id"]}.npz').exists()
                or quality.get(r["video_id"], {}).get("status")
                in ["excluded_pose_quality", "failed"]
                for r in rows
            )
            progress[exercise] = f"{complete}/{len(rows)}"
            if exercise in done or complete != len(rows):
                continue
            # Last queue group is exported by the running extractor; wait until its ZIP is complete.
            if exercise == "pushups":
                import numpy as np

                try:
                    with np.load(SUITE / exercise / "data.npz") as ready:
                        assert (
                            len(ready["ids"]) > 0
                            and np.isfinite(ready["X"]).all()
                        )
                except (OSError, ValueError, EOFError, AssertionError):
                    continue
            from suite_finalize import finalize

            out = SUITE / exercise / "outputs/metrics.json"
            data = SUITE / exercise / "data.npz"
            up_to_date = (
                out.exists()
                and data.exists()
                and json.loads(out.read_text())["dataset_sha256"]
                == hashlib.sha256(data.read_bytes()).hexdigest()
            )
            if not up_to_date:
                finalize(exercise)
                import torch

                torch.set_num_threads(4)
                from suite_train import fit

                fit(exercise, 80)
            done.add(exercise)
            print("Completed models: " + str(sorted(done)), flush=True)
            gc.collect()
        if time.time() - last_progress > 30:
            print(
                "Full-source extraction readiness: " + str(progress), flush=True
            )
            last_progress = time.time()
        if len(done) < len(TASKS):
            time.sleep(10)
    from suite_data_audit import main as audit
    from suite_report import main as report

    audit()
    report()
    subprocess.run([sys.executable, str(BASE / "test_suite.py")], check=True)
    print("ALL FOUR EXERCISES TRAINED AND CHECKED", flush=True)


if __name__ == "__main__":
    main()
