"""카메라 없이 실제 교정·점수 프로세스의 입력·초기화·종료를 확인합니다."""

import argparse
import json
import threading
import time

import numpy as np

from .coordinator import EvaluationWorker


def main():
    """합성 빈 프레임은 정확도 평가가 아닌 프로세스 연결 검사에만 사용합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-python", required=True)
    parser.add_argument("--score-python", required=True)
    args = parser.parse_args()
    results = {}
    condition = threading.Condition()

    def receive(kind, result):
        with condition:
            results[kind] = result
            condition.notify_all()

    workers = [
        EvaluationWorker(kind, "squats", runtime, receive)
        for kind, runtime in [
            ("score", args.score_python),
            ("correction", args.correction_python),
        ]
    ]
    try:
        for epoch in (1, 2):
            with condition:
                results.clear()
            stamp = time.monotonic()
            for worker in workers:
                worker.submit(
                    np.zeros((480, 640, 3), np.uint8),
                    {
                        "generation": 1,
                        "epoch": epoch,
                        "frame_id": epoch,
                        "timestamp": stamp,
                    },
                )
            with condition:
                assert condition.wait_for(
                    lambda: len(results) == 2
                    and all(r.get("epoch") == epoch for r in results.values()),
                    timeout=30,
                ), results
                assert results["score"]["status"] == "preparing", results
                assert results["score"]["movement"]["count"] == 0
                assert results["score"]["repetitions"] == []
                assert (
                    results["correction"]["status"] == "pose_not_visible"
                ), results
                assert results["correction"]["result"] is None
                assert results["correction"]["window_seconds"] == 0
        for worker in workers:
            worker.close()
        with condition:
            assert all(
                r.get("final") and r["status"] == "stopped"
                for r in results.values()
            ), results
        print(
            json.dumps(
                {
                    "both_real_workers": "passed",
                    "source_epoch_reset": "passed",
                    "empty_frame_no_fabricated_rep_or_correction": "passed",
                    "graceful_shutdown": "passed",
                    "camera_opened": False,
                },
                ensure_ascii=False,
            )
        )
    finally:
        for worker in workers:
            if worker.process.poll() is None:
                worker.close()


if __name__ == "__main__":
    main()
