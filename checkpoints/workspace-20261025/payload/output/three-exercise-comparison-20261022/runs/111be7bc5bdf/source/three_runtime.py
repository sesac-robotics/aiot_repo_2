"""후보 모델의 새 Heavy 추출 포함 처리 시간과 반복 실행 차이를 측정한다.

세 운동의 개발 예제 하나씩을 두 번 새로 추출한다. 같은 프레임 입력의
결정성 검사와 달리 관절 추출 재실행까지 포함하며 독립 시험은 아니다.
"""

import pathlib
import time

import cv2
import joblib
import numpy as np

import audit
import extract
import three_items
import three_input
import three_pipeline
import three_protocol
import three_training
from resources import check_resources


def run() -> dict:
    """운동별 개발 입력의 두 번 새 추출·채점 시간을 저장한다.

    Returns:
        dict: 보호 설정하의 벽시계 초·영상 초·비율과 점수 재실행 차이.
        초기화·두 시점 디코딩·Heavy·특징·양쪽 점수 계산을 포함한다.
        모델 로딩·전체 데이터 감사 시간은 별도로 포함하지 않는다.

    Raises:
        OSError: 모델·원본·결과 읽기 쓰기 실패.
        ValueError: 평가 가능 개발 예제 없음 또는 입력 형상 오류.
        RuntimeError: 전원·온도·메모리 보호 위반.
    """
    root = three_protocol.OUT
    report = audit.read_json(root / "development_results.json")
    inputs = three_pipeline.load()
    output = dict(
        records=[],
        independent_test=False,
        repeats=2,
        model_version=report["label_version"],
        cpu_quota_percent=50,
        memory_max_gib=3,
        sample_hz=10,
        max_edge_pixels=640,
        includes="decoder + Heavy GPU initialization/extraction + features + AI and rule scoring",
        excludes="model deserialization and whole-dataset audit",
    )
    for result in report["exercises"]:
        exercise = result["exercise"]
        candidates = (
            result["validation_predictions"]
            or result["training_person_out_predictions"]
        )
        chosen = next(
            (
                p
                for p in candidates
                if p["ai_score"] is not None and p["rule_score"] is not None
            ),
            None,
        )
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None:
            output["records"].append(
                dict(
                    exercise=exercise, hold="no_assessable_development_example"
                )
            )
            continue
        row = next(r for r in inputs if r["id"] == chosen["id"])
        model = joblib.load(result["model_path"])["model"]
        repetitions = []
        for repeat in range(2):
            check_resources()
            destination = (
                root / "runtime" / report["label_version"] / str(repeat)
            )
            target = destination / row["id"]
            if target.exists():
                raise ValueError(
                    "Fresh extraction destination already exists; use a new run version"
                )
            manifest = dict(row["manifest"])
            manifest["required_views"] = (
                ["side", "synchronized_front"]
                if exercise == "squats"
                else ["side"] if exercise == "pushups" else ["front"]
            )
            start = time.perf_counter()
            views = [
                (
                    "primary" if exercise == "shoulder" else "side",
                    manifest["video_path"],
                )
            ]
            if exercise == "squats":
                views.append(("front", manifest["front_path"]))
            sequences, receipts = {}, []
            for view, path in views:
                receipt = extract.extract_view(
                    manifest, view, path, destination=destination
                )
                if receipt.get("reused"):
                    raise ValueError(
                        "Runtime benchmark must not reuse Heavy caches"
                    )
                receipts.append(receipt)
                with np.load(destination / row["id"] / (view + ".npz")) as data:
                    seq = data["image_xy"].copy()
                capture = cv2.VideoCapture(
                    path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
                )
                try:
                    seq = three_items.mask_image_border(
                        seq, capture.get(3), capture.get(4)
                    )
                finally:
                    capture.release()
                sequences[view] = three_input.apply_framing(seq, manifest, view)
            primary = sequences["primary" if exercise == "shoulder" else "side"]
            notes = three_items.describe(
                exercise, primary, sequences.get("front")
            )
            notes = three_items.apply_view_policy(
                exercise, notes, manifest.get("required_input_views", [])
            )
            notes = three_input.apply_observation(
                notes, manifest, "primary" if exercise == "shoulder" else "side"
            )
            prediction = three_training.predict(model, dict(row, notes=notes))
            seconds = time.perf_counter() - start
            duration = (
                manifest["segment"]["end_frame"]
                - manifest["segment"]["start_frame"]
            ) / manifest["segment"]["fps"]
            repetitions.append(
                dict(
                    repeat=repeat,
                    seconds=seconds,
                    input_seconds=duration,
                    wall_time_per_input_second=seconds / duration,
                    **prediction,
                    receipts=receipts
                )
            )
        differences = {
            key: (
                abs(repetitions[0][key] - repetitions[1][key])
                if all(r[key] is not None for r in repetitions)
                else None
            )
            for key in ["ai_score", "rule_score"]
        }
        output["records"].append(
            dict(
                id=row["id"],
                exercise=exercise,
                source_split=row["split"],
                repeats=repetitions,
                fresh_extraction_score_delta_points=differences,
            )
        )
        audit.save_json(root / "runtime_results.json", output)
    audit.save_json(root / "runtime_results.json", output)
    return output


if __name__ == "__main__":
    print(len(run()["records"]))
