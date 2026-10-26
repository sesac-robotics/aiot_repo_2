"""새 직접 검토 반복의 관절 표시와 두 후보 점수를 학습 예제로 시각화한다."""

import json
import pathlib
import statistics
import time

import cv2
import joblib
import numpy as np

import audit
import render
import three_training
from recovery_features import load
from recovery_prepare import ROOT
from recovery_train import head_key, predict


def profile(models: dict, row: dict) -> dict:
    """동일한 관측 항목의 AI 회귀·규칙 계산 시간과 재계산 일치를 측정한다.

    Args:
        models: 해당 운동의 저장된 항목별 후보 모델.
        row: 이미 추출된 기하 특징과 관측 정책. 정답 등급은 사용하지 않는다.

    Returns:
        dict: 공통 평가 항목·3회 중앙값 밀리초·같은 입력 재계산 일치.
        공통 관절 추출·특징 계산·영상 읽기 시간은 제외한 점수 계산 시간이다.

    Raises:
        ValueError: 측정한 출력이 정식 예측 결과와 다를 경우.
    """
    expected = predict(models, row)
    indices = [
        i
        for i, (a, b) in enumerate(
            zip(expected["ai_levels"], expected["rule_levels"])
        )
        if a is not None and b is not None
    ]
    ai_times = []
    rule_times = []
    for _ in range(3):
        start = time.perf_counter()
        ai = []
        for i in indices:
            note = row["notes"][i]
            estimator = models[head_key(row, i)]["model"]
            value = (
                estimator["model"].predict([note["measurement"]])[0]
                if isinstance(estimator, dict)
                else estimator.predict([note["feature"]])[0]
            )
            ai.append(float(np.clip(value, 0, 2)))
        ai_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        rules = []
        for i in indices:
            note = row["notes"][i]
            head = models[head_key(row, i)]
            rules.append(
                note["rule_grade"]
                if "rule_grade" in note
                else three_training.rule_grade(note, head["boundary"])
            )
        rule_times.append((time.perf_counter() - start) * 1000)
        if not np.allclose(
            ai, [expected["ai_levels"][i] for i in indices]
        ) or rules != [expected["rule_levels"][i] for i in indices]:
            raise ValueError(
                "Timing implementation differs from canonical prediction"
            )
    return dict(
        id=row["id"],
        exercise=row["exercise"],
        common_item_ids=[row["criterion_ids"][i] for i in indices],
        ai_grading_median_ms=statistics.median(ai_times) if indices else None,
        rule_grading_median_ms=(
            statistics.median(rule_times) if indices else None
        ),
        repeated_same_features_consistent=True,
        repeats=3,
        shared_Heavy_and_feature_cost_excluded=True,
        not_end_to_end_latency=True,
    )


def run() -> dict:
    """3종의 새 학습 반복과 스쿼트 정면 영상을 만들고 전체 디코딩을 검사한다.

    Returns:
        dict: 예제별 원본·참조 등급·AI/규칙 점수·항목·영상 경로.
        예제는 학습 자료이며 독립 성능 시험이 아니다. 관절 영상은 10Hz다.

    Raises:
        OSError: 입력·모델·출력 접근 실패.
        ValueError: 후보 모델 해시·예제·캐시 연결·디코딩 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    comparison = audit.read_json(ROOT / "comparison.json")
    fingerprints = {
        r["exercise"]: r["model_sha256"] for r in comparison["exercises"]
    }
    base, extra = load()
    ready = {r["id"]: r for r in extra}
    annotations = audit.read_json(
        ROOT / "full_repetition_score_annotations_0.json"
    )
    extraction = {
        r["id"]: r for r in audit.read_json(ROOT / "extraction.json")["records"]
    }
    records = []
    for exercise in ["pushups", "squats", "shoulder"]:
        row = next(
            r
            for r in annotations
            if r["exercise"] == exercise and r["id"] in ready
        )
        model_path = ROOT / "models" / (exercise + ".joblib")
        if audit.digest(model_path) != fingerprints[exercise]:
            raise ValueError("Model fingerprint mismatch")
        prediction = predict(joblib.load(model_path), ready[row["id"]])
        videos = []
        for view in ["primary"] + (["front"] if exercise == "squats" else []):
            folder = ROOT / "previews" / row["id"] / view
            folder.mkdir(parents=True, exist_ok=True)
            cache = pathlib.Path(
                extraction[row["id"]][
                    "extraction" if view == "primary" else "front_extraction"
                ]["cache"]
            )
            name = (
                "side"
                if exercise in ["pushups", "squats"] and view == "primary"
                else "primary"
            )
            link = folder / (name + ".npz")
            if not link.exists():
                link.symlink_to(cache)
            shown = dict(row)
            if view == "front":
                shown.update(video_path=row["front_path"], exercise="shoulder")
            video = render.overlay(
                shown,
                dict(prediction, cache_folder=str(folder), display_unit="/100"),
            )
            cap = cv2.VideoCapture(str(video))
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if count == 0 or count != expected:
                raise ValueError("Preview full decode failed")
            videos.append(
                dict(view=view, path=str(video), decoded_frames=count)
            )
        records.append(
            dict(
                id=row["id"],
                exercise=exercise,
                split="train",
                not_independent_evaluation=True,
                reference=row["assessment"],
                prediction=prediction,
                notes=ready[row["id"]]["notes"],
                videos=videos,
            )
        )
    result = dict(
        records=records,
        browser_playback_verified=False,
        all_frames_decoded=True,
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
    )
    timings = []
    for exercise in ["pushups", "squats", "shoulder", "plank", "ytw"]:
        models = joblib.load(ROOT / "models" / (exercise + ".joblib"))
        timings.extend(
            profile(models, row)
            for row in base + extra
            if row["exercise"] == exercise and row["split"] == "validation"
        )
    result["fixed_development_validation_grading_profile"] = timings
    audit.save_json(ROOT / "preview_manifest.json", result)
    return dict(
        examples=len(records),
        videos=sum(len(r["videos"]) for r in records),
        all_frames_decoded=True,
    )


if __name__ == "__main__":
    print(json.dumps(run()))
