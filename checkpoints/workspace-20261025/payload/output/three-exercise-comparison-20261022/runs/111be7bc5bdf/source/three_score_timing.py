"""같은 관절 특징에서 AI와 규칙의 순수 채점 지연을 따로 측정한다.

디코딩·Heavy·특징 생성은 두 방법이 공유하므로 별도 end-to-end 결과로
보고한다. 이 측정에서는 동일 예측 확인 후 실행 순서를 번갈아 20회 반복한다.
"""

import time

import joblib
import numpy as np

import audit
import three_items
import three_pipeline
import three_protocol
import three_training
from resources import check_resources


def score_ai(model: dict, row: dict) -> tuple[list, float | None]:
    """이미 준비된 항목 특징에서 학습 AI만 추론한다.

    Args:
        model: 해당 운동의 학습 모델 사전.
        row: 관측 특징과 100점 배점이 있는 같은 영상 입력.

    Returns:
        tuple[list, float | None]: 0~2 항목 예측과 총점 또는 미판정.

    Raises:
        ValueError: 모델 입력 또는 추론이 잘못된 경우.
    """
    levels = []
    for head, note in zip(model["heads"], row["notes"], strict=True):
        value = None
        if head is not None and note["eligible"]:
            value = (
                head["model"].predict([note["measurement"]])[0]
                if isinstance(head, dict) and head["kind"] == "isotonic"
                else head.predict([note["feature"]])[0]
            )
        levels.append(
            float(np.clip(value, 0, 2)) if value is not None else None
        )
    return levels, three_items.total(levels, row["weights"])


def score_rule(model: dict, row: dict) -> tuple[list, float | None]:
    """같은 관절 측정에서 명시 조건 규칙만 계산한다.

    Args:
        model: 학습 폴드로 보정한 임계값을 가진 모델 사전.
        row: AI와 같은 관측 항목·측정·배점 입력.

    Returns:
        tuple[list, float | None]: 0·1·2 규칙 등급과 총점 또는 미판정.

    Raises:
        ValueError: 경계 또는 입력 계약 불일치.
    """
    levels = [
        (
            three_training.rule_grade(note, boundary)
            if boundary is not None
            else None
        )
        for note, boundary in zip(row["notes"], model["rules"], strict=True)
    ]
    return levels, three_items.total(levels, row["weights"])


def run() -> dict:
    """운동별 개발 입력 최대 12개의 채점 전용 지연 분포를 저장한다.

    Returns:
        dict: 각 방법의 반복 수·표본 수·밀리초 중앙값·90백분위.
        새 관절 추출·실제 재촬영·실시간 처리 성능을 의미하지 않는다.

    Raises:
        OSError: 모델·특징·결과 접근 실패.
        ValueError: 분리 구현이 공통 예측과 일치하지 않는 경우.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    report = audit.read_json(root / "development_results.json")
    inputs = three_pipeline.load()
    records = []
    for result in report["exercises"]:
        check_resources()
        model = joblib.load(result["model_path"])["model"]
        ids = [
            p["id"]
            for p in (
                result["validation_predictions"]
                or result["training_person_out_predictions"]
            )
        ][:12]
        rows = [next(r for r in inputs if r["id"] == id) for id in ids]
        times = dict(ai=[], rule=[])
        for row in rows:
            expected = three_training.predict(model, row)
            for name, function in [("ai", score_ai), ("rule", score_rule)]:
                levels, total = function(model, row)
                if (
                    levels != expected[name + "_levels"]
                    or total != expected[name + "_score"]
                ):
                    raise ValueError(
                        "Separated score differs from common prediction"
                    )
            for iteration in range(20):
                check_resources() if iteration % 10 == 0 else None
                methods = [("ai", score_ai), ("rule", score_rule)]
                if iteration % 2:
                    methods.reverse()
                for name, function in methods:
                    start = time.perf_counter()
                    function(model, row)
                    times[name].append((time.perf_counter() - start) * 1000)
        records.append(
            dict(
                exercise=result["exercise"],
                input_ids=ids,
                repeats_per_input=20,
                shared_input_verified=True,
                methods={
                    name: dict(
                        measurements=len(values),
                        p50_ms=float(np.median(values)) if values else None,
                        p90_ms=(
                            float(np.percentile(values, 90)) if values else None
                        ),
                    )
                    for name, values in times.items()
                },
            )
        )
    output = dict(
        model_version=report["label_version"],
        records=records,
        includes="prediction and weighted total only",
        excludes="decode, Heavy, feature generation, model load",
        independent_test=False,
        cpu_quota_percent=50,
    )
    audit.save_json(root / "score_timing.json", output)
    return output


if __name__ == "__main__":
    print(len(run()["records"]))
