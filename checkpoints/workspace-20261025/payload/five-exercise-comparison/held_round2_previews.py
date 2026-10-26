"""새 보강 구간의 그룹 교차검증 점수와 관절 영상을 연결한다."""

import json
import pathlib
import statistics
import time

import cv2
import joblib
import numpy as np

import audit
import recovery_w_phase
import render
import ytw_projected_rule_trial
from allfive_round1_train import manifest, predict
from allfive_round1_report import profile
from held_round2 import OUT
from held_round2_train import added_rows
from resources import check_resources


def build() -> dict:
    """운동당 한 구간의 누출 없는 개발 예측 표시와 추론 시간을 저장한다.

    Returns:
        dict: 5개 MP4·항목별 점수·동일 입력 반복 추론 시간(ms).
        영상의 점수는 그룹 폴드 외 예측이고 속도는 전체 train 모델의
        캐시 특징 추론만 측정한다. 관절 추출·특징 계산·I/O는 제외한다.

    Raises:
        ValueError: 모델·주석·관측 캐시 형상 또는 표시 영상 디코딩 실패.
        OSError: 저장된 결과·모델 접근 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    comparison = audit.read_json(OUT / "comparison.json")
    rows = {r["id"]: r for r in added_rows()}
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    receipts = {
        r["id"] + "_held2_score": r
        for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    reviews = {
        r["id"] + "_held2_score": r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    reports = []
    for result in comparison["exercises"]:
        exercise = result["exercise"]
        candidates = [
            p for p in result["group_cv_predictions"] if p["id"] in rows
        ]
        chosen = max(
            candidates,
            key=lambda p: (
                p["ai_score"] is not None and p["rule_score"] is not None,
                sum(x is not None for x in p["rule_levels"]),
            ),
        )
        identifier = chosen["id"]
        receipt = receipts[identifier]
        folder = OUT / "previews" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(receipt["extraction"]["cache"]) as source:
            data = {k: source[k].copy() for k in source.files}
        for first, last in reviews[identifier].get(
            "unreliable_frame_intervals", []
        ):
            frames = data["frame_numbers"]
            data["image_xy"][(frames >= first) & (frames <= last)] = np.nan
        primary = (
            "side" if exercise in ("pushups", "squats", "plank") else "primary"
        )
        np.savez_compressed(folder / (primary + ".npz"), **data)
        view = dict(chosen, cache_folder=str(folder), rule_details={})
        if exercise == "ytw":
            view["rule_details"] = dict(
                evidence=dict(
                    phase_detection=recovery_w_phase.detect(data["image_xy"]),
                    w_shape=dict(
                        morphology=dict(grade=chosen["rule_levels"][2])
                    ),
                )
            )
        video = render.overlay(
            annotations[identifier], view, resource_checker=check_resources
        )
        cap = cv2.VideoCapture(str(video))
        count, poster = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if poster is None:
                poster = frame
            count += 1
        cap.release()
        if count != len(data["frame_numbers"]) or poster is None:
            raise ValueError("Preview output frame count differs")
        cv2.imwrite(str(folder / "poster.jpg"), poster)
        model = joblib.load(OUT / "models" / (exercise + ".joblib"))
        row = rows[identifier]
        if exercise == "ytw":
            value = manifest([row])[0]
            function = lambda: ytw_projected_rule_trial.predict(model, value)
        else:
            function = lambda: predict(model, row)
        function()
        times, outputs = [], []
        for _ in range(3):
            start = time.perf_counter()
            outputs.append(function())
            times.append((time.perf_counter() - start) * 1000)
        if exercise != "ytw":
            separate_speed = profile(model, row)
        else:
            indices = [
                i
                for i in range(4)
                if outputs[0]["ai_levels"][i] is not None
                and outputs[0]["rule_levels"][i] is not None
            ]
            ai_times, rule_times = [], []
            for _ in range(3):
                start = time.perf_counter()
                ai = [
                    float(
                        model[i]["model"].predict(
                            [value["items"][i]["feature"]]
                        )[0]
                    )
                    for i in indices
                ]
                ai_times.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                rules = [
                    int(
                        ytw_projected_rule_trial.grades(
                            np.asarray([value["items"][i]["feature"]]),
                            model[i]["policy"],
                            i,
                        )[0]
                    )
                    for i in indices
                ]
                rule_times.append((time.perf_counter() - start) * 1000)
                if not np.allclose(
                    ai, [outputs[0]["ai_levels"][i] for i in indices]
                ) or rules != [outputs[0]["rule_levels"][i] for i in indices]:
                    raise ValueError(
                        "YTW profiling differs from saved prediction"
                    )
            separate_speed = dict(
                common_items=len(indices),
                ai_median_ms=statistics.median(ai_times),
                rule_median_ms=statistics.median(rule_times),
                Heavy_features_decode_excluded=True,
            )
        reports.append(
            dict(
                exercise=exercise,
                prediction=chosen,
                video=str(video),
                decoded_frames=count,
                poster=str(folder / "poster.jpg"),
                cached_joint_model_and_rule_median_ms=statistics.median(times),
                identical_input_repeat_deterministic=outputs[0]
                == outputs[1]
                == outputs[2],
                speed_excludes_extraction_feature_and_io=True,
                separate_speed=separate_speed,
                independent_final_test=False,
            )
        )
    report = dict(records=reports, completed=True)
    audit.save_json(OUT / "previews.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
