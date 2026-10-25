"""동일 개발 영상의 Heavy부터 두 점수 계산까지 반복 시간·일관성을 측정한다.

재촬영은 수행하지 않는다. 같은 픽셀의 독립 VIDEO 추출 반복이며 CPU
50% 제한과 GPU 조건을 기록한다. 운영 환경 지연이나 사람 재현성은 아니다.
"""

import copy
import datetime
import pathlib
import time

import joblib
import numpy as np

import audit
import benchmark
from extract import extract_view
import rule_score
from resources import check_resources


def run(repetitions: int = 3) -> dict:
    """5종 개발 영상 각 한 구간을 반복 추출하고 공유 비용을 나눠 기록한다.

    Args:
        repetitions: 운동별 독립 관절 추출 반복 수. 2~3 정수.

    Returns:
        dict: 캐시를 재사용하지 않은 추출 시간, AI·규칙 계산 시간,
        관측 보류, 점수 변동, 좌표 변동과 각 단계의 완료 상태.
        Y·전경 등 공통 관측 계약이 미완료면 양쪽 승인 총점은 None이다.

    Raises:
        OSError: 입력·모델·관절·결과 접근 실패.
        ValueError: 반복 수·특징·관절 캐시 계약 불일치 또는 시간 재사용.
        RuntimeError: 온도·전원·메모리 보호 실패.
    """
    if repetitions not in (2, 3):
        raise ValueError("Two or three repetitions required")
    path = audit.OUT / "autonomous_runtime_results.json"
    report = (
        audit.read_json(path)
        if path.exists()
        else dict(
            started_at=datetime.datetime.now().astimezone().isoformat(),
            records=[],
            independent_final_test=False,
            real_recapture_consistency_measured=False,
            repeated_same_pixels=True,
            CPUQuota_percent=50,
            model_load_excluded=True,
            repetitions=repetitions,
            completed=False,
        )
    )
    manifests = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "development_manifest.json")
    }
    comparison = audit.read_json(audit.OUT / "comparison.json")["results"]
    cases = []
    for exercise in audit.VIEWS:
        if exercise == "plank":
            export = next(
                r
                for r in audit.read_json(
                    audit.OUT / "plank_prospective_plan_exports.json"
                )["records"]
                if r["id"] == "climber_prospective"
            )
            row = dict(
                id=export["id"],
                exercise="plank",
                video_path=export["video_path"],
                video_sha256=export["video_sha256"],
                segment=dict(
                    start_frame=0,
                    end_frame=export["frames"] - 1,
                    fps=export["fps"],
                ),
                required_views=["side"],
            )
            model = audit.OUT / "plank_balance_trial/balanced_leaf2.joblib"
            ids = [
                "trunk_line",
                "elbow_sagittal_position",
                "leg_and_body_support",
                "support_continuity",
            ]
            weights = [40, 25, 20, 15]
        else:
            chosen = comparison[exercise]["comparisons"][0]["id"]
            row = manifests[chosen]
            label = next(
                r
                for r in audit.read_json(audit.OUT / "labels.json")
                if r["id"] == chosen
            )
            ids, weights = label["criterion_ids"], label["criterion_weights"]
            model = audit.OUT / (exercise + "_experimental.joblib")
        cases.append((exercise, row, model, ids, weights))
    for exercise, original, model_path, ids, weights in cases:
        bundle = joblib.load(model_path)
        for trial in range(repetitions):
            if any(
                r["exercise"] == exercise and r["trial"] == trial
                for r in report["records"]
            ):
                continue
            state = check_resources()
            cooled = time.monotonic()
            while state["cpu_c"] >= 70 or state["gpu_c"] >= 65:
                if time.monotonic() - cooled > 180:
                    raise RuntimeError("Cooling timeout; checkpoint preserved")
                time.sleep(2)
                state = check_resources()
            row = copy.deepcopy(original)
            row["id"] = f"runtime_{exercise}_{trial}_{time.time_ns()}"
            primary = (
                "side"
                if exercise in ("pushups", "squats", "plank")
                else "primary"
            )
            started = time.perf_counter()
            extraction = extract_view(row, primary, row["video_path"])
            if extraction.get("reused"):
                raise ValueError("Runtime reused an extraction cache")
            if exercise == "squats":
                frontal_path = row.get("front_video_path") or row.get(
                    "front_path"
                )
                if not frontal_path:
                    raise ValueError("Missing synchronized frontal input")
                front_extraction = extract_view(row, "front", frontal_path)
                if front_extraction.get("reused"):
                    raise ValueError("Runtime reused front cache")
            shared_seconds = time.perf_counter() - started
            seq, front, holds = benchmark.load_input(row)
            before = time.perf_counter()
            x, names = benchmark.input_features(seq, front)
            if names != bundle["names"]:
                raise ValueError("Runtime feature order mismatch")
            levels = np.clip(bundle["model"].predict([x])[0], 0, 2)
            raw_ai_score = float(np.dot(levels, weights) / 2)
            ai_ms = (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            rule = rule_score.evaluate(exercise, seq, front, ids, weights)
            rule_ms = (time.perf_counter() - before) * 1000
            approved = not holds and rule["score"] is not None
            report["records"].append(
                dict(
                    exercise=exercise,
                    trial=trial,
                    source_id=original["id"],
                    cache_id=row["id"],
                    cache_folder=str(audit.OUT / "heavy" / row["id"]),
                    source_sha256=original["video_sha256"],
                    model_sha256=audit.digest(model_path),
                    shared_decode_Heavy_cache_write_seconds=shared_seconds,
                    ai_feature_predict_ms=ai_ms,
                    rule_ms=rule_ms,
                    ai_total_pipeline_seconds=shared_seconds + ai_ms / 1000,
                    rule_total_pipeline_seconds=shared_seconds + rule_ms / 1000,
                    ai_score=raw_ai_score if approved else None,
                    rule_score=rule["score"] if approved else None,
                    raw_ai_score_diagnostic_only=raw_ai_score,
                    common_observation_holds=holds,
                    rule_hold_items=rule["hold_items"],
                    same_input_total_approved=approved,
                    resources=check_resources(),
                )
            )
            audit.save_json(path, report)
            print(exercise, trial, round(shared_seconds, 2), flush=True)
    summaries = []
    for exercise, _, _, _, _ in cases:
        rows = [r for r in report["records"] if r["exercise"] == exercise]
        seqs = []
        for r in rows:
            primary = (
                "side"
                if exercise in ("pushups", "squats", "plank")
                else "primary"
            )
            with np.load(
                pathlib.Path(r["cache_folder"]) / (primary + ".npz")
            ) as cache:
                seqs.append(cache["image_xy"].copy())
        differences = []
        mask_disagreements = []
        for seq in seqs[1:]:
            if seq.shape != seqs[0].shape:
                raise ValueError("Repeat frame shape differs")
            valid = np.isfinite(seq).all(2) & np.isfinite(seqs[0]).all(2)
            mask_disagreements.append(
                float(
                    np.mean(
                        np.isfinite(seq).all(2) != np.isfinite(seqs[0]).all(2)
                    )
                )
            )
            if valid.any():
                differences.extend(
                    np.linalg.norm(seq - seqs[0], axis=2)[valid].tolist()
                )
        scores = [r["ai_score"] for r in rows if r["ai_score"] is not None]
        rules = [r["rule_score"] for r in rows if r["rule_score"] is not None]
        summaries.append(
            dict(
                exercise=exercise,
                repetitions=len(rows),
                paired_assessed=len(scores),
                ai_pipeline_median_seconds=float(
                    np.median([r["ai_total_pipeline_seconds"] for r in rows])
                ),
                rule_pipeline_median_seconds=float(
                    np.median([r["rule_total_pipeline_seconds"] for r in rows])
                ),
                ai_score_std=float(np.std(scores)) if scores else None,
                rule_score_std=float(np.std(rules)) if rules else None,
                coordinate_difference_median_pixels=(
                    float(np.median(differences)) if differences else None
                ),
                coordinate_difference_max_pixels=(
                    max(differences) if differences else None
                ),
                observed_mask_disagreement_mean_fraction=(
                    float(np.mean(mask_disagreements))
                    if mask_disagreements
                    else None
                ),
                observed_mask_disagreement_max_fraction=(
                    max(mask_disagreements) if mask_disagreements else None
                ),
            )
        )
    report.update(
        summaries=summaries,
        completed=True,
        finished_at=datetime.datetime.now().astimezone().isoformat(),
    )
    audit.save_json(path, report)
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>5종 반복 처리 시간</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:24px}td,th{padding:10px;border-bottom:1px solid #aaa}</style><h1>같은 영상의 반복 처리 시간·일관성</h1><p><a href="unified_status.html">통합 현황</a> · <a href="autonomous_runtime_results.json">원본 결과</a></p><p>CPU 50% 제한, GPU Heavy, 모델 사전 로드 조건입니다. 원본 디코딩·관절 추출·캐시 저장과 각 점수 계산을 포함합니다. 같은 픽셀의 반복 처리이며 실제 재촬영·서비스 응답 지연을 측정한 것은 아닙니다.</p><table><tr><th>운동</th><th>공통 평가 / 반복</th><th>AI / 규칙 처리 중앙값(초)</th><th>AI / 규칙 점수 표준편차</th></tr>'
    ]
    for row in summaries:
        parts.append(
            f'<tr><td>{row["exercise"]}</td><td>{row["paired_assessed"]} / {row["repetitions"]}</td><td>{row["ai_pipeline_median_seconds"]:.3f} / {row["rule_pipeline_median_seconds"]:.3f}</td><td>{row["ai_score_std"]} / {row["rule_score_std"]}</td></tr>'
        )
    parts.append(
        "</table><p>Y–T–W의 관측 계약이 미완료인 항목은 AI에도 동일하게 보류하여 승인 총점을 만들지 않습니다. 처리 시간은 계산이 끝난 시간이며 점수 유효성·정확도와 다릅니다.</p></html>"
    )
    (audit.OUT / "autonomous_runtime.html").write_text(
        "".join(parts), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(run()["summaries"])
