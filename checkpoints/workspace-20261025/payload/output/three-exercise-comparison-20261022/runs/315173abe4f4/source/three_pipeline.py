"""3종의 관측 가능 부분 라벨을 추출·학습·개발 검증하는 로컬 실행기다.

기존 플랭크·YTW 파일을 쓰지 않고 새 출력에만 모델·예측을 저장한다.
원본 QEVD JSON 구조의 점수 확장을 읽고 사람·원본별 누출을 검사한다.
"""

import argparse
import hashlib
import pathlib
import time

import joblib
import cv2
import numpy as np

import audit
import benchmark
import extract
import rule_score
import three_items
import three_phases
import three_protocol
import three_training
import three_qevd_export
from resources import check_resources


def records() -> list[dict]:
    """기존 변환본과 별도 추가 직접 초안을 내부 계산 입력으로 반환한다.

    Returns:
        list[dict]: QEVD 기본 필드·직접 등급을 가진 내부 레코드.
        test 분할은 추출·학습·개발 검증 단계에서 사용하지 않는다.

    Raises:
        OSError: 필수 JSON 접근 실패.
        ValueError: JSON 계약 또는 중복 ID 오류.
    """
    three_qevd_export.export()
    rows = audit.read_json(
        three_protocol.OUT / "qevd-json/score_annotations.json"
    )
    result = []
    for row in rows:
        three_protocol.validate_record(row)
        if row["split"] == "test":
            continue
        local = dict(row, **row["assessment"])
        local["qevd_video_path"] = row["video_path"]
        local["video_path"] = row["local_video_path"]
        result.append(local)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate assessment IDs")
    return result


def compatible(row: dict) -> bool:
    """라벨 구도와 해당 운동 입력의 일치를 보수적으로 확인한다.

    Args:
        row: 직접 등급과 촬영 구도·출처가 있는 레코드.

    Returns:
        bool: 구도 확인 가능한 자료 여부. QEVD의 구도 미확인 부분 라벨은
        재검토 전에는 측면용 특징에 섞지 않는다.
    """
    if not row.get("eligible_for_item_training", False) and not row.get(
        "eligible_for_total_score_training", False
    ):
        return False
    if not any(g is not None for g in row["criterion_levels"]):
        return False
    if row["exercise"] == "squats":
        return bool(row.get("front_path"))
    suffix = "_side" if row["exercise"] == "pushups" else "_front"
    return (
        row["id"].endswith(suffix)
        or bool(row.get("eligible_for_total_score_training"))
        or row.get("direct_view_verified", False)
    )


def cache_rows() -> dict:
    """구도 일치 자료의 관절 캐시를 검증해 재사용하거나 Heavy로 추출한다.

    Returns:
        dict: 원본별 캐시 위치·보류 이유·추출 수. 자동 처리이며 직접 검토 아님.

    Raises:
        OSError: 원본·캐시 접근 실패.
        ValueError: 원본 해시·영상 구간 계약 불일치.
        RuntimeError: 관절 추출·온도·전원·메모리 보호 실패.
    """
    root = three_protocol.OUT
    report = dict(records=[], excluded=[], completed=False)
    for row in records():
        if not compatible(row):
            report["excluded"].append(
                dict(
                    id=row["id"],
                    reason="label_or_view_not_verified_for_this_input",
                )
            )
            continue
        check_resources()
        row["required_views"] = (
            ["side", "synchronized_front"]
            if row["exercise"] == "squats"
            else ["side"] if row["exercise"] == "pushups" else ["front"]
        )
        views = [
            (
                "primary" if row["exercise"] == "shoulder" else "side",
                row["video_path"],
            )
        ]
        if row["exercise"] == "squats":
            row["front_sha256"] = audit.digest(pathlib.Path(row["front_path"]))
            views.append(("front", row["front_path"]))
        paths = {}
        for view, path in views:
            previous = audit.OUT / "heavy" / row["id"] / (view + ".npz")
            reusable = False
            if previous.exists():
                try:
                    benchmark.validate_cache(previous, row, view)
                    reusable = True
                except ValueError:
                    reusable = False
            if reusable:
                paths[view] = dict(
                    path=str(previous), existing_baseline_reused=True
                )
            else:
                result = extract.extract_view(
                    row, view, path, destination=root / "heavy"
                )
                paths[view] = dict(
                    path=str(root / "heavy" / row["id"] / (view + ".npz")),
                    existing_baseline_reused=False,
                    extraction=result,
                )
        report["records"].append(dict(id=row["id"], paths=paths))
        audit.save_json(root / "cache_manifest.json", report)
    report["completed"] = True
    audit.save_json(root / "cache_manifest.json", report)
    return report


def load() -> list[dict]:
    """검증한 캐시의 같은 프레임을 두 방식의 공통 입력으로 만든다.

    Returns:
        list[dict]: 좌표·항목 특징·직접 등급·참여자·원본 식별자가 있는 입력.

    Raises:
        OSError: 라벨·캐시 읽기 실패.
        ValueError: 캐시 계약 또는 동기화 프레임 불일치.
    """
    paths = {
        r["id"]: r["paths"]
        for r in audit.read_json(three_protocol.OUT / "cache_manifest.json")[
            "records"
        ]
    }
    ready = []
    dimensions = {}

    def bounded(sequence, path):
        if path not in dimensions:
            cap = cv2.VideoCapture(
                path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            try:
                dimensions[path] = (cap.get(3), cap.get(4))
            finally:
                cap.release()
        return three_items.mask_image_border(sequence, *dimensions[path])

    for row in records():
        if row["id"] not in paths:
            continue
        views = paths[row["id"]]
        name = "primary" if row["exercise"] == "shoulder" else "side"
        primary_path = pathlib.Path(views[name]["path"])
        benchmark.validate_cache(primary_path, row, name)
        with np.load(primary_path) as data:
            primary, frames = (
                data["image_xy"].copy(),
                data["frame_numbers"].copy(),
            )
        primary = bounded(primary, row["video_path"])
        front = None
        if "front" in views:
            row["front_sha256"] = audit.digest(pathlib.Path(row["front_path"]))
            front_path = pathlib.Path(views["front"]["path"])
            benchmark.validate_cache(front_path, row, "front")
            with np.load(front_path) as data:
                if not np.array_equal(frames, data["frame_numbers"]):
                    raise ValueError("Synchronized landmark frames mismatch")
                front = data["image_xy"].copy()
            front = bounded(front, row["front_path"])
        notes = three_items.apply_view_policy(
            row["exercise"],
            three_items.describe(row["exercise"], primary, front),
            row.get("required_input_views", []),
        )
        ready.append(
            dict(
                id=row["id"],
                exercise=row["exercise"],
                person=row["person"],
                source=row.get("source_original_sha256") or row["video_sha256"],
                sources={
                    s
                    for s in [
                        row["video_sha256"],
                        row.get("source_original_sha256"),
                        row.get("front_sha256"),
                    ]
                    if s
                },
                split=row["split"],
                primary=primary,
                front=front,
                frames=frames,
                notes=notes,
                levels=row["criterion_levels"],
                weights=row["criterion_weights"],
                criterion_ids=row["criterion_ids"],
                manifest=row,
            )
        )
    return ready


def metrics(rows: list[dict], key: str) -> dict:
    """같은 입력의 판정률과 직접 총점이 있는 자료의 오차를 구분해 반환한다.

    Args:
        rows: 예측과 target_score를 가진 레코드 목록.
        key: ai_score·rule_score·baseline_score 등의 예측 키.

    Returns:
        dict: 수량·판정률·MAE·P90·만점/비만점 그룹 오차. 점 단위.
        미판정·라벨 없음은 오차 0으로 채우지 않는다.
    """
    paired = [
        r
        for r in rows
        if r["target_score"] is not None and r.get(key) is not None
    ]
    errors = [abs(r[key] - r["target_score"]) for r in paired]
    group_errors = {}
    for group, normal in [("full_score", True), ("nonfull_score", False)]:
        values = [
            abs(r[key] - r["target_score"])
            for r in paired
            if (r["target_score"] == 100) == normal
        ]
        group_errors[group] = dict(
            records=len(values),
            MAE_points=float(np.mean(values)) if values else None,
        )
    return dict(
        submitted=len(rows),
        assessed=sum(r.get(key) is not None for r in rows),
        assessed_fraction=(
            sum(r.get(key) is not None for r in rows) / len(rows)
            if rows
            else None
        ),
        labeled_pairs=len(paired),
        MAE_points=float(np.mean(errors)) if errors else None,
        P90_absolute_error_points=(
            float(np.percentile(errors, 90)) if errors else None
        ),
        groups=group_errors,
    )


def evaluate(model: dict, row: dict) -> dict:
    """동일 입력에서 항목 AI·명시 규칙과 점수·관측·반복 일관성을 측정한다.

    Args:
        model: 학습 폴드에서만 생성한 항목별 모델·규칙 경계.
        row: load의 한 자료. 점수 초안은 오차 계산에만 사용한다.

    Returns:
        dict: 점 단위 예측·오차 계산용 초안·보류·동일 입력 반복 결과.

    Raises:
        ValueError: 입력 운동 또는 항목 계약 불일치.
    """
    start = time.perf_counter()
    first = three_training.predict(model, row)
    elapsed = (time.perf_counter() - start) * 1000
    repeated = three_training.predict(model, row)
    if first != repeated:
        raise ValueError("Same input produced different scores")
    phase = three_phases.detect(row["exercise"], row["primary"], row["frames"])
    automatic = dict(
        ai_score=None,
        rule_score=None,
        same_reference_attempt=False,
        phase=phase,
    )
    if len(phase["repetitions"]) == 1:
        repetition = phase["repetitions"][0]
        a, b = repetition["start_index"], repetition["end_index"] + 1
        overlap = (b - a) / len(row["primary"])
        automatic["reference_envelope_sample_overlap"] = overlap
        if overlap >= 0.6:
            primary = row["primary"][a:b]
            front = row["front"][a:b] if row["front"] is not None else None
            selected = dict(
                row,
                notes=three_items.apply_view_policy(
                    row["exercise"],
                    three_items.describe(row["exercise"], primary, front),
                    row["manifest"].get("required_input_views", []),
                ),
            )
            automatic.update(
                three_training.predict(model, selected),
                same_reference_attempt=True,
            )
    return dict(
        id=row["id"],
        person=row["person"],
        exercise=row["exercise"],
        target_levels=row["levels"],
        target_score=three_items.total(row["levels"], row["weights"]),
        **first,
        reference_unknown_items_with_prediction=[
            row["criterion_ids"][i]
            for i, grade in enumerate(row["levels"])
            if grade is None
            and (
                first["ai_levels"][i] is not None
                or first["rule_levels"][i] is not None
            )
        ],
        baseline_score=model["training_mean_score"],
        same_input_repeat_max_delta_points=0,
        scoring_only_ms=elapsed,
        observations=[
            {k: v for k, v in n.items() if k != "feature"} for n in row["notes"]
        ],
        automatic_within_manual_envelope=automatic,
    )


def paired_metrics(rows: list[dict]) -> dict:
    """AI·규칙·평균 기준선을 모두 계산한 동일 표본에서 오차를 비교한다.

    Args:
        rows: 직접 총점과 각 방법의 예측을 가진 평가 목록.

    Returns:
        dict: 세 방법의 공통 표본 수·ID·오차. 전체 판정률은 별도 지표다.
        한 방법의 미판정을 오차 0 또는 평균값으로 채우지 않는다.
    """
    keys = ("ai_score", "rule_score", "baseline_score")
    paired = [
        row
        for row in rows
        if row["target_score"] is not None
        and all(row.get(key) is not None for key in keys)
    ]
    return dict(
        submitted=len(rows),
        paired_records=len(paired),
        paired_ids=[row["id"] for row in paired],
        metrics={key: metrics(paired, key) for key in keys},
        coverage_must_use_all_submitted_rows=True,
    )


def train() -> dict:
    """참여자·원본 제외 개발 비교와 고정 검증 비교를 수행하고 후보를 저장한다.

    Returns:
        dict: 3종의 항목별 학습 수·개발 오차·판정률·후보 모델 경로.
        최종 시험은 읽지 않으며 공통 모델을 자동 교체하지 않는다.

    Raises:
        OSError: 결과·모델 저장 실패.
        ValueError: 참여자·원본 누출 또는 점수 계약 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    rows = load()
    label_paths = [
        root / "score_annotations.json",
        root / "additional_score_annotations.json",
    ]
    code_paths = [
        pathlib.Path(__file__).with_name(name)
        for name in [
            "three_pipeline.py",
            "three_items.py",
            "three_training.py",
            "three_phases.py",
            "three_qevd_export.py",
            "three_protocol.py",
        ]
    ]
    version = hashlib.sha256(
        "".join(
            audit.digest(p) for p in label_paths + code_paths if p.exists()
        ).encode()
    ).hexdigest()[:12]
    run_folder = root / "runs" / version
    run_folder.mkdir(parents=True, exist_ok=True)
    report = dict(
        exercises=[],
        independent_final_exam_evaluated=False,
        expert_reference=False,
        common_models_replaced=False,
        label_targets="direct_nonexpert_experimental_grades_only",
        rule_calibration="training_groups_only",
    )
    for exercise in three_protocol.EXERCISES:
        subset = [r for r in rows if r["exercise"] == exercise]
        cohort = "legacy_and_new_same_known_identity_space"
        if exercise == "squats" and any(
            r["manifest"].get("provenance", {}).get("dataset") == "REHAB24"
            for r in subset
        ):
            # 두 REHAB 버전의 익명·실명 참여자 대응이 없어 서로 다른 폴드로 섞지 않는다.
            subset = [
                r
                for r in subset
                if r["manifest"].get("provenance", {}).get("dataset")
                == "REHAB24"
            ]
            cohort = "REHAB24_only_no_cross_dataset_identity_assumptions"
        fitting = [r for r in subset if r["split"] == "train"]
        validation = [r for r in subset if r["split"] == "validation"]
        if {r["person"] for r in fitting} & {
            r["person"] for r in validation
        } or set().union(*(r["sources"] for r in fitting)) & set().union(
            *(r["sources"] for r in validation)
        ):
            raise ValueError("Training/validation person or source overlap")
        check_resources()
        model = three_training.fit(fitting)
        directory = run_folder / "models" / exercise
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            dict(
                model=model,
                train_ids=[r["id"] for r in fitting],
                json_schema=three_protocol.SCHEMA,
                input="MediaPipe_Heavy_observed_item_geometry",
                expert_validated=False,
            ),
            directory / "candidate.joblib",
        )
        validation_predictions = [evaluate(model, row) for row in validation]
        oof = []
        for person in sorted({r["person"] for r in fitting}):
            check_resources()
            held = [r for r in fitting if r["person"] == person]
            sources = set().union(*(r["sources"] for r in held))
            inner = [
                r
                for r in fitting
                if r["person"] != person and not sources & r["sources"]
            ]
            if len(inner) < 2:
                continue
            folded = three_training.fit(inner)
            oof.extend(evaluate(folded, row) for row in held)
        item = dict(
            exercise=exercise,
            cohort=cohort,
            train_segments=len(fitting),
            train_people=len({r["person"] for r in fitting}),
            item_training=model["item_training"],
            rule_boundaries=model["rules"],
            validation_predictions=validation_predictions,
            validation_metrics={
                key: metrics(validation_predictions, key)
                for key in ("ai_score", "rule_score", "baseline_score")
            },
            training_person_out_predictions=oof,
            training_person_out_metrics={
                key: metrics(oof, key)
                for key in ("ai_score", "rule_score", "baseline_score")
            },
            validation_paired_comparison=paired_metrics(validation_predictions),
            training_person_out_paired_comparison=paired_metrics(oof),
            model_path=str(directory / "candidate.joblib"),
            model_sha256=audit.digest(directory / "candidate.joblib"),
            same_input_frames_for_both_methods=True,
            timing_excludes_Heavy=True,
            acceptance_complete=False,
        )
        report["exercises"].append(item)
        report["label_version"] = version
        report["label_sha256"] = {
            p.name: audit.digest(p) for p in label_paths if p.exists()
        }
        report["feature_version"] = three_items.VERSION
        report["code_sha256"] = {p.name: audit.digest(p) for p in code_paths}
        audit.save_json(run_folder / "development_results.json", report)
        audit.save_json(root / "development_results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["extract", "train"])
    mode = parser.parse_args().mode
    result = cache_rows() if mode == "extract" else train()
    print(
        "completed",
        mode,
        len(result.get("records", result.get("exercises", []))),
    )
