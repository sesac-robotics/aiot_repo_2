"""직접 검토 초안으로 학습한 AI 회귀와 규칙을 개발 검증에서 비교한다.

최종 시험 분할은 읽지 않는다. 입력 특징은 관절 기하·시계열뿐이며,
Rule Base 출력·제공자 문자열·라벨 메모는 학습 입력에 포함하지 않는다.
모델은 검증 전 실험 산출물이고 서비스 모델을 교체하지 않는다.
"""

import collections
import hashlib
import json
import logging
import pathlib
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

import audit
import geometry
import rule_score

LOGGER = logging.getLogger(__name__)


def validate_cache(path: pathlib.Path, row: dict, view: str) -> None:
    """캐시의 버전·구간·원본·설정 해시를 현재 입력 계약과 대조한다.

    Args:
        path: 관절 NPZ 파일의 pathlib.Path 경로.
        row: 원본·구간 해시가 있는 감사 레코드.
        view: side·front·primary 중 저장 시점 이름.

    Returns:
        None: 계약이 일치하면 정상 반환한다.

    Raises:
        OSError: 캐시 또는 메타데이터 파일을 읽을 수 없는 경우.
        ValueError: 버전·원본·구간·파일 해시 연결이 일치하지 않는 경우.
    """
    metadata = audit.read_json(path.with_suffix(".json"))
    config = metadata["config"]
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    source = row["front_sha256"] if view == "front" else row["video_sha256"]
    if (
        config["schema"] != audit.CACHE_SCHEMA
        or config["segment"] != row["segment"]
        or config["source_sha256"] != source
        or metadata["fingerprint"] != fingerprint
    ):
        raise ValueError("Cache version or input contract mismatch")
    with np.load(path) as data:
        if str(data["fingerprint"]) != fingerprint:
            raise ValueError("Cache payload fingerprint mismatch")


def load_input(row: dict) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """캐시에서 두 방식이 함께 사용할 관절과 품질 보류 사유를 반환한다.

    Args:
        row: 감사 manifest 레코드. 주 시점·정면·프레임 구간을 지정한다.

    Returns:
        tuple: 주 좌표 N×33×2, 선택 정면 좌표, 보류 사유 목록.
        좌표는 픽셀이며 누락은 NaN이다.

    Raises:
        OSError: 필요한 관절 캐시를 읽을 수 없는 경우.
        ValueError: 캐시 구간 또는 두 시점의 프레임 번호가 다른 경우.
    """
    folder = audit.OUT / "heavy" / row["id"]
    primary_name = (
        "side"
        if row["exercise"] in ("pushups", "squats", "plank")
        else "primary"
    )
    validate_cache(folder / (primary_name + ".npz"), row, primary_name)
    with np.load(folder / (primary_name + ".npz")) as data:
        primary = data["image_xy"]
        indices = data["frame_numbers"]
    reasons = []
    front = None
    if len(primary) < 5:
        reasons.append("fewer_than_five_samples")
    sides = ([11, 13, 15, 23, 25, 27, 31], [12, 14, 16, 24, 26, 28, 32])
    if row["exercise"] in ("pushups", "squats", "plank"):
        observed = max(
            float(np.isfinite(primary[:, side]).all(2).mean(0).min())
            for side in sides
        )
    else:
        observed = float(
            np.isfinite(primary[:, [11, 12, 13, 14, 15, 16, 23, 24]])
            .all(2)
            .mean(0)
            .min()
        )
    if observed < 0.8:
        reasons.append("primary_required_joint_observation_below_80_percent")
    if row["exercise"] == "squats":
        validate_cache(folder / "front.npz", row, "front")
        with np.load(folder / "front.npz") as data:
            front = data["image_xy"]
            if not np.array_equal(indices, data["frame_numbers"]):
                raise ValueError("Front and side sample frames mismatch")
        observed = float(
            np.isfinite(front[:, [11, 12, 23, 24, 25, 26, 27, 28]])
            .all(2)
            .mean(0)
            .min()
        )
        if observed < 0.8:
            reasons.append("front_required_joint_observation_below_80_percent")
    return primary, front, reasons


def input_features(
    primary: np.ndarray, front: np.ndarray | None
) -> tuple[np.ndarray, list[str]]:
    """주·정면 관절의 같은 시간 특징 벡터와 이름을 반환한다.

    Args:
        primary: N×33×2 주 시점 픽셀 좌표.
        front: 같은 구간 정면 좌표. 없으면 단일 시점이다.

    Returns:
        tuple: 특징 벡터와 동일 순서 이름. 수치 단위는 geometry를 따른다.

    Raises:
        ValueError: 좌표 배열 형상이 잘못된 경우.
    """
    features, names = geometry.model_features(primary)
    names = ["primary_" + name for name in names]
    if front is not None:
        additional, extra_names = geometry.model_features(front)
        features = np.concatenate([features, additional])
        names.extend("front_" + name for name in extra_names)
    return features, names


def grouped_diagnostic(model: Pipeline, train: list[dict]) -> dict:
    """학습 참여자를 한 명씩 제외하여 고정 모델의 개발 오차를 측정한다.

    Args:
        model: 아직 학습하지 않은 고정 설정 회귀 파이프라인.
        train: 관절 특징·직접 검토 등급·참여자와 원본 해시 레코드.

    Returns:
        dict: 참여자별 제외 예측, 점 단위 MAE와 학습 평균 기준선.
        최종 시험이 아니며 이 결과로 설정을 자동 조정하지 않는다.

    Raises:
        ValueError: 제외 참여자의 원본이 학습 폴드에 중복된 경우.
    """
    people = sorted({row["manifest"]["person"] for row in train})
    folds = []
    for person in people:
        fitting = [r for r in train if r["manifest"]["person"] != person]
        held = [r for r in train if r["manifest"]["person"] == person]
        if len(fitting) < 2:
            continue
        hashes = lambda rows: {
            r["manifest"].get(key)
            for r in rows
            for key in (
                "video_sha256",
                "front_sha256",
                "source_original_sha256",
            )
            if r["manifest"].get(key)
        }
        if hashes(fitting) & hashes(held):
            raise ValueError("Source leakage in participant diagnostic")
        fitted = clone(model).fit(
            np.asarray([r["x"] for r in fitting]),
            np.asarray([r["label"]["criterion_levels"] for r in fitting]),
        )
        predicted = np.clip(fitted.predict([r["x"] for r in held]), 0, 2)
        baseline = float(np.mean([r["label"]["score"] for r in fitting]))
        items = []
        for row, levels in zip(held, predicted, strict=True):
            score = float(np.dot(levels, row["label"]["criterion_weights"]) / 2)
            target = row["label"]["score"]
            items.append(
                dict(
                    id=row["manifest"]["id"],
                    target_score=target,
                    predicted_score=score,
                    absolute_error=abs(score - target),
                    training_mean_score=baseline,
                    baseline_absolute_error=abs(baseline - target),
                )
            )
        folds.append(
            dict(
                held_person=person,
                fit_records=len(fitting),
                fit_people=len(people) - 1,
                predictions=items,
            )
        )
    items = [item for fold in folds for item in fold["predictions"]]
    return dict(
        method="leave_one_training_participant_out",
        folds=folds,
        independent_final_test=False,
        hyperparameters_tuned=False,
        samples=len(items),
        people=len(folds),
        ai_MAE_points=(
            float(np.mean([r["absolute_error"] for r in items]))
            if items
            else None
        ),
        baseline_MAE_points=(
            float(np.mean([r["baseline_absolute_error"] for r in items]))
            if items
            else None
        ),
        person_macro_MAE_points=(
            float(
                np.mean(
                    [
                        np.mean(
                            [r["absolute_error"] for r in fold["predictions"]]
                        )
                        for fold in folds
                    ]
                )
            )
            if folds
            else None
        ),
    )


def noise_sensitivity(model: Pipeline, exercise: str, row: dict) -> dict:
    """같은 관절 입력에 1픽셀 잡음을 넣어 점수 변동을 측정한다.

    Args:
        model: 직접 검토 등급으로 학습한 sklearn 회귀 파이프라인.
        exercise: 운동 영어 코드.
        row: 관절 배열·기준 배점·항목 ID가 있는 준비 레코드.

    Returns:
        dict: 8회 동일 잡음 입력에서 각 방식의 점수 표준편차와 평가 수.
        단위는 점이며 실제 재촬영·카메라 변화 일관성을 뜻하지 않는다.

    Raises:
        ValueError: 관절 형상 또는 모델 특징 차원이 일치하지 않는 경우.
    """
    rng = np.random.default_rng(2110)
    ai_values, rule_values = [], []
    for _ in range(8):
        primary = row["primary"] + rng.normal(0, 1, row["primary"].shape)
        front = (
            row["front"] + rng.normal(0, 1, row["front"].shape)
            if row["front"] is not None
            else None
        )
        features, _ = input_features(primary, front)
        levels = np.clip(model.predict([features])[0], 0, 2)
        label = row["label"]
        ai_values.append(float(np.dot(levels, label["criterion_weights"]) / 2))
        result = rule_score.evaluate(
            exercise,
            primary,
            front,
            label["criterion_ids"],
            label["criterion_weights"],
        )
        if result["score"] is not None:
            rule_values.append(result["score"])
    return dict(
        protocol="8_trials_Gaussian_1_pixel_std_observed_coordinates_only",
        ai_std_points=float(np.std(ai_values)),
        rule_std_points=(float(np.std(rule_values)) if rule_values else None),
        ai_assessed=len(ai_values),
        rule_assessed=len(rule_values),
        real_recapture_consistency_measured=False,
    )


def run() -> dict:
    """각 운동의 기존 학습·개발 검증 분할에서 고정 회귀 모델을 비교한다.

    Returns:
        dict: 두 방식의 개발 검증 오차·관측 보류·예측·시간과 모델 상태.

    Raises:
        ValueError: 학습·검증 간 참여자·원본 중복 또는 특징 순서 불일치.
        OSError: 감사 자료·캐시 읽기 또는 보고서·모델 저장 실패.

    Notes:
        독립 최종 시험이나 전문가 정확도 결과가 아니다. 관측률 경계와
        회귀 설정은 이번 검증 결과를 보기 전에 고정한 공학 초안이다.
    """
    expected_heavy_hash = audit.digest(
        audit.BASE / "mediapipe-pilot/models/heavy.task"
    )
    manifests = audit.read_json(audit.OUT / "development_manifest.json")
    labels = audit.read_json(audit.SOURCE / "labels.json")
    extra = audit.OUT / "supplemental_labels.json"
    if extra.exists():
        labels.extend(audit.read_json(extra))
    targets = {row["id"]: row for row in labels}
    report = dict(
        schema="five-exercise-development-comparison-v2",
        cache_schema=audit.CACHE_SCHEMA,
        heavy_model_sha256=expected_heavy_hash,
        benchmark_code_sha256=audit.digest(pathlib.Path(__file__)),
        label_snapshot_sha256=audit.digest(audit.OUT / "labels.json"),
        geometry_code_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/geometry.py"
        ),
        W_morphology_code_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/w_morphology.py"
        ),
        rule_code_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/rule_score.py"
        ),
        independent_final_test=False,
        test_records_used_for_fitting_or_evaluation=False,
        expert_validated=False,
        production_ready=False,
        rule_version=rule_score.VERSION,
        policy=dict(
            min_train_records=8,
            min_train_people=3,
            required_joint_observation=0.8,
            trees=128,
            leaf_samples=2,
            random_state=21021,
        ),
        results={},
        limitations_ko=[
            "기존에 반복 검토한 개발 검증 자료와 AI 실험 초안 라벨의 일치도다.",
            "규칙의 수치 경계·접지 대리값은 아직 검증되지 않았다.",
            "일관성·속도는 작은 로컬 실험이며 전체 운영 성능이 아니다.",
            "새 참여자·촬영 구도·오류 사례와 독립 최종 시험이 필요하다.",
        ],
    )
    for exercise in audit.VIEWS:
        prepared = []
        unavailable = []
        for row in manifests:
            if (
                row["exercise"] != exercise
                or not row["extraction_ready"]
                or row["original_split"] not in ("train", "validation")
            ):
                continue
            try:
                primary, front, reasons = load_input(row)
            except FileNotFoundError:
                unavailable.append(dict(id=row["id"], reason="cache_not_ready"))
                continue
            if reasons:
                unavailable.append(
                    dict(
                        id=row["id"],
                        reasons=reasons,
                        split=row["original_split"],
                    )
                )
                continue
            metadata_paths = (audit.OUT / "heavy" / row["id"]).glob("*.json")
            for metadata_path in metadata_paths:
                if metadata_path.stem not in ("side", "front", "primary"):
                    continue
                if (
                    audit.read_json(metadata_path)["config"]["model_sha256"]
                    != expected_heavy_hash
                ):
                    raise ValueError(
                        "Heavy extractor model changed; rebuild input caches"
                    )
            features, names = input_features(primary, front)
            prepared.append(
                dict(
                    manifest=row,
                    label=targets[row["id"]],
                    primary=primary,
                    front=front,
                    x=features,
                    names=names,
                )
            )
        train = [
            r for r in prepared if r["manifest"]["original_split"] == "train"
        ]
        validation = [
            r
            for r in prepared
            if r["manifest"]["original_split"] == "validation"
        ]
        people = {r["manifest"]["person"] for r in train}
        result = dict(
            train_records=len(train),
            train_people=len(people),
            validation_records=len(validation),
            unavailable=unavailable,
            comparisons=[],
            train_score_distribution=dict(
                collections.Counter(str(r["label"]["score"]) for r in train)
            ),
            validation_score_distribution=dict(
                collections.Counter(
                    str(r["label"]["score"]) for r in validation
                )
            ),
            model_status="insufficient_training_data",
        )
        report["results"][exercise] = result
        if exercise == "plank":
            result["rule_only_previews"] = []
            for row in train:
                label = row["label"]
                rules = rule_score.evaluate(
                    exercise,
                    row["primary"],
                    row["front"],
                    label["criterion_ids"],
                    label["criterion_weights"],
                )
                result["rule_only_previews"].append(
                    dict(
                        id=label["id"],
                        label_score=label["score"],
                        ai_score=None,
                        rule_score=rules["score"],
                        difference=None,
                        rule_details=rules,
                        noise_sensitivity=None,
                        cache_folder=str(audit.OUT / "heavy" / label["id"]),
                        evaluation_role="development_train_rule_preview_not_test",
                        AI_model_available=False,
                    )
                )
        distinct_targets = len({r["label"]["score"] for r in train})
        result["training_blockers"] = []
        if len(train) < 8:
            result["training_blockers"].append(
                "fewer_than_eight_eligible_records"
            )
        if len(people) < 3:
            result["training_blockers"].append(
                "fewer_than_three_source_participant_groups"
            )
        if distinct_targets < 2:
            result["training_blockers"].append("no_score_target_variation")
        if result["training_blockers"]:
            continue
        train_people = {r["manifest"]["person"] for r in train}
        validation_people = {r["manifest"]["person"] for r in validation}
        train_sources = {
            r["manifest"].get(key)
            for r in train
            for key in (
                "video_sha256",
                "front_sha256",
                "source_original_sha256",
            )
            if r["manifest"].get(key)
        }
        validation_sources = {
            r["manifest"].get(key)
            for r in validation
            for key in (
                "video_sha256",
                "front_sha256",
                "source_original_sha256",
            )
            if r["manifest"].get(key)
        }
        if (
            train_people & validation_people
            or train_sources & validation_sources
        ):
            raise ValueError("Participant or original/derived source leakage")
        if len({r["label"]["score"] for r in train}) < 2:
            result["model_status"] = "insufficient_score_diversity"
            continue
        x = np.asarray([r["x"] for r in train])
        y = np.asarray([r["label"]["criterion_levels"] for r in train])
        if any(r["names"] != train[0]["names"] for r in prepared):
            raise ValueError("Feature schema mismatch")
        model = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=21021,
                n_jobs=1,
            ),
        )
        result["training_participant_diagnostic"] = grouped_diagnostic(
            model, train
        )
        started = time.perf_counter()
        model.fit(x, y)
        result["fit_seconds"] = time.perf_counter() - started
        bundle = dict(
            model=model,
            names=train[0]["names"],
            train_ids=[r["manifest"]["id"] for r in train],
            expert_validated=False,
            rubric_version=train[0]["label"]["rubric_version"],
        )
        model_path = audit.OUT / (exercise + "_experimental.joblib")
        joblib.dump(bundle, model_path)
        result["model_sha256"] = audit.digest(model_path)
        result["model_status"] = "unvalidated_development_model"
        result["criterion_grade_counts"] = [
            dict(collections.Counter(str(v) for v in column)) for column in y.T
        ]
        for row in validation:
            label = row["label"]
            ai_times, rule_times = [], []
            for iteration in range(12):
                started = time.perf_counter()
                features, _ = input_features(row["primary"], row["front"])
                levels = np.clip(model.predict([features])[0], 0, 2)
                ai_score = float(np.dot(levels, label["criterion_weights"]) / 2)
                ai_elapsed = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                rules = rule_score.evaluate(
                    exercise,
                    row["primary"],
                    row["front"],
                    label["criterion_ids"],
                    label["criterion_weights"],
                )
                rule_elapsed = (time.perf_counter() - started) * 1000
                if iteration:
                    ai_times.append(ai_elapsed)
                    rule_times.append(rule_elapsed)
            ai_ms = float(np.median(ai_times))
            rule_ms = float(np.median(rule_times))
            folder = audit.OUT / "heavy" / label["id"]
            view_names = (
                ["side", "front"]
                if exercise == "squats"
                else (
                    ["side"]
                    if exercise in ("pushups", "plank")
                    else ["primary"]
                )
            )
            shared_seconds = sum(
                audit.read_json(folder / (view + ".json"))["extract_seconds"]
                for view in view_names
            )
            result["comparisons"].append(
                dict(
                    shared_recorded_extraction_seconds=shared_seconds,
                    recorded_ai_pipeline_seconds=shared_seconds + ai_ms / 1000,
                    recorded_rule_pipeline_seconds=shared_seconds
                    + rule_ms / 1000,
                    pipeline_timing_note_ko="같은 시점별 저장 추출 시간과 반복 점수 계산 중앙값의 합. CPU 제한 변경·초기화·오프라인 구간 읽기를 포함하며 실제 재촬영 또는 서비스 응답 지연 검사가 아님.",
                    id=label["id"],
                    label_score=label["score"],
                    ai_score=ai_score,
                    ai_levels=levels.tolist(),
                    rule_score=rules["score"],
                    rule_details=rules,
                    difference=(
                        ai_score - rules["score"]
                        if rules["score"] is not None
                        else None
                    ),
                    ai_ms=ai_ms,
                    rule_ms=rule_ms,
                    timing_protocol=dict(
                        repetitions=11,
                        warmup=1,
                        ai_includes_feature_calculation=True,
                        shared_joint_cache_read_excluded=True,
                        video_and_Heavy_excluded=True,
                        ai_p95_ms=float(np.percentile(ai_times, 95)),
                        rule_p95_ms=float(np.percentile(rule_times, 95)),
                    ),
                    noise_sensitivity=noise_sensitivity(model, exercise, row),
                    cache_folder=str(audit.OUT / "heavy" / label["id"]),
                    expert_validated=False,
                )
            )
        comparisons = result["comparisons"]
        item_metrics = []
        for item_index, criterion in enumerate(
            train[0]["label"]["criterion_ids"]
        ):
            paired_items = [
                c
                for c in comparisons
                if c["rule_details"]["levels"][item_index] is not None
            ]
            ai_errors = [
                abs(
                    c["ai_levels"][item_index]
                    - targets[c["id"]]["criterion_levels"][item_index]
                )
                for c in paired_items
            ]
            rule_errors = [
                abs(
                    c["rule_details"]["levels"][item_index]
                    - targets[c["id"]]["criterion_levels"][item_index]
                )
                for c in paired_items
            ]
            weight = train[0]["label"]["criterion_weights"][item_index]
            item_metrics.append(
                dict(
                    criterion_id=criterion,
                    weight=weight,
                    paired_samples=len(paired_items),
                    ai_MAE_grade=(
                        float(np.mean(ai_errors)) if ai_errors else None
                    ),
                    rule_MAE_grade=(
                        float(np.mean(rule_errors)) if rule_errors else None
                    ),
                    ai_MAE_weighted_points=(
                        float(np.mean(ai_errors) * weight / 2)
                        if ai_errors
                        else None
                    ),
                    rule_MAE_weighted_points=(
                        float(np.mean(rule_errors) * weight / 2)
                        if rule_errors
                        else None
                    ),
                    scope="observable criterion only; not a renormalized total",
                )
            )
        result["item_metrics"] = item_metrics
        both = [r for r in comparisons if r["rule_score"] is not None]
        baseline = float(np.mean([r["label"]["score"] for r in train]))
        result["metrics"] = dict(
            paired_samples=len(both),
            ai_MAE_points=(
                float(
                    np.mean(
                        [abs(r["ai_score"] - r["label_score"]) for r in both]
                    )
                )
                if both
                else None
            ),
            rule_MAE_points=(
                float(
                    np.mean(
                        [abs(r["rule_score"] - r["label_score"]) for r in both]
                    )
                )
                if both
                else None
            ),
            train_mean_baseline_MAE_points=(
                float(np.mean([abs(baseline - r["label_score"]) for r in both]))
                if both
                else None
            ),
            ai_median_ms=(
                float(np.median([r["ai_ms"] for r in comparisons]))
                if comparisons
                else None
            ),
            rule_median_ms=(
                float(np.median([r["rule_ms"] for r in comparisons]))
                if comparisons
                else None
            ),
            ai_assessed=len(comparisons),
            rule_assessed=len(both),
            eligible_validation_labels=sum(
                r["exercise"] == exercise
                and r["eligible_label"]
                and r["original_split"] == "validation"
                for r in manifests
            ),
        )
        result["failure_examples"] = sorted(
            [
                dict(
                    id=r["id"],
                    ai_absolute_error=abs(r["ai_score"] - r["label_score"]),
                    rule_absolute_error=(
                        abs(r["rule_score"] - r["label_score"])
                        if r["rule_score"] is not None
                        else None
                    ),
                )
                for r in comparisons
            ],
            key=lambda r: r["ai_absolute_error"],
            reverse=True,
        )[:3]
        LOGGER.info("%s: %s", exercise, result["metrics"])
    audit.save_json(audit.OUT / "comparison.json", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), ensure_ascii=False, indent=2))
