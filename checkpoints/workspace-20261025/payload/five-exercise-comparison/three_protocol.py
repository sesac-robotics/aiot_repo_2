"""3종 집중 검증의 경로·사전 기준과 QEVD 호환 JSON 어댑터를 제공한다.

원본 QEVD 목록의 네 필드를 유지하고 점수·출처는 별도 확장 키에 둔다.
다른 데이터셋의 정상·오류 문자열은 보존할 뿐 점수로 변환하지 않는다.
팀 공통 API·DB 계약과 기존 5종 결과는 변경하지 않는다.
"""

import copy
import datetime
import pathlib

import audit

OUT = audit.BASE / "output/three-exercise-comparison-20261022"
EXERCISES = ("pushups", "squats", "shoulder")
SCHEMA = "qevd-compatible-assessment-v1"


def validate_record(record: dict) -> None:
    """QEVD 기본 구조와 독립 점수 확장의 범위·결측 계약을 검사한다.

    Args:
        record: 네 기본 키와 assessment·provenance 등이 있는 한 레코드.

    Returns:
        None: 구조와 배점·등급·총점이 일치하면 정상 반환한다.

    Raises:
        ValueError: 필드 형식·분할·등급·총점 또는 원본 식별자가 잘못된 경우.
    """
    if (
        not isinstance(record.get("video_path"), str)
        or not record["video_path"]
    ):
        raise ValueError("QEVD video_path must be a nonempty string")
    for key in ("labels", "labels_descriptive"):
        if not isinstance(record.get(key), list) or not all(
            isinstance(v, str) for v in record[key]
        ):
            raise ValueError("QEVD labels must remain string lists")
    if record.get("split") not in ("train", "validation", "test"):
        raise ValueError("Invalid split")
    if record.get("exercise") not in EXERCISES:
        raise ValueError("Only the three active exercises are permitted")
    if not record.get("person") or not record.get("video_sha256"):
        raise ValueError("Person and original video identity required")
    assessment = record["assessment"]
    levels = assessment["criterion_levels"]
    weights = assessment["criterion_weights"]
    if len(levels) != 4 or len(weights) != 4 or sum(weights) != 100:
        raise ValueError("Four items and 100 point weights required")
    if any(
        type(g) is not int or g not in (0, 1, 2)
        for g in levels
        if g is not None
    ):
        raise ValueError("Grades must be 0/1/2 or null")
    expected = (
        None
        if None in levels
        else sum(w * g / 2 for w, g in zip(weights, levels, strict=True))
    )
    if assessment["score"] != expected:
        raise ValueError("Missing items cannot be filled or renormalized")
    if assessment.get("expert_validated") is not False:
        raise ValueError("These drafts are not expert reference labels")


def convert(row: dict, *, dataset: str) -> dict:
    """기존 직접 검토 초안을 QEVD 기본 필드를 보존한 확장 레코드로 바꾼다.

    Args:
        row: 기존 라벨 사전. 등급·배점·원본·참여자·구간을 포함한다.
        dataset: 출처 데이터셋 식별자. 예: QEVD, REHAB26, MultiView.

    Returns:
        dict: 원본 문자열과 독립 점수 초안을 구분한 새 사전.
        입력 사전은 변경하지 않으며 변환만으로 새 직접 검토를 주장하지 않는다.

    Raises:
        KeyError: 필수 원본·평가 필드 누락.
        ValueError: 변환 후 계약 불일치.
    """
    result = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key
        not in (
            "criterion_ids",
            "criterion_weights",
            "criterion_levels",
            "score",
        )
    }
    result.update(
        schema_version=SCHEMA,
        labels=copy.deepcopy(row.get("labels", [])),
        labels_descriptive=copy.deepcopy(row.get("labels_descriptive", [])),
        assessment={
            "criterion_ids": row["criterion_ids"],
            "criterion_weights": row["criterion_weights"],
            "criterion_levels": row["criterion_levels"],
            "score": row["score"],
            "rubric_version": row["rubric_version"],
            "unit": "observable_project_core_points_0_100",
            "label_source": row.get("label_source"),
            "review_method": row.get("review_method"),
            "expert_validated": False,
        },
        provenance={
            "dataset": dataset,
            "original_schema_version": row.get("schema_version"),
            "provider_labels_are_not_score_targets": True,
        },
    )
    if dataset == "QEVD":
        result["worker_id"] = int(row["person"].removeprefix("qevd_"))
    else:
        # QEVD 원래 숫자 식별자와 외부 데이터 참여자 이름을 충돌시키지 않는다.
        result["worker_id"] = None
    validate_record(result)
    return result


def initialize() -> dict:
    """후속 점수를 보기 전에 3종의 실험용 합격 기준과 보존 해시를 저장한다.

    Returns:
        dict: 교육용 개발 완료의 수량·오차·관측·일관성·속도 기준.
        전문가 검증·실사용 인증과 다르며 기준 미달은 미달로 기록한다.

    Raises:
        OSError: 원본·결과 접근 또는 쓰기 실패.
        ValueError: 기존 동결 파일이 현재 보존 대상과 다를 경우.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        return audit.read_json(target)
    preserved = [
        audit.OUT / "labels.json",
        audit.OUT / "supplemental_labels.json",
        audit.OUT / "plank_experimental.joblib",
        audit.OUT / "ytw_experimental.joblib",
        audit.OUT / "followup_W_model/model.joblib",
        audit.OUT / "followup_plank_model/model.joblib",
        audit.OUT / "followup_results.html",
        audit.OUT / "followup_iteration_verification.json",
        pathlib.Path(__file__).parent / "rule_score.py",
    ]
    report = dict(
        created_at=datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9))
        ).isoformat(),
        scope=list(EXERCISES),
        json_reference_files=[
            str(
                audit.BASE
                / "data/qevd-selected/raw/coach/fine_grained_labels.json"
            ),
            str(
                audit.BASE
                / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
            ),
        ],
        base_fields=["video_path", "labels", "labels_descriptive", "split"],
        assessment_is_separate_extension=True,
        directory_reorganization_requested=False,
        preserved_sha256={str(p): audit.digest(p) for p in preserved},
        engineering_acceptance={
            "min_training_people_per_exercise": 8,
            "min_training_segments_per_exercise": 20,
            "min_development_validation_people": 4,
            "min_development_validation_segments": 10,
            "min_prospective_final_people": 4,
            "min_prospective_final_segments": 10,
            "min_final_error_people": 3,
            "paired_MAE_points_max": 10,
            "paired_P90_absolute_error_points_max": 20,
            "each_assessable_fraction_min": 0.8,
            "normal_and_error_subgroup_MAE_points_max": 15,
            "AI_MAE_better_than_train_mean_baseline": True,
            "repeated_same_input_max_delta_points": 1e-6,
            "guarded_end_to_end_seconds_per_input_second_max": 2,
        },
        acceptance_status="predeclared_project_engineering_targets_not_team_or_expert_approved_accuracy_standard",
        grade_weights_and_rubric_preserved=True,
        no_rule_or_provider_string_targets=True,
        no_unknown_imputation_or_total_renormalization=True,
        calibration_only_training_or_development_groups=True,
        final_exam_freeze_required=True,
        exposed_old_test_labels_not_new_independent_exam=True,
        production_or_expert_certification=False,
    )
    audit.save_json(target, report)
    return report


if __name__ == "__main__":
    print(initialize()["scope"])
