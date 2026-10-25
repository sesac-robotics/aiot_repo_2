"""5종 비교 실험의 라벨·관측 계약·분할을 감사한다.

기존 라벨을 변경하지 않고 로컬 실험 폴더에 JSON을 저장한다.
영상 내용의 직접 검토나 전문가 검증을 대신하지 않는다.
"""

import collections
import hashlib
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
SOURCE = BASE / "output/label-review-20261021/all-five-v2"
OUT = BASE / "output/five-exercise-comparison-20261021"
CACHE_SCHEMA = "heavy-observed-sequence-v2-sequential"
VIEWS = {
    "pushups": ["side"],
    "squats": ["side", "synchronized_front"],
    "plank": ["side"],
    "ytw": ["front_or_oblique_full_arms_and_hinged_trunk"],
    "shoulder": ["front"],
}
UNITS = {
    "pushups": "상단 → 하강 → 상단의 한 반복",
    "squats": "동기화 측면·정면의 서기 → 하강 → 서기 한 반복",
    "plank": "편집 없는 측면 전신의 약 3초 지지",
    "ytw": "Y → T → W → 원위치의 한 시도",
    "shoulder": "어깨 높이 하단 → 머리 위 상단 → 하단 한 반복",
}


def read_json(path: pathlib.Path) -> object:
    """UTF-8 JSON 파일을 읽어 Python 값으로 반환한다.

    Args:
        path: 읽을 JSON 파일 경로. 물리 단위 없음.

    Returns:
        object: JSON의 목록·사전·문자열·수치 등 원래 구조의 값.

    Raises:
        OSError: 파일을 읽을 수 없는 경우.
        json.JSONDecodeError: 파일 내용이 유효한 JSON이 아닌 경우.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: pathlib.Path, value: object) -> None:
    """유한한 JSON 값을 UTF-8 임시 파일로 쓴 뒤 원자적으로 교체한다.

    Args:
        path: 저장할 JSON 경로. 상위 폴더가 없으면 생성한다.
        value: JSON으로 직렬화 가능한 값. 물리 단위는 각 필드 계약을 따른다.

    Returns:
        None: 저장 파일을 교체하고 정상 반환한다.

    Raises:
        OSError: 폴더 생성·쓰기·교체 실패.
        ValueError: NaN·무한대 또는 순환 참조가 포함된 경우.
        TypeError: JSON으로 직렬화할 수 없는 값이 포함된 경우.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path: pathlib.Path) -> str:
    """파일 전체 바이트의 SHA-256을 계산한다.

    Args:
        path: 읽을 파일 경로. 물리 단위 없음.

    Returns:
        str: 64자리 영문 소문자 16진수 해시.

    Raises:
        OSError: 파일을 읽을 수 없는 경우.
    """
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def audit() -> dict:
    """최신 라벨 전체와 연결 원본을 검사하고 비교 준비 상태를 저장한다.

    Returns:
        dict: 검사 오류, 운동별 수량, 원본 연결 및 학습 보류 사유.

    Raises:
        OSError: 필수 자료 읽기 또는 결과 저장 실패.
        ValueError: 필수 JSON의 구조가 예상과 다른 경우.

    Notes:
        기존 test는 개발 중 이미 노출된 탐색 시험으로 유지한다.
        참여자 식별자는 제공자 메타데이터이며 사람 재식별 보증이 아니다.
    """
    loaded = {
        name: read_json(SOURCE / (name + ".json"))
        for name in (
            "labels",
            "summary",
            "rubrics",
            "verification",
            "qevd_inventory",
            "usage_and_input_contract",
        )
    }
    labels = list(loaded["labels"])
    supplemental_path = OUT / "supplemental_labels.json"
    if supplemental_path.exists():
        labels.extend(read_json(supplemental_path))
    checks_path = OUT / "visual_view_checks.json"
    checks = (
        {row["id"]: row for row in read_json(checks_path)}
        if checks_path.exists()
        else {}
    )
    rubrics = {
        (rubric["version"], exercise): definition
        for rubric in loaded["rubrics"]
        for exercise, definition in rubric["definitions"].items()
    }
    errors = []
    cache = {}
    groups = collections.defaultdict(set)
    manifests = []
    contracts = []
    for (version, exercise), rubric in rubrics.items():
        contracts.append(
            dict(
                exercise=exercise,
                rubric_version=version,
                criterion_ids=[item["id"] for item in rubric["items"]],
                criterion_names_ko=[
                    item["name_ko"] for item in rubric["items"]
                ],
                weights=rubric["weights"],
                anchors=rubric["items"],
                required_views=VIEWS[exercise],
                segment_unit_ko=UNITS[exercise],
                total_formula="sum(weight * level / 2)",
                unknown_policy_ko=(
                    "가림·잘림·관측 방향 불일치·수행자 불명·구간 불완전·"
                    "등급 경계 불확실은 null. 하나라도 null이면 총점 null."
                ),
                scope_ko=rubric.get("scope", rubric.get("exclusions")),
                expert_validated=False,
            )
        )
    seen = set()
    for row in labels:
        ident = row["id"]
        reasons = []
        if ident in seen:
            errors.append([ident, "duplicate_id"])
        seen.add(ident)
        rubric = rubrics.get((row["rubric_version"], row["exercise"]))
        if rubric is None:
            errors.append([ident, "unknown_rubric"])
            continue
        if row["criterion_ids"] != [i["id"] for i in rubric["items"]]:
            errors.append([ident, "criterion_ids_mismatch"])
        if row["criterion_weights"] != rubric["weights"]:
            errors.append([ident, "weights_mismatch"])
        levels = row["criterion_levels"]
        if len(levels) != len(rubric["weights"]) or any(
            level is not None
            and (type(level) is not int or level not in (0, 1, 2))
            for level in levels
        ):
            errors.append([ident, "invalid_levels"])
        expected = (
            None
            if None in levels
            else sum(
                weight * level / 2
                for weight, level in zip(rubric["weights"], levels)
            )
        )
        if expected != row["score"] and row["score"] is not None:
            errors.append([ident, "score_formula_mismatch"])
        segment = row["segment"]
        if not (
            segment["fps"] > 0
            and 0 <= segment["start_frame"] < segment["end_frame"]
        ):
            errors.append([ident, "invalid_segment"])
        paths = [row["video_path"]]
        if row.get("front_path"):
            paths.append(row["front_path"])
        if row.get("source_original_video_path"):
            paths.append(row["source_original_video_path"])
        for value in paths:
            if value not in cache:
                path = pathlib.Path(value)
                cache[value] = digest(path) if path.is_file() else None
            if cache[value] is None:
                errors.append([ident, "missing_video", value])
        if cache[row["video_path"]] != row["video_sha256"]:
            errors.append([ident, "video_hash_mismatch"])
        if row.get("source_original_video_path"):
            if (
                cache[row["source_original_video_path"]]
                != row["source_original_sha256"]
            ):
                errors.append([ident, "original_parent_hash_mismatch"])
            lineage = read_json(pathlib.Path(row["lineage_file"]))
            if (
                lineage["original_sha256"] != row["source_original_sha256"]
                or lineage["derived_sha256"] != row["video_sha256"]
                or lineage["spatial_crop_xyxy"] != row["spatial_crop_xyxy"]
            ):
                errors.append([ident, "derived_lineage_mismatch"])
        # 참고용 반복도 연결하되 reference_only는 분할 충돌에서 제외한다.
        if row["split"] != "reference_only":
            for field in ("person", "label_group", "performance_group"):
                if row.get(field):
                    groups["identity:" + row[field]].add(row["split"])
            for value in paths:
                if cache[value]:
                    groups["hash:" + cache[value]].add(row["split"])
        eligible = bool(row.get("eligible_for_total_score_training"))
        if not eligible:
            reasons.append("label_total_ineligible")
        if row["split"] == "reference_only":
            reasons.append("reference_only")
        if row["exercise"] == "squats" and not row.get("front_path"):
            reasons.append("synchronized_front_missing")
        if row.get("requires_subject_selection"):
            reasons.append("subject_roi_not_yet_verified_for_extractor")
        views = row.get("required_input_views")
        legacy_view = row.get("required_view")
        if (
            views is None
            and legacy_view
            and row["exercise"] in ("ytw", "shoulder")
        ):
            views = VIEWS[row["exercise"]]
        view_check = checks.get(ident)
        if view_check and view_check.get("confirmed_views"):
            views = view_check["confirmed_views"]
        if view_check and view_check.get("hold_reason"):
            reasons.append("direct_visual_view_contract_mismatch")
        elif views is None:
            reasons.append(
                "view_contract_inferred_requires_visual_confirmation"
            )
        if (
            row["exercise"] == "plank"
            and (segment["end_frame"] - segment["start_frame"]) / segment["fps"]
            < 2.9
        ):
            reasons.append("hold_shorter_than_approximately_3_seconds")
        manifests.append(
            dict(
                id=ident,
                exercise=row["exercise"],
                rubric_version=row["rubric_version"],
                person=row["person"],
                label_group=row.get("label_group"),
                original_split=row["split"],
                evaluation_role={
                    "train": "development_train",
                    "validation": "development_validation_exposed",
                    "test": "development_test_exposed",
                    "reference_only": "reference_only",
                }.get(row["split"], "unsupported_split"),
                independent_final_test=False,
                video_path=row["video_path"],
                video_sha256=row["video_sha256"],
                source_original_sha256=row.get("source_original_sha256"),
                front_path=row.get("front_path"),
                front_sha256=cache.get(row.get("front_path")),
                segment=segment,
                required_views=VIEWS[row["exercise"]],
                view_contract_origin=(
                    "direct_visual_recheck"
                    if view_check
                    else (
                        "legacy_label_required_view"
                        if legacy_view and views
                        else "label" if views else "rubric_inference"
                    )
                ),
                eligible_label=eligible,
                preparation_holds=reasons,
                extraction_ready=not reasons,
                label_origin="AI_direct_sampled_frame_review_provisional",
                expert_validated=False,
            )
        )
    leaks = {
        key: sorted(value) for key, value in groups.items() if len(value) > 1
    }
    counts = {}
    for exercise in VIEWS:
        rows = [r for r in manifests if r["exercise"] == exercise]
        counts[exercise] = dict(
            review_records=len(rows),
            total_label_candidates=sum(r["eligible_label"] for r in rows),
            extraction_ready=sum(r["extraction_ready"] for r in rows),
            eligible_by_original_split=dict(
                collections.Counter(
                    r["original_split"] for r in rows if r["eligible_label"]
                )
            ),
            holds=dict(
                collections.Counter(
                    reason for r in rows for reason in r["preparation_holds"]
                )
            ),
            independent_final_test_count=0,
        )
    report = dict(
        source=str(SOURCE),
        source_json_sha256={
            name: digest(SOURCE / (name + ".json")) for name in loaded
        },
        records=len(labels),
        source_video_paths_checked=len(cache),
        supplemental_records=len(labels) - len(loaded["labels"]),
        supplemental_sha256=(
            digest(supplemental_path) if supplemental_path.exists() else None
        ),
        structural_errors=errors,
        split_conflicts=leaks,
        exercises=counts,
        passed=not errors and not leaks,
        all_five_ready_for_training=False,
        expert_validated=False,
        source_licenses_reaudited=False,
        direct_visual_reviews_by_audit_script=0,
        separately_recorded_view_rechecks=len(checks),
        notes_ko=[
            "구조·해시 자동 감사이며 영상 직접 재검토가 아니다.",
            "검증 JSON의 기존 통과 표시는 현재 모델 정확도가 아니다.",
            "훈련·검증·시험 기존 배정은 변경하지 않았다.",
            "같은 해시·식별자의 충돌을 검사하나 미식별 중복은 남는다.",
            "출처 이용 조건 재확인과 동기화·수행자 확인이 별도로 필요하다.",
        ],
    )
    save_json(OUT / "audit.json", report)
    save_json(OUT / "labels.json", labels)
    save_json(OUT / "contracts.json", contracts)
    save_json(OUT / "development_manifest.json", manifests)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
