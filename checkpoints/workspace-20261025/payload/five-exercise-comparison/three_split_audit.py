"""3종 전체의 참여자·원본 분할과 제공자 라벨 중복을 점수 사용 없이 감사한다."""

import collections

import audit
import three_protocol


def run() -> dict:
    """점수 목표를 참조하지 않고 학습·검증·시험의 식별자 겹침을 저장한다.

    Returns:
        dict: 참여자·원본 충돌, 수량, 과거 시험 노출과 출처 식별의 한계.
        외부 데이터셋 간 실제 인물 대응이 확인됐다는 뜻은 아니다.

    Raises:
        OSError: 라벨·manifest 접근 또는 결과 저장 실패.
        ValueError: 한 참여자 또는 동일 원본이 여러 분할에 섞인 경우.
    """
    root = three_protocol.OUT
    rows = audit.read_json(root / "score_annotations.json")
    additional = root / "additional_score_annotations.json"
    if additional.exists():
        rows += audit.read_json(additional)
    people, originals = collections.defaultdict(set), collections.defaultdict(
        set
    )
    old_fronts = {
        r["id"]: r.get("front_sha256")
        for r in audit.read_json(audit.OUT / "development_manifest.json")
    }
    for row in rows:
        people[row["person"]].add(row["split"])
        for digest in [
            row["video_sha256"],
            row.get("source_original_sha256"),
            row.get("front_sha256") or old_fronts.get(row["id"]),
        ]:
            if digest:
                originals[digest].add(row["split"])
    conflicts = dict(
        people={k: sorted(v) for k, v in people.items() if len(v) > 1},
        originals={k: sorted(v) for k, v in originals.items() if len(v) > 1},
    )
    report = dict(
        records=len(rows),
        person_groups=len(people),
        original_hash_groups=len(originals),
        conflicts=conflicts,
        passed=not any(conflicts.values()),
        metadata_only_no_score_targets_used=True,
        old_test_records_not_new_independent_exam=True,
        cross_dataset_actual_person_identity_verified=False,
        REHAB24_REHAB26_candidate_fit_policy="separate_cohorts_until_provider_identity_mapping_verified",
    )
    audit.save_json(root / "split_audit.json", report)
    if not report["passed"]:
        raise ValueError("Person/source split conflicts require quarantine")
    return report


if __name__ == "__main__":
    print(run()["passed"])
