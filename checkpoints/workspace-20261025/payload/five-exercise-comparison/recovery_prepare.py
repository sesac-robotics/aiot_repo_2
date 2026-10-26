"""기존 QEVD 부분 초안을 실제 구도별 관절 입력으로 복구한다.

새 외부 자료를 검색하지 않는다. 기존 원본·라벨·모델을 수정하지 않고
별도 후보 폴더에 원본 연결과 캐시만 저장한다.
"""

import json
import pathlib

import audit
import extract
from resources import check_resources
from recovery_variant import standing_allowed

ROOT = audit.BASE / "output/five-exercise-recovery-20261022"


def run(extract_landmarks: bool = True) -> dict:
    """기존 QEVD 개발 부분 라벨과 최근 초안을 정면 입력으로 준비한다.

    Args:
        extract_landmarks: True이면 관절을 추출한다. False이면 같은 원본·구간의
            관측 메타데이터만 갱신하며 기존 캐시는 입력 해시 검사 후 재사용한다.

    Returns:
        dict: 원본·참여자·분할·라벨·관절 캐시 연결 목록. 추출은 직접 검토가
        아니며 제공자 문자열은 라벨·특징으로 변환하지 않는다.

    Raises:
        OSError: 원본·라벨·출력 접근 실패.
        ValueError: 분할·해시·프레임 계약 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    ROOT.mkdir(exist_ok=True)
    old = audit.read_json(audit.OUT / "labels.json")
    selected = [
        dict(x, recovery_origin="prior_direct_labels")
        for x in old
        if x["id"].startswith("qevd_")
        and x["exercise"] in ("pushups", "squats", "ytw")
        and x["split"] != "test"
        and any(g is not None for g in x["criterion_levels"])
    ]
    for row in audit.read_json(
        audit.BASE
        / "output/existing-data-audit-20261022/score_annotations.json"
    ):
        if row["split"] == "test":
            continue
        selected.append(
            dict(
                row,
                **row["assessment"],
                recovery_origin="recent_existing_review"
            )
        )
    if (ROOT / "additional_score_annotations.json").exists():
        for row in audit.read_json(ROOT / "additional_score_annotations.json"):
            selected.append(
                dict(row, **row["assessment"], recovery_origin="seventy_batch1")
            )
    ids = set()
    for path in sorted(ROOT.glob("full_repetition_score_annotations_*.json")):
        for row in audit.read_json(path):
            selected.append(
                dict(
                    row,
                    **row["assessment"],
                    recovery_origin="full_repetition_review"
                )
            )
    variant_path = ROOT / "variant_policy_sources.json"
    excluded_sources = (
        {
            r["video_sha256"]
            for r in audit.read_json(variant_path)["records"]
            if not r["allowed_for_standing_target"]
        }
        if variant_path.exists()
        else set()
    )
    excluded = [
        dict(
            id=r["id"],
            source_sha256=r["video_sha256"],
            reason="seated_source_excluded_by_existing_standing_contract",
        )
        for r in selected
        if not standing_allowed(r, excluded_sources)
    ]
    selected = [r for r in selected if standing_allowed(r, excluded_sources)]
    audit.save_json(
        ROOT / "variant_policy_exclusions.json", dict(records=excluded)
    )
    people = {}
    sources = {}
    for row in selected:
        if row["id"] in ids:
            raise ValueError("Duplicate annotation ID")
        ids.add(row["id"])
        for key, registry in [
            (row["person"], people),
            (row["video_sha256"], sources),
        ]:
            if key in registry and registry[key] != row["split"]:
                raise ValueError("Split leakage")
            registry[key] = row["split"]
        row["required_views"] = row.get("required_input_views") or ["front"]
        row["recovery_view"] = (
            "side"
            if "side" in row["required_views"]
            else (
                "front_oblique"
                if "front_oblique" in row["required_views"]
                else "front"
            )
        )
    audit.save_json(
        ROOT / "input_plan.json",
        dict(
            records=selected,
            source_labels_used_as_features=False,
            existing_direct_drafts_reused_not_new_reviews=True,
        ),
    )
    if not extract_landmarks:
        return dict(planned=len(selected), landmarks_extracted_this_call=False)
    report = dict(records=[], completed=False)
    for row in selected:
        check_resources()
        receipt = extract.extract_view(
            row, "primary", row["video_path"], destination=ROOT / "heavy"
        )
        record = dict(id=row["id"], extraction=receipt)
        if "synchronized_front" in row["required_views"]:
            if (
                audit.digest(pathlib.Path(row["front_path"]))
                != row["front_sha256"]
            ):
                raise ValueError("Synchronized front source changed")
            record["front_extraction"] = extract.extract_view(
                row, "front", row["front_path"], destination=ROOT / "heavy"
            )
        report["records"].append(record)
        audit.save_json(ROOT / "extraction.json", report)
    report["completed"] = True
    audit.save_json(ROOT / "extraction.json", report)
    return dict(extracted=len(report["records"]), completed=True)


if __name__ == "__main__":
    import sys

    print(json.dumps(run("plan" not in sys.argv)))
