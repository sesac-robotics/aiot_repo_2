"""기존 QEVD 부분 초안을 실제 구도별 관절 입력으로 복구한다.

새 외부 자료를 검색하지 않는다. 기존 원본·라벨·모델을 수정하지 않고
별도 후보 폴더에 원본 연결과 캐시만 저장한다.
"""

import json

import audit
import extract
from resources import check_resources

ROOT = audit.BASE / "output/five-exercise-recovery-20261022"


def run() -> dict:
    """기존 QEVD 개발 부분 라벨과 최근 초안을 정면 입력으로 준비한다.

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
        dict(x)
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
        selected.append(dict(row, **row["assessment"]))
    ids = set()
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
            "front_oblique" if row["exercise"] == "shoulder" else "front"
        )
    audit.save_json(
        ROOT / "input_plan.json",
        dict(
            records=selected,
            source_labels_used_as_features=False,
            existing_direct_drafts_reused_not_new_reviews=True,
        ),
    )
    report = dict(records=[], completed=False)
    for row in selected:
        check_resources()
        receipt = extract.extract_view(
            row, "primary", row["video_path"], destination=ROOT / "heavy"
        )
        report["records"].append(dict(id=row["id"], extraction=receipt))
        audit.save_json(ROOT / "extraction.json", report)
    report["completed"] = True
    audit.save_json(ROOT / "extraction.json", report)
    return dict(extracted=len(report["records"]), completed=True)


if __name__ == "__main__":
    print(json.dumps(run()))
