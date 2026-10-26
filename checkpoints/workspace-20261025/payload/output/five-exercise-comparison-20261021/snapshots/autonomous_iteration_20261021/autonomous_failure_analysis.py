"""동결 시험의 오류와 총점·항목 규칙 간 차이를 사후 진단한다.

사후 진단으로 동결 점수·라벨·경계를 바꾸지 않는다. 총점 규칙의
공통 측면 선택과 항목별 관측 측면 선택을 서로 다른 버전으로 표시한다.
"""

import pathlib

import numpy as np

import audit
import rule_score


def run() -> dict:
    """기존 캐시에서 총점 규칙 근거를 재구성하고 불변 결과를 대조한다.

    Returns:
        dict: 재학습 없이 계산한 규칙 근거·항목판 차이·오탐 목록.
        사후 분석이며 새 독립 시험이나 개선 점수가 아니다.

    Raises:
        OSError: 동결 정책·결과·캐시 접근 실패.
        ValueError: 규칙 해시나 동결 점수와의 불일치.
    """
    freeze = audit.read_json(audit.OUT / "autonomous_prospective_freeze.json")
    if (
        audit.digest(pathlib.Path(rule_score.__file__))
        != freeze["files"]["common_rule"]["sha256"]
    ):
        raise ValueError("Frozen rule differs")
    source = audit.OUT / "autonomous_prospective_results.json"
    before = audit.digest(source)
    report = dict(
        prospective_result_sha256=before,
        posthoc_only=True,
        threshold_or_model_modified=False,
        records=[],
    )
    for row in audit.read_json(source)["records"]:
        if not row["full_total_approved"]:
            continue
        with np.load(row["extraction"]["cache"]) as cache:
            seq = cache["image_xy"]
        items = [r["criterion_id"] for r in row["items"]]
        rule = rule_score.evaluate("plank", seq, None, items, [40, 25, 20, 15])
        if rule["score"] != row["full_rule_score"]:
            raise ValueError("Frozen rule score differs")
        differences = [
            dict(
                criterion_id=item["criterion_id"],
                frozen_total_rule_grade=level,
                separate_item_rule_grade=item["rule_grade"],
                separate_item_side=item["observation"]["side"],
            )
            for item, level in zip(row["items"], rule["levels"], strict=True)
            if level != item["rule_grade"]
        ]
        report["records"].append(
            dict(
                id=row["id"],
                frozen_rule=rule,
                item_version_differences=differences,
                note_ko=(
                    "초기 골반·넓적다리 바닥 접촉 초안과 달리 관절 각도 대리값은 "
                    "부분 등급을 부여했다. 관절 기하만으로 실제 접촉을 확정할 수 없다."
                    if row["id"] == "auto_italo_initial"
                    else "동결 총점판은 공통 오른쪽 몸통 선 161.81도로 1등급을 "
                    "부여했다. 별도 항목판은 관측률로 왼쪽을 선택해 2등급이었다. "
                    "이 차이를 숨기거나 시험 결과를 보고 측면을 바꾸지 않는다."
                ),
            )
        )
    events = audit.read_json(audit.OUT / "upper_event_nested_results.json")
    report["upper_event_false_positives"] = [
        r for r in events["predictions"] if r["target"] == 0 and r["event"]
    ]
    report["event_limit_ko"] = (
        "머리·몸통 회전 및 배경 대조 구간에서 오탐이 남았다. 입력에서 얼굴을 "
        "제외했어도 어깨·팔꿈치 투영 변화와 관절 추정 변동은 남는다. 원인 "
        "분리는 추가 촬영 대조 실험이 필요하며 현재 단일 원인으로 확정하지 않는다."
    )
    if audit.digest(source) != before:
        raise ValueError("Prospective result changed during diagnosis")
    audit.save_json(audit.OUT / "autonomous_failure_analysis.json", report)
    return report


if __name__ == "__main__":
    print(run()["records"])
