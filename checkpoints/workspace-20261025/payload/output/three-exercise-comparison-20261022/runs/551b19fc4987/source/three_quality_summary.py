"""전체 입력 판정률과 직접 총점이 관측된 표본의 판정률을 분리해 보고한다.

부분 라벨에서 모델이 총점을 낸 경우는 별도 점검 대상으로 남기며 이를
정답·오차 0으로 취급하지 않는다. 기존 비교 결과를 읽기만 한다.
"""

import statistics

import audit
import three_protocol


def summarize(rows: list[dict]) -> dict:
    """동일 평가 입력의 관측 범위별 분모와 항목 오차를 명시한다.

    Args:
        rows: 직접 초안·AI·규칙 예측·관측 근거를 가진 개발 비교 목록.

    Returns:
        dict: 전체·총점 초안 가능 입력 수, 방법별 판정률·항목 MAE와 점검 ID.
        초안 null이 관측 불가인지 등급 경계 불확실인지 자동 단정하지 않는다.
    """
    reference = [r for r in rows if r["target_score"] is not None]
    methods = {}
    for method in ["ai", "rule"]:
        assessed = [r for r in rows if r[method + "_score"] is not None]
        paired = [r for r in reference if r[method + "_score"] is not None]
        items = []
        for index in range(4):
            errors = [
                abs(r[method + "_levels"][index] - r["target_levels"][index])
                for r in rows
                if r[method + "_levels"][index] is not None
                and r["target_levels"][index] is not None
            ]
            items.append(
                dict(
                    item=(
                        rows[0]["observations"][index]["item"] if rows else None
                    ),
                    labeled_pairs=len(errors),
                    MAE_grade_units=statistics.mean(errors) if errors else None,
                )
            )
        methods[method] = dict(
            all_submitted_assessed=len(assessed),
            all_submitted_assessed_fraction=(
                len(assessed) / len(rows) if rows else None
            ),
            reference_observable_assessed=len(paired),
            reference_observable_assessed_fraction=(
                len(paired) / len(reference) if reference else None
            ),
            reference_unknown_but_total_predicted_ids=[
                r["id"] for r in assessed if r["target_score"] is None
            ],
            items=items,
        )
    return dict(
        submitted=len(rows),
        reference_total_available=len(reference),
        reference_total_unknown=len(rows) - len(reference),
        methods=methods,
    )


def run() -> dict:
    """운동·개발 검증 방식별 관측 범위와 오차의 한계를 저장한다.

    Returns:
        dict: 판정률·부분 항목 오차·단일 등급 학습 항목과 남은 검증 목록.

    Raises:
        OSError: 비교 결과 읽기 또는 요약 저장 실패.
        KeyError: 비교 레코드 계약 누락.
    """
    root = three_protocol.OUT
    report = audit.read_json(root / "development_results.json")
    output = dict(
        model_version=report["label_version"],
        exercises=[],
        expert_reference=False,
        independent_final_exam=False,
    )
    for item in report["exercises"]:
        output["exercises"].append(
            dict(
                exercise=item["exercise"],
                fixed_development_validation=summarize(
                    item["validation_predictions"]
                ),
                leave_person_out_development=summarize(
                    item["training_person_out_predictions"]
                ),
                single_grade_training_items=[
                    r["item"]
                    for r in item["item_training"]
                    if len(r["grade_counts"]) < 2
                ],
                fewer_than_two_error_people=[
                    r["item"]
                    for r in item["item_training"]
                    if r["error_people"] < 2
                ],
            )
        )
    audit.save_json(root / "quality_summary.json", output)
    return output


if __name__ == "__main__":
    print(len(run()["exercises"]))
