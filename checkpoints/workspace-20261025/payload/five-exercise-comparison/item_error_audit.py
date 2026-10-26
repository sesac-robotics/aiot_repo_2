"""총점에서 상쇄되는 항목별 오차와 미관측 항목을 분리해 기록한다.

직접 초안 대비 개발 진단이다. 학습·검증·규칙 미리보기 역할을 합치지
않고 미관측 항목을 0점·만점 또는 재정규화 총점으로 만들지 않는다.
"""

import math
import audit


def compare_items(target: list, predicted: list, weights: list) -> dict:
    """항목 등급 차이의 가중 절댓값과 총점 오차 상쇄량을 반환한다.

    Args:
        target: 직접 초안 항목 등급 목록. 0~2 또는 None이다.
        predicted: 모델 또는 규칙 항목 등급 목록. 0~2 또는 None이다.
        weights: 같은 순서의 양수 배점 목록. 단위는 점이다.

    Returns:
        dict: 항목별 예측-초안 배점 차이, 비교 가능 배점, 가중 절대
        오차 합(비교 가능한 항목만), 총점 오차·상쇄량. 하나라도 미관측이면
        총점 오차·상쇄량은 None이며 총점을 추정하지 않는다.

    Raises:
        ValueError: 길이 불일치·빈 목록·유효하지 않은 등급·배점.
    """
    if not target or not len(target) == len(predicted) == len(weights):
        raise ValueError("Item arrays must be nonempty and aligned")
    differences = []
    for expected, actual, weight in zip(
        target, predicted, weights, strict=True
    ):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Weights must be finite positive points")
        for value in (expected, actual):
            if value is not None and (
                not math.isfinite(value) or not 0 <= value <= 2
            ):
                raise ValueError("Grades must be finite within 0..2 or None")
        differences.append(
            None
            if expected is None or actual is None
            else weight * (actual - expected) / 2
        )
    paired = [v for v in differences if v is not None]
    complete = len(paired) == len(weights)
    absolute = sum(abs(v) for v in paired)
    total = abs(sum(paired)) if complete else None
    return dict(
        signed_item_error_points=differences,
        paired_items=len(paired),
        total_items=len(weights),
        paired_weight_points=sum(
            w
            for w, d in zip(weights, differences, strict=True)
            if d is not None
        ),
        paired_item_absolute_error_sum_points=absolute,
        absolute_total_error_points=total,
        cancellation_points=absolute - total if complete else None,
        expert_ground_truth=False,
    )


def run() -> dict:
    """현재 비교 결과의 AI·규칙 항목 오차를 역할별 레코드로 저장한다.

    Returns:
        dict: 항목 오차·가림을 포함한 개발 진단. 검증 비교와 학습 자료의
        규칙 미리보기는 각 evaluation_role로 구분하며 최종 시험은 아니다.

    Raises:
        ValueError: 등급·배점 배열 계약 불일치.
        KeyError: 필수 라벨 또는 결과 항목 누락.
        OSError: 입력 또는 결과 파일 접근 실패.
    """
    labels = {r["id"]: r for r in audit.read_json(audit.OUT / "labels.json")}
    results = []
    for exercise, group in audit.read_json(audit.OUT / "comparison.json")[
        "results"
    ].items():
        for role, rows in (
            ("exposed_development_validation", group.get("comparisons", [])),
            (
                "development_train_rule_preview_not_test",
                group.get("rule_only_previews", []),
            ),
        ):
            for row in rows:
                label = labels[row["id"]]
                record = dict(
                    id=row["id"],
                    exercise=exercise,
                    evaluation_role=role,
                    criterion_ids=label["criterion_ids"],
                    criterion_weights=label["criterion_weights"],
                    draft_levels=label["criterion_levels"],
                    expert_validated=False,
                    independent_final_test=False,
                )
                for name, levels in (
                    ("AI", row.get("ai_levels")),
                    ("Rule", row["rule_details"]["levels"]),
                ):
                    if levels is not None:
                        record[name] = dict(
                            predicted_levels=levels,
                            **compare_items(
                                label["criterion_levels"],
                                levels,
                                label["criterion_weights"],
                            )
                        )
                results.append(record)
    report = dict(
        records=results,
        unit="project_weighted_points",
        targets_are_experimental_drafts=True,
        final_test_count=0,
        training_previews_aggregated_as_validation=False,
        labels_modified=False,
    )
    audit.save_json(audit.OUT / "item_error_audit.json", report)
    return report


if __name__ == "__main__":
    print(len(run()["records"]))
