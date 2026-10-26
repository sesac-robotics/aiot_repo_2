"""새 오류 시범에서 항목 오차가 상쇄되어 총점만 좋아 보이는 경우를 드러낸다."""

import audit
import three_protocol


def run() -> dict:
    """새 원본 그룹을 통째로 제외한 개발 예측의 항목별 가중 오차를 저장한다.

    Returns:
        dict: 항목별 점 단위 부호 오차·절대합·총점 오차와 상쇄 여부.
        미판정 항목은 0점 오차로 채우지 않는다.

    Raises:
        OSError: 개발 예측·직접 주석 읽기 또는 결과 저장 실패.
        ValueError: 예측·직접 등급·배점 개수 불일치.
    """
    root = three_protocol.OUT
    report = audit.read_json(root / "development_results.json")
    labels = {
        r["id"]: r
        for r in audit.read_json(root / "additional_score_annotations.json")
    }
    records = []
    for exercise in report["exercises"]:
        for row in exercise["training_person_out_predictions"]:
            if not row["id"].startswith("youtube_DVGi7h0lL3A_"):
                continue
            reference = labels[row["id"]]["assessment"]
            for method in ("ai", "rule"):
                errors = []
                for item, weight, target, predicted in zip(
                    reference["criterion_ids"],
                    reference["criterion_weights"],
                    row["target_levels"],
                    row[method + "_levels"],
                    strict=True,
                ):
                    errors.append(
                        dict(
                            item=item,
                            target_level=target,
                            predicted_level=predicted,
                            weighted_error_points=(
                                None
                                if target is None or predicted is None
                                else weight * (predicted - target) / 2
                            ),
                        )
                    )
                values = [e["weighted_error_points"] for e in errors]
                known = all(v is not None for v in values)
                signed = sum(values) if known else None
                absolute = sum(abs(v) for v in values) if known else None
                records.append(
                    dict(
                        id=row["id"],
                        method=method,
                        evaluation="leave_person_out_development",
                        target_score=row["target_score"],
                        predicted_score=row[method + "_score"],
                        items=errors,
                        signed_total_error_points=signed,
                        sum_absolute_item_errors_points=absolute,
                        item_error_cancellation=bool(
                            known and absolute - abs(signed) > 1e-6
                        ),
                    )
                )
    result = dict(
        model_version=report["label_version"],
        records=records,
        expert_reference=False,
        independent_final_test=False,
        caution_ko="작은 총점 오차가 항목 오류 인식을 입증하지 않는다. 새 참여자의 지지 오류를 놓쳤는지 항목 등급을 함께 확인한다.",
    )
    audit.save_json(root / "new_error_case_audit.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
