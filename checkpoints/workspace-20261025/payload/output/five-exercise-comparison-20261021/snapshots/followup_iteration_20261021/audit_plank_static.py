"""정지 항목 개발 실험의 누락 편향·재실행 일치·분할을 감사한다."""

import collections

import numpy as np

import audit


def run() -> dict:
    """동결 라벨·교차 검증·총점 보류와 별도 추출 캐시를 검사한다.

    Returns:
        dict: 등급·참여자별 평가 가능률, 재실행 좌표 차이(픽셀),
        직접 확인한 실패 7장 중 주요 원인과 자동 감사 결과.

    Raises:
        OSError: 필수 파일 접근 실패.
        ValueError: 라벨 변경·분할 누출·총점 생성·원본 불일치.
    """
    folder = audit.OUT / "plank_static_review"
    report = audit.read_json(folder / "comparison.json")
    labels = audit.read_json(folder / "direct_labels.json")
    if (
        audit.digest(folder / "direct_labels.json")
        != report["protocol"]["label_sha256"]
    ):
        raise ValueError("Frozen direct labels changed")
    if any(
        row["total_score"] is not None
        for row in report["rows"] + labels["records"]
    ):
        raise ValueError("Static image unexpectedly has full score")
    a = np.load(folder / "heavy_static.npz", allow_pickle=False)
    b = np.load(folder / "heavy_static_repeat.npz", allow_pickle=False)
    for name in ["ids", "source_sha256", "model_sha256", "schema"]:
        if not np.array_equal(a[name], b[name]):
            raise ValueError("Repeat extraction input/model mismatch")
    for fold in report["folds"]:
        if fold["held_person"] in fold["train_people"]:
            raise ValueError("Person token leaks across fold")
        tested = [
            r
            for r in report["rows"]
            if r["person_token"] == fold["held_person"]
            and r["ai_oof_grade"] is not None
        ]
        if len(tested) != fold["test_images"]:
            raise ValueError("Held-out count mismatch")
    by_grade = {}
    for grade in [0, 1, 2, None]:
        rows = [r for r in report["rows"] if r["draft_grade"] == grade]
        observed = [r for r in rows if r["pose"] is not None]
        by_grade[str(grade)] = dict(
            images=len(rows),
            observable=len(observed),
            coverage=len(observed) / len(rows) if rows else None,
        )
        if grade is not None:
            for key in ["ai_oof_grade", "rule_grade", "train_mean_grade"]:
                values = [
                    abs(r[key] - grade)
                    for r in rows
                    if r["ai_oof_grade"] is not None
                ]
                by_grade[str(grade)][key + "_mae"] = (
                    float(np.mean(values)) if values else None
                )
    by_person = {}
    for person in sorted({r["person_token"] for r in report["rows"]}):
        rows = [r for r in report["rows"] if r["person_token"] == person]
        by_person[person] = dict(
            images=len(rows),
            observable=sum(r["pose"] is not None for r in rows),
            common_evaluated=sum(r["ai_oof_grade"] is not None for r in rows),
        )
    difference = np.abs(a["image_xy"] - b["image_xy"])
    result = dict(
        label_sha256=audit.digest(folder / "direct_labels.json"),
        automated_audit_not_direct_review=True,
        full_totals_created=0,
        duplicate_image_hashes=[
            k
            for k, v in collections.Counter(
                r["image_sha256"] for r in report["rows"]
            ).items()
            if v > 1
        ],
        person_overlap_across_folds=False,
        coverage_by_draft_grade=by_grade,
        equal_grade_weight_mae={
            key: float(
                np.mean([by_grade[str(g)][key + "_mae"] for g in [0, 1, 2]])
            )
            for key in ["ai_oof_grade", "rule_grade", "train_mean_grade"]
        },
        coverage_by_person=by_person,
        repeat_extraction=dict(
            independent_process=True,
            images=len(a["ids"]),
            same_observation_mask=bool(
                np.array_equal(
                    np.isfinite(a["image_xy"]), np.isfinite(b["image_xy"])
                )
            ),
            maximum_observed_coordinate_delta_px=float(np.nanmax(difference)),
            exact_finite_coordinates_equal=bool(
                np.array_equal(a["image_xy"], b["image_xy"], equal_nan=True)
            ),
            not_temporal_or_recapture_consistency=True,
        ),
        direct_overlay_review=dict(
            images=7,
            ids=[
                "555-1-2-26-Z56",
                "553-1-2-26-Z59",
                "556-1-2-26-Z59",
                "553-1-2-26-Z115",
                "554-1-2-26-Z59",
                "558-1-2-26-Z59",
                "554-1-2-26-Z61",
            ],
            findings_ko=[
                "Z56의 큰 골반 상승 장면은 다리 관절 미관측이며 기구 기둥·밴드가 겹침",
                "554-Z59는 실제 다리 대신 굽힌 다리처럼 관절이 추정되어 골반 내각 126.7도로 계산됨",
                "558-Z59의 직접 몸통 이탈 초안과 175도 화면 대리값이 불일치; 사선 투영과 옷·요추 굴곡은 단일 골반 내각으로 충분히 설명되지 않음",
                "554-Z61은 높은 정렬 등급 초안인데 다른 참여자만 학습한 AI가 약 0.54등급으로 예측함",
            ],
            labels_changed_after_result=False,
        ),
        conclusions_ko=[
            "낮은 정렬 등급의 관측 손실이 커 전체 실패를 제외한 오차만으로 성능을 평가하면 낙관적일 수 있음",
            "Z56 참여자 8장은 전부 관절 관측 실패하여 참여자 분리 비교에 들어가지 못함",
            "AI가 학습 평균 기준선을 넘지 못하므로 현재 모델을 공통 총점 모델로 채택하지 않음",
            "같은 이미지 재실행은 일치하나 가림·구도·참여자 일반화와 동작 시간 검증은 해결되지 않음",
            "GPU 최초 실행 로그의 Tensor 경고를 숨기지 않음; 별도 재실행은 정상 종료하고 좌표가 일치함",
        ],
    )
    audit.save_json(folder / "quality_audit.json", result)
    return result


if __name__ == "__main__":
    result = run()
    print(
        {
            k: result[k]
            for k in [
                "coverage_by_draft_grade",
                "repeat_extraction",
                "full_totals_created",
            ]
        }
    )
