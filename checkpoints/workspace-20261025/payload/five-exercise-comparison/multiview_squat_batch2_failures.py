"""보유 스쿼트 후보의 실패·반복 변동·오류 라벨 부족을 자동 집계한다."""

import collections
import json
import pathlib

import audit
from multiview_squat_batch2 import OUT


def run() -> dict:
    """저장 특징·예측만 대조해 후속 개선 항목을 기록한다.

    Returns:
        dict: 정상·오류별 총점 오차, 같은 초안 반복 간 점수 차이,
        뒤꿈치 특징의 중첩과 오류 없는 항목 목록. 자동 집계이며 직접 영상
        검토나 새 라벨·학습·검증 수치에 따른 경계 변경은 수행하지 않는다.

    Raises:
        OSError: 저장 결과를 읽거나 분석을 저장할 수 없는 경우.
        ValueError: 학습 입력 지문이 비교 결과와 다른 경우.
    """
    report = audit.read_json(OUT / "comparison.json")
    if report["fit_input_manifest_sha256"] != audit.digest(
        OUT / "fit_input_manifest.json"
    ):
        raise ValueError("Feature manifest changed")
    rows = audit.read_json(OUT / "fit_input_manifest.json")
    features = {r["id"]: r for r in rows}
    predictions = [p for p in report["predictions"] if p["subset"] == "batch2"]
    failures = []
    for p in predictions:
        if (
            p["target_score"] is None
            or p["ai_score"] is None
            or p["rule_score"] is None
        ):
            continue
        knee = next(
            n for n in features[p["id"]]["items"] if n["item"] == "knee_foot"
        )
        failures.append(
            dict(
                id=p["id"],
                person=p["person"],
                reference_score=p["target_score"],
                ai_error_points=p["ai_score"] - p["target_score"],
                rule_error_points=p["rule_score"] - p["target_score"],
                reference_knee_foot=p["levels"][2],
                ai_knee_foot=p["ai_levels"][2],
                rule_knee_foot=p["rule_levels"][2],
                relative_heel_change=knee["feature"][-1],
            )
        )
    repeated = []
    grouped = collections.defaultdict(list)
    for p in predictions:
        if (
            p["target_score"] is not None
            and p["ai_score"] is not None
            and p["rule_score"] is not None
        ):
            grouped[p["id"].rsplit("_rep", 1)[0]].append(p)
    for source, bucket in grouped.items():
        if len(bucket) < 2 or len({tuple(p["levels"]) for p in bucket}) != 1:
            continue
        repeated.append(
            dict(
                source_group=source,
                reference_levels=bucket[0]["levels"],
                segments=len(bucket),
                ai_range_points=max(p["ai_score"] for p in bucket)
                - min(p["ai_score"] for p in bucket),
                rule_range_points=max(p["rule_score"] for p in bucket)
                - min(p["rule_score"] for p in bucket),
                not_independent_people=True,
            )
        )
    distribution = []
    for split in ("train", "validation"):
        for item in (
            "descent_range",
            "return_extension",
            "knee_foot",
            "trunk_balance",
        ):
            levels = collections.Counter()
            groups = collections.defaultdict(set)
            for row in rows:
                if row["split"] != split:
                    continue
                for n, g in zip(row["items"], row["levels"]):
                    if n["item"] == item and n["eligible"] and g is not None:
                        levels[str(g)] += 1
                        groups[str(g)].add(row["person"])
            distribution.append(
                dict(
                    split=split,
                    item=item,
                    eligible_grade_counts=dict(levels),
                    grade_metadata_groups={
                        g: len(v) for g, v in groups.items()
                    },
                )
            )
    result = dict(
        comparison_sha256=audit.digest(OUT / "comparison.json"),
        automatic_analysis_not_direct_review=True,
        validation_total_coverage=dict(
            submitted=len(predictions),
            paired_total=len(failures),
            ratio=len(failures) / len(predictions),
        ),
        failures=failures,
        same_reference_repetition_variation=repeated,
        item_grade_distribution=distribution,
        observations_ko=[
            "검증 015의 뒤꿈치 들림 2구간을 AI가 95점 이상으로 예측한다. 안정적인 추적이 있어도 현재 특징과 학습 분포가 오류를 잘 구분하지 못한다.",
            "발 지지가 정상인 012 구간의 상대 뒤꿈치 변화량 약 0.046~0.051과 오류 015의 한 구간 약 0.048이 겹친다. 단일 상대 높이 경계로 완전히 구별할 수 없다.",
            "현재 상대 뒤꿈치 특징은 첫 세 관측 표본을 기준으로 계산하므로 시작 자세와 발 투영·관절 흔들림의 영향을 받는다. 이 설명은 저장 특징·구현에 근거한 원인 가설이며 인과 검증은 추가로 필요하다.",
            "복귀 신전과 정면 몸통 균형의 학습 가능 라벨은 정상 등급에만 몰려 있다. 해당 항목의 0 오차를 오류 검출 능력으로 설명할 수 없다.",
            "뒤꿈치 네 경계 보정은 학습 손실을 줄였지만 현재 개발 검증 총점 오차는 보정 전과 같았다. 기존 첫 후보보다 Rule 총점 오차가 커져 공통 모델로 승격하지 않는다.",
            "후속 특징 개선은 학습 그룹 내부에서 지면 기준 발뒤꿈치·발끝 기울기·정지 구간 기준 높이를 비교하고, 새로 보지 않은 참여자·원본을 확보한 뒤 확정 시험한다. 현재 검증에 맞춘 숫자 수동 변경은 하지 않는다.",
        ],
        promoted=False,
        independent_final_test=False,
    )
    audit.save_json(OUT / "failure_analysis.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
