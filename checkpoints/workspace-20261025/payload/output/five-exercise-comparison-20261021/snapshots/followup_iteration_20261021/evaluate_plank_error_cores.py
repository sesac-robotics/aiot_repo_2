"""직접 초안을 동결한 추가 플랭크 구간의 관절·항목 규칙을 진단한다."""

import numpy as np

import audit
from extract import extract_view
import rule_score


def run() -> dict:
    """5개 구간에서 Heavy를 추출하고 직접 관측 가능한 항목만 비교한다.

    Returns:
        dict: 원본 프레임·캐시·관측 비율·직접 초안과 규칙 항목 결과.
        짧거나 사선인 자료를 공통 3초 총점 학습으로 편입하지 않는다.
        규칙은 라벨을 생성하지 않으며 보류 항목을 재배점하지 않는다.

    Raises:
        OSError: 초안·원본·결과 파일 접근 실패.
        ValueError: 관절·구간·원본 SHA 불일치.
        RuntimeError: 자원 보호 또는 관절 추출 실패.
    """
    path = audit.OUT / "plank_error_direct_labels.json"
    frozen = audit.digest(path)
    direct = audit.read_json(path)
    results = []
    for row in direct["reviews"]:
        item = dict(row, exercise="plank", required_views=["item_only_review"])
        extraction = extract_view(
            item, "primary", row["video_path"], audit.OUT / "plank_error_cache"
        )
        with np.load(extraction["cache"]) as data:
            seq = data["image_xy"].copy()
            frames = data["frame_numbers"].tolist()
        rule = rule_score.evaluate(
            "plank", seq, None, direct["criterion_ids"], direct["weights"]
        )
        paired = []
        for name, weight, draft, predicted in zip(
            direct["criterion_ids"],
            direct["weights"],
            row["draft_levels"],
            rule["levels"],
            strict=True,
        ):
            paired.append(
                dict(
                    criterion_id=name,
                    draft_grade=draft,
                    rule_grade=predicted if draft is not None else None,
                    absolute_weighted_error=(
                        abs(draft - predicted) * weight / 2
                        if draft is not None and predicted is not None
                        else None
                    ),
                )
            )
        results.append(
            dict(
                id=row["id"],
                extraction=extraction,
                original_frame_numbers=frames,
                observed_joint_fraction=float(np.isfinite(seq).all(2).mean()),
                items=paired,
                rule_diagnostic=rule,
                total_score=None,
                AI_total_score=None,
                full_score_training_eligible=False,
            )
        )
    if frozen != audit.digest(path):
        raise ValueError("Direct drafts changed during extraction")
    report = dict(
        results=results,
        direct_labels_sha256=frozen,
        source_count=2,
        directly_observed_hip_raise_drafts=2,
        full_score_training_records_added=0,
        independent_final_test=False,
        expert_validated=False,
        raw_rule_total_not_approved_as_score=True,
    )
    audit.save_json(audit.OUT / "plank_error_results.json", report)
    return report


if __name__ == "__main__":
    print([(x["id"], x["items"]) for x in run()["results"]])
