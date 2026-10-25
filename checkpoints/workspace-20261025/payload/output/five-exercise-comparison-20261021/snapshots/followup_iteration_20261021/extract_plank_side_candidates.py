"""직접 초안이 동결된 3초 측면 플랭크 후보의 공통 관측을 검사한다."""

import numpy as np

import audit
import benchmark
from extract import extract_view
import rule_score


def run() -> dict:
    """새 3개 구간을 기존 Heavy·공통 입력 조건으로 검사한다.

    Returns:
        dict: 구간별 추출·보류 사유·규칙 진단. 직접 초안 SHA를 보존하며
        규칙 결과는 정답 라벨로 사용하지 않는다. 독립 최종 시험이 아니다.

    Raises:
        OSError: 입력·캐시·결과 접근 실패.
        ValueError: 동결 초안·원본·관절 계약 불일치.
        RuntimeError: 자원 보호 또는 관절 추출 실패.
    """
    drafts_path = audit.OUT / "plank_side_direct_labels.json"
    frozen = audit.digest(drafts_path)
    direct = audit.read_json(drafts_path)
    exports = audit.read_json(audit.OUT / "plank_side_export_plan_exports.json")
    results = []
    for export in exports["records"]:
        row = dict(
            id=export["id"],
            exercise="plank",
            video_sha256=export["video_sha256"],
            required_views=["side"],
            segment=dict(
                start_frame=0, end_frame=export["frames"] - 1, fps=export["fps"]
            ),
        )
        extraction = extract_view(row, "side", export["video_path"])
        seq, _, holds = benchmark.load_input(row)
        rule = rule_score.evaluate(
            "plank", seq, None, direct["criterion_ids"], direct["weights"]
        )
        results.append(
            dict(
                id=row["id"],
                extraction=extraction,
                segment=row["segment"],
                holds=holds,
                common_input_passed=not holds,
                rule_diagnostic=rule,
                AI_total_score=None,
                independent_final_test=False,
                observed_joint_fraction=np.isfinite(seq)
                .all(2)
                .mean(0)
                .tolist(),
            )
        )
    if frozen != audit.digest(drafts_path):
        raise ValueError("Direct draft changed")
    report = dict(
        results=results,
        direct_labels_sha256=frozen,
        expert_validated=False,
        independent_final_test=False,
        rule_used_as_label=False,
    )
    audit.save_json(audit.OUT / "plank_side_joint_results.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            (r["id"], r["holds"], r["rule_diagnostic"]["levels"])
            for r in run()["results"]
        ]
    )
