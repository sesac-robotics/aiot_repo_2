"""공통 관절 관측률 기준으로 보류된 입력의 관절별 누락을 기록한다.

영상 직접 검토를 대신하지 않으며 라벨 등급이나 관절을 보정하지 않는다.
"""

import numpy as np

import audit

JOINT_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    31: "left_foot_index",
    32: "right_foot_index",
}


def run() -> dict:
    """현재 비교에서 보류된 자료의 관측률을 관절별로 저장한다.

    Returns:
        dict: 자료 ID·보류 사유·시점별 0~1 관측 비율. 자동 진단이다.

    Raises:
        OSError: 비교 JSON 또는 관절 캐시를 읽을 수 없는 경우.
        ValueError: 관절 좌표 배열 형식이 잘못된 경우.
    """
    comparison = audit.read_json(audit.OUT / "comparison.json")
    rows = []
    for exercise, result in comparison["results"].items():
        for entry in result["unavailable"]:
            folder = audit.OUT / "heavy" / entry["id"]
            views = {}
            for path in folder.glob("*.npz"):
                if path.stem not in ("side", "front", "primary"):
                    continue
                with np.load(path) as data:
                    coverage = np.isfinite(data["image_xy"]).all(2).mean(0)
                views[path.stem] = {
                    name: float(coverage[index])
                    for index, name in JOINT_NAMES.items()
                }
            rows.append(dict(exercise=exercise, **entry, joint_coverage=views))
    report = dict(
        rows=rows,
        direct_visual_review=False,
        required_temporal_coverage=0.8,
        labels_modified=False,
        missing_joints_filled=False,
        scope_ko="공통 입력 관측률 보류의 자동 진단. 직접 라벨 관측 가능 여부와 Heavy 관절 추출 성공 여부는 별개다.",
    )
    labels = {r["id"]: r for r in audit.read_json(audit.OUT / "labels.json")}
    failures = []
    for row in comparison["results"]["ytw"]["comparisons"]:
        target = labels[row["id"]]
        index = target["criterion_ids"].index("w_shape")
        rule = row["rule_details"]["levels"][index]
        if rule is not None and rule != target["criterion_levels"][index]:
            failures.append(
                dict(
                    id=row["id"],
                    label_draft_W_grade=target["criterion_levels"][index],
                    rule_W_grade=rule,
                    ai_W_grade=row["ai_levels"][index],
                    evidence=row["rule_details"]["evidence"]["w_shape"],
                    phase_detection=row["rule_details"]["evidence"][
                        "phase_detection"
                    ],
                    expert_validated=False,
                    label_changed_to_match_rule=False,
                    next_review_ko="측면·사선 전완 방향을 확인할 수 있는 영상 보충과 독립 항목 재검토. 노출된 개발 검증 점수에 맞춘 경계 조정은 하지 않음.",
                )
            )
    audit.save_json(
        audit.OUT / "ytw_w_failure_cases.json",
        dict(
            cases=failures,
            historical_cases_path="snapshots/pre_w_contract_fix/ytw_w_failure_cases.json",
            current_rule_version=comparison["rule_version"],
            independent_final_test=False,
            score_labels_modified=False,
        ),
    )
    audit.save_json(audit.OUT / "observation_failure_details.json", report)
    return report


if __name__ == "__main__":
    print(run())
