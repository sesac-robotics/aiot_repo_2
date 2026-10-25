"""W 개발 규칙의 수치 경계 민감도를 기록한다. 경계를 최적화하지 않는다."""

import numpy as np
import audit
import w_morphology


def run() -> dict:
    """관절·자동 구간을 고정하고 한 경계씩 바꿨을 때 등급 변화를 저장한다.

    Returns:
        dict: 설정별 변화 건수·자료별 등급. 경계는 굽힘 도, 몸통·상완 길이
        비율, 좌우 차이 도 단위이다. 직접 라벨을 선택·최적화에 쓰지 않으며
        실제 재촬영 일관성이나 독립 정확도를 뜻하지 않는다.

    Raises:
        OSError: 캐시·진단 파일 읽기 또는 결과 저장 실패.
        ValueError: 관절·구간 형상 불일치.
    """
    rows = []
    for name in ("w_challenge_results.json", "w_challenge_v2_results.json"):
        rows.extend(audit.read_json(audit.OUT / name)["results"])
    original = w_morphology.POLICY.copy()
    variations = [
        (key, original[key] + delta)
        for key, deltas in (
            ("bend_good_deg", (-5, 5)),
            ("hand_good_torso_ratio", (-0.025, 0.025)),
            ("elbow_up_good_arm_ratio", (-0.1, 0.1)),
            ("symmetry_good_deg", (-10, 10)),
        )
        for delta in deltas
    ]
    reports = []
    try:
        for key, value in variations:
            w_morphology.POLICY.clear()
            w_morphology.POLICY.update(original)
            w_morphology.POLICY[key] = value
            cases = []
            for row in rows:
                window = row["phase_detection"]["phases"].get("W_attempt")
                if not window:
                    continue
                with np.load(row["extraction"]["cache"]) as data:
                    result = w_morphology.evaluate(
                        data["image_xy"],
                        window["start_index"],
                        window["end_index"],
                    )
                cases.append(
                    dict(
                        id=row["id"],
                        baseline_grade=row["automatic"]["grade"],
                        varied_grade=result["grade"],
                    )
                )
            reports.append(
                dict(
                    parameter=key,
                    value=value,
                    changed=sum(
                        x["baseline_grade"] != x["varied_grade"] for x in cases
                    ),
                    compared=len(cases),
                    cases=cases,
                )
            )
    finally:
        w_morphology.POLICY.clear()
        w_morphology.POLICY.update(original)
    report = dict(
        policy=w_morphology.VERSION,
        baseline=original,
        variations=reports,
        thresholds_optimized=False,
        labels_used_for_optimization=False,
        independent_final_test=False,
        scope="cached_coordinates_fixed_auto_windows_single_parameter_sensitivity_not_recapture",
    )
    audit.save_json(audit.OUT / "w_policy_sensitivity.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            (r["parameter"], r["value"], r["changed"])
            for r in run()["variations"]
        ]
    )
