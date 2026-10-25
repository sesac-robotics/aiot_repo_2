"""기존 W 개발 사례의 시간 집계와 앞뒤 0.1·0.2초 경계 변동을 비교한다."""

import numpy as np

import audit
import w_morphology
import w_temporal_trial
from resources import check_resources


def run() -> dict:
    """기존·추가 W 캐시의 동일 자동 구간을 비교해 저장한다.

    Returns:
        dict: 원래 규칙·시간 집계안의 등급 오차와 경계 변동률. 직접 라벨은
        비교에만 쓰고 경계값·구간 선택에는 쓰지 않는다. 독립 시험이 아니다.

    Raises:
        OSError: 캐시·결과 파일 접근 실패.
        ValueError: 캐시 좌표·구간 형상 불일치.
        RuntimeError: 자원 보호 중단.
    """
    results = []
    for stem in ["w_challenge", "w_challenge_v2", "w_targeted", "next_w"]:
        check_resources()
        data = audit.read_json(audit.OUT / f"{stem}_results.json")
        for row in data["results"]:
            phases = row.get("phase_detection", row.get("phases", {}))
            win = phases.get("phases", {}).get("W_attempt")
            if not win:
                continue
            with np.load(row["extraction"]["cache"]) as cache:
                seq = cache["image_xy"].copy()
                # 두 방식에 동일한 미관측 마스크를 적용한다.
                if "observed" in cache:
                    seq[~cache["observed"].astype(bool)] = np.nan
            a, b = win["start_index"], win["end_index"]
            variants = {}
            for name, module in [
                ("median", w_morphology),
                ("temporal", w_temporal_trial),
            ]:
                base = module.evaluate(seq, a, b)
                perturbations = []
                for da, db in [
                    (-2, -2),
                    (-1, -1),
                    (1, 1),
                    (2, 2),
                    (-2, 2),
                    (2, -2),
                ]:
                    low, high = max(0, a + da), min(len(seq) - 1, b + db)
                    if high - low + 1 < 3:
                        continue
                    out = module.evaluate(seq, low, high)
                    perturbations.append(
                        dict(start=low, end=high, grade=out["grade"])
                    )
                variants[name] = dict(base=base, perturbations=perturbations)
            results.append(
                dict(
                    batch=stem,
                    id=row["id"],
                    draft=row.get("draft_grade", row.get("W_grade_draft")),
                    same_attempt_core_overlaps=row.get(
                        "same_attempt_core_overlaps", True
                    ),
                    variants=variants,
                )
            )
    summary = {}
    for name in ["median", "temporal"]:
        paired = [
            r
            for r in results
            if r["same_attempt_core_overlaps"]
            and r["draft"] is not None
            and r["variants"][name]["base"]["grade"] is not None
        ]
        changes, observed = 0, 0
        for row in results:
            item = row["variants"][name]
            for change in item["perturbations"]:
                if (
                    change["grade"] is not None
                    and item["base"]["grade"] is not None
                ):
                    observed += 1
                    changes += change["grade"] != item["base"]["grade"]
        summary[name] = dict(
            paired=len(paired),
            grade_MAE=(
                float(
                    np.mean(
                        [
                            abs(
                                r["draft"]
                                - r["variants"][name]["base"]["grade"]
                            )
                            for r in paired
                        ]
                    )
                )
                if paired
                else None
            ),
            boundary_changed=changes,
            boundary_observable=observed,
        )
    report = dict(
        results=results,
        summary=summary,
        adopted=False,
        independent_final_test=False,
        common_policy_modified=False,
        coordinate_mask_applied_to_both=True,
        boundary_variation_seconds=[0.1, 0.2],
        scope="development_trial_not_recapture_or_final_accuracy",
    )
    audit.save_json(audit.OUT / "w_temporal_trial_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["summary"])
