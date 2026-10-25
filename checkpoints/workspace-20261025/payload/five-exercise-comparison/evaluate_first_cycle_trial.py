"""첫 W 시행과 다른 반복 시연을 고르는 문제를 단계 선택 정책별로 비교한다."""

import numpy as np

import audit
from cpu_analysis_guard import check_resources
import ytw_phases
import ytw_first_cycle_trial


def run() -> dict:
    """직접 핵심 구간과 자동 구간의 겹침을 비교하며 점수를 학습하지 않는다.

    Returns:
        dict: 기존·최초 시행 선택별 W 탐지 수, 핵심 구간 겹침 수·IoU.
        이미 노출된 개발 자료의 후향 진단이며 새 독립 성능이 아니다.

    Raises:
        OSError: 직접 구간·관절·결과 접근 실패.
        ValueError: 캐시 프레임·좌표 길이 불일치.
        RuntimeError: 자원 보호 조건 위반.
    """
    rows = []
    for stem in ["w_challenge", "w_challenge_v2", "w_targeted", "next_w"]:
        entries = {
            r["id"]: r["extraction"]
            for r in audit.read_json(audit.OUT / f"{stem}_results.json")[
                "results"
            ]
        }
        for draft in audit.read_json(audit.OUT / f"{stem}_direct_review.json")[
            "reviews"
        ]:
            if draft["direct_W_core_frames"]:
                rows.append(
                    dict(
                        batch=stem, draft=draft, extraction=entries[draft["id"]]
                    )
                )
    entries = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_extraction.json")[
            "W_records"
        ]
    }
    for draft in audit.read_json(audit.OUT / "followup_W_direct.json")[
        "reviews"
    ]:
        if draft["direct_W_core_frames"]:
            rows.append(
                dict(
                    batch="followup",
                    draft=draft,
                    extraction=entries[draft["id"]],
                )
            )
    results = []
    for row in rows:
        check_resources()
        draft = row["draft"]
        with np.load(row["extraction"]["cache"]) as cache:
            seq = cache["image_xy"].copy()
            frames = cache["frame_numbers"]
            if "observed" in cache:
                seq[~cache["observed"].astype(bool)] = np.nan
        a, b = draft["direct_W_core_frames"]
        variants = {}
        for name, module in [
            ("longest", ytw_phases),
            ("first", ytw_first_cycle_trial),
        ]:
            phase = module.detect(seq)
            window = phase["phases"].get("W_attempt")
            interval = (
                [
                    int(frames[window["start_index"]]),
                    int(frames[window["end_index"]]),
                ]
                if window
                else None
            )
            overlap = (
                max(0, min(b, interval[1]) - max(a, interval[0]) + 1)
                if interval
                else 0
            )
            union = (
                max(b, interval[1]) - min(a, interval[0]) + 1
                if interval
                else None
            )
            variants[name] = dict(
                interval=interval,
                core_overlap=overlap > 0,
                IoU=overlap / union if union else 0.0,
                complete=phase["complete_attempt_sequence"],
                hold_reason=phase["hold_reason"],
            )
        results.append(
            dict(
                batch=row["batch"],
                id=draft["id"],
                direct_core=[a, b],
                variants=variants,
            )
        )
    summary = {
        name: dict(
            reference_cases=len(results),
            detected=sum(r["variants"][name]["complete"] for r in results),
            core_overlapping=sum(
                r["variants"][name]["core_overlap"] for r in results
            ),
            mean_IoU=float(
                np.mean([r["variants"][name]["IoU"] for r in results])
            ),
        )
        for name in ["longest", "first"]
    }
    report = dict(
        results=results,
        summary=summary,
        posthoc_development_trial=True,
        independent_final_test=False,
        expert_validated=False,
        common_policy_modified=False,
        note_ko="첫 시행이라는 선택 계약을 시험한 것이다. 겹침만으로 단계 전체의 정확성이나 W 점수 정확도를 보증하지 않는다.",
    )
    audit.save_json(audit.OUT / "first_cycle_trial_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["summary"])
