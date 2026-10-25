"""채택 전 낮은 팔 구간 확장안을 기존·오류 개발 캐시에 대조한다.

라벨·공통 규칙·학습 모델을 수정하지 않는다. 재사용 개발 진단이다.
"""

import numpy as np
import audit
import ytw_phases
import ytw_phase_trial
import ytw_phase_arm_trial
import ytw_partial_trial
import ytw_bend_dwell_trial
import w_morphology


def run() -> dict:
    """세 묶음의 기존 캐시에서 단계 확장 전후 구간·등급을 저장한다.

    Returns:
        dict: 후보별 같은 시도 겹침과 W 등급. 0~2 등급이며 총점은 아니다.

    Raises:
        OSError: 직접 검토 파일 또는 캐시 접근 실패.
        ValueError: 캐시 좌표 형상 오류.
    """
    results = []
    for stem, cache in (
        ("w_challenge", "w_challenge_cache"),
        ("w_challenge_v2", "w_challenge_v2_cache"),
        ("w_targeted", "w_targeted_cache"),
    ):
        direct = audit.read_json(audit.OUT / f"{stem}_direct_review.json")
        for row in direct["reviews"]:
            with np.load(audit.OUT / cache / row["id"] / "primary.npz") as data:
                seq, frames = data["image_xy"], data["frame_numbers"]
            item = dict(
                batch=stem,
                id=row["id"],
                draft=row["W_grade_draft"],
                variants={},
            )
            core = row["direct_W_core_frames"]
            for name, module in (
                ("current", ytw_phases),
                ("trial", ytw_phase_trial),
                ("arm_trial", ytw_phase_arm_trial),
                ("partial_trial", ytw_partial_trial),
                ("bend_dwell", ytw_bend_dwell_trial),
            ):
                phases = module.detect(seq)
                win = phases["phases"].get("W_attempt")
                grade = (
                    w_morphology.evaluate(
                        seq, win["start_index"], win["end_index"]
                    )["grade"]
                    if win
                    else None
                )
                overlap = bool(
                    core
                    and win
                    and frames[win["start_index"]] <= core[1]
                    and frames[win["end_index"]] >= core[0]
                )
                item["variants"][name] = dict(
                    grade=grade, same_attempt_overlap=overlap, phases=phases
                )
            results.append(item)
    report = dict(
        results=results,
        independent_final_test=False,
        adopted=False,
        annotation_reaudit_after_rule_exposure=True,
    )
    audit.save_json(audit.OUT / "phase_bend_dwell_trial.json", report)
    return report


if __name__ == "__main__":
    report = run()
    for batch in sorted({r["batch"] for r in report["results"]}):
        for name in (
            "current",
            "trial",
            "arm_trial",
            "partial_trial",
            "bend_dwell",
        ):
            paired = [
                r
                for r in report["results"]
                if r["batch"] == batch
                and r["draft"] is not None
                and r["variants"][name]["grade"] is not None
                and r["variants"][name]["same_attempt_overlap"]
            ]
            print(
                batch,
                name,
                len(paired),
                (
                    sum(
                        abs(r["draft"] - r["variants"][name]["grade"])
                        for r in paired
                    )
                    / len(paired)
                    if paired
                    else None
                ),
            )
