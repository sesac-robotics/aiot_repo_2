"""W 입력·구간 실험을 같은 표본과 명시적인 시간 조건으로 감사한다.

등급 초안은 재사용한 개발 자료이며 전문가 정답이 아니다. 구간의 작은
겹침을 검출 성공으로 세지 않도록 포함률·중앙점·시간 IoU를 함께 저장한다.
"""

import pathlib
import numpy as np
import audit


def interval_metrics(core: list, window: list | None) -> dict:
    """양 끝 포함 원본 프레임 구간 두 개의 위치 지표를 반환한다.

    Args:
        core: 직접 표시한 W 핵심 구간의 시작·끝 프레임 번호.
        window: 자동 구간의 시작·끝 프레임 번호. 미검출은 None이다.

    Returns:
        dict: 무차원 겹침 비율·핵심 중앙점 포함 여부. strict_core_match는
        핵심 포함률 0.5 이상이고 중앙점을 포함하는 개발 진단 조건이다.
        직접 핵심 구간은 전체 단계 경계 정답이 아니므로 IoU는 정확도가 아니다.

    Raises:
        ValueError: 구간 길이가 두 개가 아니거나 역순·음수인 경우.
    """
    for pair in (core, window):
        if pair is not None and (
            len(pair) != 2 or pair[0] < 0 or pair[1] < pair[0]
        ):
            raise ValueError("Invalid inclusive frame interval")
    if window is None:
        return dict(
            overlap=False,
            core_coverage=0.0,
            temporal_iou=0.0,
            contains_core_midpoint=False,
            strict_core_match=False,
        )
    overlap = max(0, min(core[1], window[1]) - max(core[0], window[0]) + 1)
    coverage = overlap / (core[1] - core[0] + 1)
    union = core[1] - core[0] + window[1] - window[0] + 2 - overlap
    midpoint = window[0] <= sum(core) / 2 <= window[1]
    return dict(
        overlap=overlap > 0,
        core_coverage=coverage,
        temporal_iou=overlap / union,
        contains_core_midpoint=midpoint,
        strict_core_match=bool(midpoint and coverage >= 0.5),
    )


def matched_summary(rows: list[dict]) -> dict:
    """원본·크롭 양쪽에서 등급을 계산한 동일 사례만 오차를 비교한다.

    Args:
        rows: draft와 baseline_manual·roi_manual의 0~2 등급을 담은 목록.

    Returns:
        dict: 알려진 초안 수, 평가 가능 수, 공통 사례 ID와 양쪽 등급 MAE.
        미관측 초안에 대한 자동 등급 수는 별도 기록하며 오차에 넣지 않는다.

    Raises:
        KeyError: 필수 필드가 없는 경우.
    """

    def grade(row, name):
        return (row[name + "_manual"] or {}).get("grade")

    known = [r for r in rows if r["draft"] is not None]
    paired = [
        r
        for r in known
        if all(grade(r, n) is not None for n in ("baseline", "roi"))
    ]
    result = dict(
        known=len(known),
        paired_ids=[r["id"] for r in paired],
        paired_count=len(paired),
    )
    for name in ("baseline", "roi"):
        result[name] = dict(
            assessed=sum(grade(r, name) is not None for r in known),
            paired_MAE_grade=(
                sum(abs(grade(r, name) - r["draft"]) for r in paired)
                / len(paired)
                if paired
                else None
            ),
            grades_on_unknown=sum(
                r["draft"] is None and grade(r, name) is not None for r in rows
            ),
        )
    result["lost_assessment_ids"] = [
        r["id"]
        for r in known
        if grade(r, "baseline") is not None and grade(r, "roi") is None
    ]
    return result


def run() -> dict:
    """저장된 개발 실험만 읽어 대응 비교와 강화된 위치 진단을 저장한다.

    Returns:
        dict: 입력·구간 실험 지표와 코드·입력 SHA-256. 공통 채택은 False다.

    Raises:
        OSError: 코드·캐시·결과 파일 접근 실패.
        ValueError: 프레임 구간이 잘못된 경우.
        KeyError: 필수 구조가 없는 경우.
    """
    inputs = [
        "phase_bend_dwell_trial.json",
        "w_roi_trial_results.json",
        "w_roi_context_results.json",
    ]
    roi = {
        name: matched_summary(audit.read_json(audit.OUT / filename)["results"])
        for name, filename in (
            ("upper_body", inputs[1]),
            ("head_to_knee", inputs[2]),
        )
    }
    phase_rows = audit.read_json(audit.OUT / inputs[0])["results"]
    direct = {}
    for stem in ("w_challenge", "w_challenge_v2", "w_targeted"):
        inputs.append(f"{stem}_direct_review.json")
        for row in audit.read_json(audit.OUT / inputs[-1])["reviews"]:
            direct[(stem, row["id"])] = row
    diagnostics = []
    for row in phase_rows:
        source = direct[(row["batch"], row["id"])]
        core = source["direct_W_core_frames"]
        if core is None:
            continue
        with np.load(
            audit.OUT / (row["batch"] + "_cache") / row["id"] / "primary.npz"
        ) as data:
            frames = data["frame_numbers"]
        for name, variant in row["variants"].items():
            window = variant["phases"]["phases"].get("W_attempt")
            detected = (
                [
                    int(frames[window["start_index"]]),
                    int(frames[window["end_index"]]),
                ]
                if window
                else None
            )
            diagnostics.append(
                dict(
                    batch=row["batch"],
                    id=row["id"],
                    variant=name,
                    core_frames=core,
                    automatic_frames=detected,
                    draft=row["draft"],
                    grade=variant["grade"],
                    **interval_metrics(core, detected),
                )
            )
    summaries = []
    for batch in sorted({r["batch"] for r in diagnostics}):
        for variant in sorted({r["variant"] for r in diagnostics}):
            rows = [
                r
                for r in diagnostics
                if r["batch"] == batch
                and r["variant"] == variant
                and r["draft"] is not None
            ]
            strict = [
                r
                for r in rows
                if r["strict_core_match"] and r["grade"] is not None
            ]
            summaries.append(
                dict(
                    batch=batch,
                    variant=variant,
                    known_with_core=len(rows),
                    any_overlap=sum(r["overlap"] for r in rows),
                    strict_core_match=sum(r["strict_core_match"] for r in rows),
                    strict_assessed=len(strict),
                    strict_grade_MAE=(
                        sum(abs(r["grade"] - r["draft"]) for r in strict)
                        / len(strict)
                        if strict
                        else None
                    ),
                )
            )
    code = pathlib.Path(__file__).parent
    report = dict(
        roi=roi,
        phase_summaries=summaries,
        phase_rows=diagnostics,
        adopted=False,
        independent_final_test=False,
        labels_modified=False,
        protocol_ko="구간 핵심의 중앙점 포함 및 50% 이상 포함. 사후 개발 진단이며 단계 경계 정답·독립 정확도가 아님.",
        input_sha256={name: audit.digest(audit.OUT / name) for name in inputs},
        code_sha256={
            name: audit.digest(code / name)
            for name in (
                "w_trial_audit.py",
                "ytw_bend_dwell_trial.py",
                "ytw_phases.py",
                "w_morphology.py",
                "w_roi_trial.py",
                "w_roi_context_trial.py",
            )
        },
    )
    audit.save_json(audit.OUT / "w_trial_audit.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(result["roi"])
    print(
        [
            r
            for r in result["phase_summaries"]
            if r["variant"] in ("current", "bend_dwell")
        ]
    )
