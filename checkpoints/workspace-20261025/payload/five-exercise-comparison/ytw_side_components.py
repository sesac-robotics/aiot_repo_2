"""확인된 측면 영상에서 전경 항목과 Y 팔의 측면 성분을 계산한다.

한쪽 팔의 측면 도달은 양팔 Y 방향 전체를 뜻하지 않는다. 별도 시점·
별도 시도의 결과를 합쳐 Y–T–W 총점을 만들지 않는다.
"""

import numpy as np

import audit
import geometry


def evaluate(sequence: np.ndarray, verified_side_view: bool) -> dict:
    """측면의 몸통 기울기·흔들림과 한쪽 팔 도달 성분을 평가한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        verified_side_view: 직접 구도 검토에서 측면을 확인했는지 여부.

    Returns:
        dict: 전경 0·1·2 등급, 한쪽 Y 측면 성분 0·1·2, 근거 각도(도).
        기울기 p10 35도 이상·변동폭 12도 이하가 전경 2 초안이다.
        Y 성분은 팔 도달 p90 150도·도달 시 팔꿈치 p10 155도 이상이다.
        미관측·정면이면 None이며 양팔 Y 전체 등급·총점은 항상 None이다.

    Raises:
        ValueError: 좌표 형상 또는 빈 시계열 오류.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 pixels")
    held = dict(
        hinge_grade=None,
        y_sagittal_component_grade=None,
        full_y_grade=None,
        total_score=None,
        expert_validated=False,
        side_verified=bool(verified_side_view),
    )
    if not verified_side_view or len(seq) < 5:
        return dict(
            held, hold_reason="side_view_or_temporal_samples_unconfirmed"
        )
    choices = []
    for side, shoulder, elbow, wrist, hip in [
        ("left", 11, 13, 15, 23),
        ("right", 12, 14, 16, 24),
    ]:
        coverage = float(
            np.isfinite(seq[:, [shoulder, hip]]).all((1, 2)).mean()
        )
        choices.append((coverage, side, shoulder, elbow, wrist, hip))
    coverage, side, shoulder, elbow, wrist, hip = max(
        choices, key=lambda r: r[0]
    )
    if coverage < 0.8:
        return dict(held, hold_reason="trunk_observation_below_80_percent")
    trunk = seq[:, shoulder] - seq[:, hip]
    tilt = np.degrees(np.arctan2(np.abs(trunk[:, 0]), np.abs(trunk[:, 1])))
    finite = tilt[np.isfinite(tilt)]
    low, high = np.percentile(finite, [10, 90])
    median = float(np.median(finite))
    hinge = 2 if low >= 35 and high - low <= 12 else 1 if median >= 15 else 0
    arm_side, shoulder, elbow, wrist, hip = max(
        [("left", 11, 13, 15, 23), ("right", 12, 14, 16, 24)],
        key=lambda r: float(
            np.isfinite(seq[:, list(r[1:])]).all((1, 2)).mean()
        ),
    )
    arm = geometry.angle_deg(seq[:, hip], seq[:, shoulder], seq[:, elbow])
    bend = geometry.angle_deg(seq[:, shoulder], seq[:, elbow], seq[:, wrist])
    mask = np.isfinite(arm) & np.isfinite(bend)
    reach_grade = None
    reach = None
    extension = None
    if mask.mean() >= 0.8:
        reach = float(np.percentile(arm[mask], 90))
        peak = mask & (arm >= max(120, reach - 15))
        if peak.sum() >= 3:
            extension = float(np.percentile(bend[peak], 10))
            reach_grade = (
                2
                if reach >= 150 and extension >= 155
                else 1 if reach >= 120 and extension >= 130 else 0
            )
    return dict(
        held,
        selected_side=side,
        selected_arm_side=arm_side,
        hinge_grade=hinge,
        y_sagittal_component_grade=reach_grade,
        trunk_tilt_p10_p90_deg=[float(low), float(high)],
        trunk_tilt_median_deg=median,
        arm_reach_p90_deg=reach,
        elbow_extension_at_reach_p10_deg=extension,
        hold_reason=None,
    )


def run() -> dict:
    """이미 직접 확인한 Paul 측면 시도의 부분 규칙 결과를 저장한다.

    Returns:
        dict: 동일 참여자·원본 한 개의 세 구간 결과. 독립 검증·전체 Y
        정확도·별도 AI 학습 완료로 표현하지 않는다.

    Raises:
        OSError: 관절 캐시·출력 접근 실패.
        ValueError: 관절 입력 형상 오류.
    """
    records = []
    for identifier in [
        "youtube_paul_y_side",
        "youtube_paul_t_side",
        "youtube_paul_w_side",
    ]:
        with np.load(audit.OUT / "heavy" / identifier / "primary.npz") as cache:
            seq = cache["image_xy"].copy()
        result = evaluate(seq, True)
        if identifier != "youtube_paul_y_side":
            result["y_sagittal_component_grade"] = None
        records.append(
            dict(
                id=identifier,
                direct_hinge_draft=2,
                direct_y_sagittal_component_draft=(
                    2 if identifier == "youtube_paul_y_side" else None
                ),
                rule=result,
                source_group="youtube_I7_A0R_9z-0_all_segments",
                independent_final_test=False,
                AI_component_model=None,
            )
        )
    report = dict(
        records=records,
        person_groups=1,
        source_groups=1,
        full_total_created=False,
        common_rules_modified=False,
        expert_validated=False,
        independent_final_test=False,
        criteria_version="side_components_engineering_v1_unvalidated",
    )
    audit.save_json(audit.OUT / "ytw_side_component_results.json", report)
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>Y·전경 측면 성분</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px}td,th{padding:10px;border-bottom:1px solid #aaa}</style><h1>Y·전경 측면 성분 규칙</h1><p><a href="unified_status.html">통합 현황</a> · <a href="ytw_side_component_results.json">수치 근거</a></p><p>기존 Paul 측면 영상에서 직접 확인한 전경과 Y 팔의 측면 도달을 별도로 계산했습니다. 같은 참여자·원본 한 개이며 별도 시도들을 동기화한 다중 시점으로 취급하지 않습니다. 한쪽 측면 도달을 양팔 Y 전체 점수로 바꾸지 않습니다.</p><table><tr><th>구간</th><th>전경 직접 초안 / 규칙</th><th>Y 측면 성분 직접 초안 / 규칙</th></tr>'
    ]
    for r in records:
        parts.append(
            f'<tr><td>{r["id"]}</td><td>{r["direct_hinge_draft"]} / {r["rule"]["hinge_grade"]}</td><td>{r["direct_y_sagittal_component_draft"]} / {r["rule"]["y_sagittal_component_grade"]}</td></tr>'
        )
    parts.append(
        "</table><p>수치 경계는 미검증 공학 초안입니다. 한 참여자에게서 일부 성분을 계산한 것으로, Y–T–W 총점·별도 AI 학습·독립 정확도 검증이 완료된 것은 아닙니다. 다른 참여자의 같은 시도에서 Y 방향과 전경을 관측할 자료가 더 필요합니다.</p></html>"
    )
    (audit.OUT / "ytw_side_components.html").write_text(
        "".join(parts), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(run())
