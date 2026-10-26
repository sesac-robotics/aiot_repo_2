"""YTW 자동 단계와 관측된 투영 특징만 계산한다. 정답 점수는 입력하지 않는다."""

import numpy as np

from app.scoring import (
    geometry,
    recovery_w_phase,
    w_morphology,
    ytw_partial_trial,
    ytw_phases,
)


def describe_ytw(seq: np.ndarray) -> list[dict]:
    """자동 구간 인식에서 보이는 Y·T·W와 전경 대리 특징을 반환한다.

    Args:
        seq: N×33×2 픽셀 관절. 누락은 NaN이며 보간하지 않는다.

    Returns:
        list[dict]: 네 항목의 관측·특징·규칙 등급. Y·전경의 규칙 총점은
        검증되지 않아 None이다. AI 입력에는 단계 정답·규칙 등급이 없다.

    Raises:
        ValueError: 좌표 또는 자동 구간 계약 오류.
    """
    phases = ytw_partial_trial.detect(seq)["phases"]
    channels = geometry.channels(seq)
    results = []
    for name, phase in [
        ("y_reach", "Y_attempt"),
        ("t_reach", "T_attempt"),
        ("w_shape", "W_attempt"),
    ]:
        window = phases.get(phase)
        feature = None
        grade = None
        if window:
            a, b = (window["start_index"], window["end_index"])
            subset = seq[a : b + 1]
            if name == "w_shape":
                result = w_morphology.evaluate(seq, a, b)
                if result["grade"] is not None:
                    feature = [
                        result["arms"][s][k]
                        for s in ["left", "right"]
                        for k in [
                            "projected_bend_deg",
                            "hand_above_elbow_torso_ratio",
                            "elbow_up_upper_arm_ratio",
                        ]
                    ] + [result["bilateral_bend_difference_deg"]]
                    grade = result["grade"]
            elif (
                len(subset) >= 3
                and np.isfinite(subset[:, [11, 12, 13, 14, 15, 16, 23, 24]])
                .all((1, 2))
                .mean()
                >= 0.8
            ):
                elevation = np.minimum(
                    channels["left_elevation_deg"][a : b + 1],
                    channels["right_elevation_deg"][a : b + 1],
                )
                elbow = np.minimum(
                    channels["left_elbow_deg"][a : b + 1],
                    channels["right_elbow_deg"][a : b + 1],
                )
                feature = [
                    float(np.nanpercentile(x, q))
                    for x in [elevation, elbow]
                    for q in [10, 50, 90]
                ]
                if name == "t_reach":
                    error = abs(feature[1] - 90)
                    grade = 2 if error <= 15 else 1 if error <= 30 else 0
        results.append(
            dict(
                item=name,
                eligible=feature is not None,
                feature=feature,
                rule_grade=grade,
                measurement=None,
                higher_is_better=True,
                window=window,
            )
        )
    required = [11, 12, 23, 24]
    feature = None
    if (
        len(seq) >= 5
        and np.isfinite(seq[:, required]).all((1, 2)).mean() >= 0.8
    ):
        shoulder = (seq[:, 11] + seq[:, 12]) / 2
        hip = (seq[:, 23] + seq[:, 24]) / 2
        torso = np.linalg.norm(shoulder - hip, axis=1)
        ratio = np.linalg.norm(seq[:, 11] - seq[:, 12], axis=1) / np.maximum(
            torso, 1
        )
        tilt = np.degrees(
            np.arctan2(shoulder[:, 0] - hip[:, 0], hip[:, 1] - shoulder[:, 1])
        )
        feature = [
            float(np.nanpercentile(x, q))
            for x in [ratio, tilt]
            for q in [10, 50, 90]
        ]
    results.append(
        dict(
            item="hinge_and_trunk_hold",
            eligible=feature is not None,
            feature=feature,
            rule_grade=None,
            measurement=None,
            higher_is_better=True,
            proxy_only=True,
        )
    )
    return results


def phase_matches(
    window: dict | None, frames: np.ndarray, interval: list[int]
) -> bool:
    """자동 단계의 중심·과반 표본이 직접 표시한 시도 구간에 있는지 검사한다.

    Args:
        window: 자동 인식기의 시작·끝·중심 인덱스 또는 None.
        frames: 추출 표본의 원본 프레임 번호.
        interval: 직접 표시한 해당 단계 시도의 포함 시작·끝 프레임.

    Returns:
        bool: 중심과 과반 표본이 구간 안이면 True. 직접 시점을 자동 단계
        선택에 넘기지 않으며 불일치 라벨·입력의 결합만 차단한다.

    Raises:
        ValueError: 자동 단계 인덱스 또는 직접 구간이 유효하지 않은 경우.
    """
    if window is None:
        return False
    a, b = (window["start_index"], window["end_index"])
    anchor = window.get("anchor_index", (a + b) // 2)
    first, last = interval
    if not 0 <= a <= anchor <= b < len(frames) or first > last:
        raise ValueError("Invalid stage/frame contract")
    chosen = frames[a : b + 1]
    return bool(
        first <= frames[anchor] <= last
        and ((chosen >= first) & (chosen <= last)).mean() >= 0.5
    )


def refined_w(sequence: np.ndarray) -> dict | None:
    """점수·제공자 정보 없이 자동 W 시점과 실제 관측 특징을 구한다.

    Args:
        sequence: N×33×2 관측 픽셀 좌표.

    Returns:
        dict | None: 자동 시점·7개 화면 기하·명시 형태 규칙 또는 관측 불가.
        기존 개발 오판을 계기로 설계한 별도 후보이며 독립 검증은 아니다.

    Raises:
        ValueError: 좌표·표본·관측 형상 계약 위반.
    """
    detected = recovery_w_phase.detect(sequence)
    window = detected["phases"].get("W_attempt")
    if not window:
        return None
    result = w_morphology.evaluate(
        sequence, window["start_index"], window["end_index"]
    )
    if result["grade"] is None:
        return None
    feature = [
        result["arms"][side][key]
        for side in ("left", "right")
        for key in (
            "projected_bend_deg",
            "hand_above_elbow_torso_ratio",
            "elbow_up_upper_arm_ratio",
        )
    ] + [result["bilateral_bend_difference_deg"]]
    return dict(
        feature=feature,
        window=window,
        rule_grade=result["grade"],
        w_refined=detected.get("w_refined", False),
    )


def observed_absence(sequence: np.ndarray) -> dict | None:
    """점수 없이 관측된 T→팔 내림과 중간 W 근거의 부재를 확인한다.

    Args:
        sequence: 10Hz N×33×2 원본 픽셀 좌표. 누락은 NaN이다.

    Returns:
        dict | None: T·팔 내림 원본 표본 인덱스와 실제 W 형태 측정 또는
        근거 부족. 후반 관측 80%·6표본·팔 내림 3표본 이상을 요구한다.
        W 손 높이·팔 굽힘이 3표본 연속 보이면 후보를 만들지 않는다.

    Raises:
        ValueError: 좌표·빈 입력 또는 기하 형상 계약 위반.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or (not len(seq)):
        raise ValueError("Expected observed N x 33 x 2")
    if len(seq) < 12:
        return None
    required = [11, 12, 13, 14, 15, 16, 23, 24]
    valid = np.isfinite(seq[:, required]).all((1, 2))
    shoulder = (seq[:, 11] + seq[:, 12]) / 2
    hip = (seq[:, 23] + seq[:, 24]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    up = np.divide(
        shoulder - hip,
        torso[:, None],
        out=np.full_like(shoulder, np.nan),
        where=torso[:, None] > 1,
    )
    lateral = np.stack([-up[:, 1], up[:, 0]], axis=1)
    wrist = seq[:, [15, 16]] - shoulder[:, None]
    width = np.mean(abs(np.sum(wrist * lateral[:, None], axis=2)), axis=1)
    width = np.where(valid, width, np.nan)
    smooth = np.full(len(seq), np.nan)
    for i in range(1, len(seq) - 1):
        if valid[i - 1 : i + 2].all():
            smooth[i] = np.median(width[i - 1 : i + 2])
    if np.isfinite(smooth).sum() < 9:
        return None
    peak = float(np.nanpercentile(smooth, 90))
    if peak < 1:
        return None
    wide = ytw_phases._runs(valid & (smooth >= peak * 0.85), 3)
    if not wide:
        return None
    a, b = max(wide, key=lambda r: (r[1] - r[0], -r[0]))
    post = np.arange(len(seq)) > b
    if post.sum() < 6 or valid[post].mean() < 0.8:
        return None
    height = []
    bend = []
    for s, e, w in ((11, 13, 15), (12, 14, 16)):
        height.append(
            np.sum((seq[:, w] - seq[:, e]) * up, axis=1) / np.maximum(torso, 1)
        )
        bend.append(180 - geometry.angle_deg(seq[:, s], seq[:, e], seq[:, w]))
    w_evidence = (
        post
        & valid
        & (np.minimum(*height) >= -0.05)
        & (np.minimum(*bend) >= 15)
    )
    if ytw_phases._runs(w_evidence, 3):
        return None
    hand_down = np.mean(
        np.sum(wrist * -up[:, None], axis=2), axis=1
    ) / np.maximum(torso, 1)
    low = (
        post
        & valid
        & (smooth <= peak * 0.55)
        & (hand_down >= 0.6)
        & (np.maximum(*height) < -0.1)
    )
    runs = ytw_phases._runs(low, 3)
    if not runs:
        return None
    start, end = runs[0]
    if start - b > 10:
        return None
    morphology = w_morphology.evaluate(seq, start, end)
    if morphology["grade"] is None:
        return None
    feature = [
        morphology["arms"][s][k]
        for s in ("left", "right")
        for k in (
            "projected_bend_deg",
            "hand_above_elbow_torso_ratio",
            "elbow_up_upper_arm_ratio",
        )
    ] + [morphology["bilateral_bend_difference_deg"]]
    return dict(
        window=dict(
            start_index=start, end_index=end, anchor_index=(start + end) // 2
        ),
        T_window=[a, b],
        feature=feature,
        measured_shape_grade=morphology["grade"],
        post_observed_fraction=float(valid[post].mean()),
        actual_contact_not_inferred=True,
    )
