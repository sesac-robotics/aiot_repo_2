"""정면 투영의 팔 폭 변화와 유지 구간으로 Y·T·W 시도를 분리한다.

시도 순서와 정상 모양은 별개다. 정면만으로 Y의 전방 깊이를 확정하지
않으며 직접 표시 구간·점수 라벨·제공자 문자열을 입력으로 받지 않는다.
"""

import numpy as np

VERSION = "ytw-width-low-arm-trial-v4-development-only"


def _runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """참인 연속 구간을 양 끝 포함 인덱스로 반환한다."""
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [
        (int(a), int(b))
        for a, b in zip(starts, ends, strict=True)
        if b - a + 1 >= minimum
    ]


def detect(sequence: np.ndarray, sample_hz: float = 10) -> dict:
    """관절 시계열에서 유지되는 좁음→넓음→좁음 팔 시도 구간을 찾는다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        sample_hz: 표본 빈도 Hz. 현재 공통 추출 계약은 10Hz다.

    Returns:
        dict: phases별 시작·끝·대표 인덱스 또는 보류 사유와 측정값.
        Y·W는 순서상의 시도이며 올바른 3차원 모양 확인을 뜻하지 않는다.

    Raises:
        ValueError: 좌표 형상·표본 빈도가 잘못된 경우.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or sample_hz <= 0:
        raise ValueError("Expected N x 33 x 2 coordinates and positive Hz")
    result = dict(
        version=VERSION,
        phases={},
        complete_attempt_sequence=False,
        canonical_Y_depth_confirmed=False,
        sample_hz=sample_hz,
        hold_reason=None,
    )
    minimum = max(3, int(np.ceil(0.2 * sample_hz)) + 1)
    if len(seq) < minimum * 3:
        result["hold_reason"] = "insufficient_temporal_samples"
        return result
    sh = (seq[:, 11] + seq[:, 12]) / 2
    hip = (seq[:, 23] + seq[:, 24]) / 2
    down = hip - sh
    length = np.linalg.norm(down, axis=1)
    length = np.where(length > 1, length, np.nan)
    vertical = down / length[:, None]
    lateral = np.stack([-vertical[:, 1], vertical[:, 0]], axis=1)
    wrist = seq[:, [15, 16]] - sh[:, None, :]
    width = (
        np.mean(np.abs(np.sum(wrist * lateral[:, None, :], axis=2)), axis=1)
        / length
    )
    height = (
        np.mean(np.sum(wrist * vertical[:, None, :], axis=2), axis=1) / length
    )
    valid = np.isfinite(seq[:, [11, 12, 13, 14, 15, 16, 23, 24]]).all((1, 2))
    # 완전한 이웃 세 표본만 평활화하며 누락을 보간하지 않는다.
    smoothed = np.full(len(seq), np.nan)
    for i in range(1, len(seq) - 1):
        if valid[i - 1 : i + 2].all():
            smoothed[i] = np.median(width[i - 1 : i + 2])
    active = valid & np.isfinite(smoothed) & (height < 1.1)
    if active.sum() < minimum * 3:
        result["hold_reason"] = "insufficient_observed_raised_arm_samples"
        return result
    peak = float(np.percentile(smoothed[active], 90))
    if peak < 0.5 or np.nanmax(smoothed) - np.nanmin(smoothed) < 0.35:
        result["hold_reason"] = "insufficient_lateral_span_change"
        return result
    # 진입 85%, 유지 70%의 서로 다른 경계로 T 내부의 작은 폭 변화를
    # 별도 W로 잘라 내지 않는다. 미관측·팔 내림 구간은 연결하지 않는다.
    wide = [
        (a, b)
        for a, b in _runs(active & (smoothed > peak * 0.70), minimum)
        if (smoothed[a : b + 1] >= peak * 0.85).sum() >= minimum
    ]
    candidates = []
    for a, b in wide:
        before = _runs(
            active & (np.arange(len(seq)) < a) & (smoothed <= peak * 0.70),
            minimum,
        )
        after = _runs(
            active & (np.arange(len(seq)) > b) & (smoothed <= peak * 0.70),
            minimum,
        )
        if before and after:
            y, w = before[-1], after[0]
            if a - y[1] <= sample_hz and w[0] - b <= sample_hz:
                candidates.append((b - a + 1, y, (a, b), w))
    if not candidates:
        result["hold_reason"] = "ordered_three_dwell_windows_not_observed"
        return result
    _, y, t, w = max(candidates, key=lambda row: row[0])
    for name, (a, b) in zip(
        ("Y_attempt", "T_attempt", "W_attempt"), (y, t, w), strict=True
    ):
        result["phases"][name] = dict(
            start_index=a,
            end_index=b,
            anchor_index=(a + b) // 2,
            samples=b - a + 1,
            median_lateral_span=float(np.median(smoothed[a : b + 1])),
        )
    result.update(
        complete_attempt_sequence=True,
        lateral_span_p90=peak,
        thresholds=dict(
            active_hand_height_torso_ratio=1.1,
            wide_enter_relative_span=0.85,
            wide_exit_relative_span=0.70,
            min_samples=minimum,
        ),
    )
    return result
