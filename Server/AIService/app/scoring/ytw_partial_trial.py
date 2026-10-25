"""Y 누락 시 관측된 T 뒤 W 시도만 분리하는 개발 실험이다.

Y·전경 깊이·총점을 추정하지 않는다. 규칙 점수·라벨을 입력으로 받지 않는다.
"""

import numpy as np

from app.scoring import ytw_phases

VERSION = "ytw-observed-TW-partial-trial-v6"


def detect(sequence: np.ndarray, sample_hz: float = 10) -> dict:
    """기존 세 단계가 없을 때 관측된 T→좁은 팔 시도만 반환한다.

    Args:
        sequence: N×33×2 픽셀 관절. 미관측은 NaN이다.
        sample_hz: 양수 표본 빈도 Hz.

    Returns:
        dict: 단계 구간과 후보 선택 근거. 보완 경로는 Y를 만들지 않고
        complete_attempt_sequence=False를 유지한다. W 시도는 정상 모양을
        의미하지 않으며 총점 판정 허가가 아니다.

    Raises:
        ValueError: 좌표 형상·빈도 오류.
    """
    result = ytw_phases.detect(sequence, sample_hz)
    result["base_version"] = result["version"]
    result["version"] = VERSION
    if result["complete_attempt_sequence"]:
        result["partial_fallback_used"] = False
        return result
    seq = np.asarray(sequence, dtype=float)
    minimum = max(3, int(np.ceil(0.2 * sample_hz)) + 1)
    if len(seq) < minimum * 2:
        return result
    sh = (seq[:, 11] + seq[:, 12]) / 2
    down = (seq[:, 23] + seq[:, 24]) / 2 - sh
    length = np.linalg.norm(down, axis=1)
    vertical = np.divide(
        down,
        length[:, None],
        out=np.full_like(down, np.nan),
        where=length[:, None] > 1,
    )
    lateral = np.stack([-vertical[:, 1], vertical[:, 0]], axis=1)
    wr = seq[:, [15, 16]] - sh[:, None]
    width = np.mean(np.abs(np.sum(wr * lateral[:, None], axis=2)), axis=1)
    height = np.mean(np.sum(wr * vertical[:, None], axis=2), axis=1)
    arms = np.mean(
        np.linalg.norm(seq[:, [13, 14]] - seq[:, [11, 12]], axis=2)
        + np.linalg.norm(seq[:, [15, 16]] - seq[:, [13, 14]], axis=2),
        axis=1,
    )
    ratio = np.divide(
        height, arms, out=np.full(len(seq), np.nan), where=arms > 1
    )
    valid = np.isfinite(seq[:, [11, 12, 13, 14, 15, 16, 23, 24]]).all((1, 2))
    smooth = np.full(len(seq), np.nan)
    for i in range(1, len(seq) - 1):
        if valid[i - 1 : i + 2].all():
            smooth[i] = np.median(width[i - 1 : i + 2])
    observed = valid & np.isfinite(smooth)
    if observed.sum() < minimum * 2:
        return result
    peak = float(np.percentile(smooth[observed], 90))
    if peak < 1 or peak - np.nanmin(smooth) < 0.35 * np.nanmedian(length):
        return result
    wide = [
        (a, b)
        for a, b in ytw_phases._runs(observed & (smooth > peak * 0.70), minimum)
        if (smooth[a : b + 1] >= peak * 0.85).sum() >= minimum
    ]
    for a, b in wide:
        after = ytw_phases._runs(
            observed
            & (np.arange(len(seq)) > b)
            & (smooth <= peak * 0.70)
            & (ratio < 0.9),
            minimum,
        )
        if not after or after[0][0] - b > sample_hz:
            continue
        result["phases"] = {}
        for name, (start, end) in zip(
            ("T_attempt", "W_attempt"), ((a, b), after[0]), strict=True
        ):
            result["phases"][name] = dict(
                start_index=start,
                end_index=end,
                anchor_index=(start + end) // 2,
                samples=end - start + 1,
            )
        result.update(
            partial_fallback_used=True,
            complete_attempt_sequence=False,
            hold_reason="Y_unconfirmed_observed_TW_only",
            W_attempt_available=True,
            thresholds=dict(
                wide_enter_relative_span=0.85,
                wide_exit_relative_span=0.70,
                hand_down_arm_length_ratio_max=0.9,
                min_samples=minimum,
            ),
        )
        return result
    result["partial_fallback_used"] = False
    return result
