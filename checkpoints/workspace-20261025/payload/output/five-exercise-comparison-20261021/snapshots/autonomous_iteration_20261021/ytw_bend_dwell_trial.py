"""T 뒤 팔 굽힘 증가·폭 감소·유지로 W 시도를 찾는 별도 개발 실험이다.

등급·직접 구간·제공자 문자열은 입력으로 쓰지 않는다. Y를 관측했다고
추정하지 않으며 구간 식별은 자세 충족 또는 총점 판정을 뜻하지 않는다.
"""

import numpy as np
import geometry
import ytw_phases

VERSION = "ytw-bend-rise-dwell-trial-v7"


def detect(sequence: np.ndarray, sample_hz: float = 10) -> dict:
    """기존 완전 구간을 보존하고 T 후 안정된 굽힘 시도를 보완한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.
        sample_hz: 양수 표본 빈도 Hz.

    Returns:
        dict: 표본 인덱스 구간·공학 경계·보완 경로 표시. 추가 경로는
        T와 W만 반환하고 Y·완전 시퀀스를 생성하지 않는다. 굽힘 변화가
        없는 생략·휴식은 W로 강제 점수화하지 않는다.

    Raises:
        ValueError: 좌표 형상·표본 빈도 오류.
    """
    result = ytw_phases.detect(sequence, sample_hz)
    result["base_version"] = result["version"]
    result["version"] = VERSION
    if result["complete_attempt_sequence"]:
        result["bend_fallback_used"] = False
        return result
    seq = np.asarray(sequence, dtype=float)
    minimum = max(3, int(np.ceil(0.2 * sample_hz)) + 1)
    valid = np.isfinite(seq[:, [11, 12, 13, 14, 15, 16, 23, 24]]).all((1, 2))
    if valid.sum() < minimum * 2:
        return result
    sh = (seq[:, 11] + seq[:, 12]) / 2
    down = (seq[:, 23] + seq[:, 24]) / 2 - sh
    length = np.linalg.norm(down, axis=1)
    lateral = np.divide(
        np.stack([-down[:, 1], down[:, 0]], axis=1),
        length[:, None],
        out=np.full_like(down, np.nan),
        where=length[:, None] > 1,
    )
    width = np.mean(
        np.abs(
            np.sum((seq[:, [15, 16]] - sh[:, None]) * lateral[:, None], axis=2)
        ),
        axis=1,
    )
    bend = (
        360
        - geometry.angle_deg(seq[:, 11], seq[:, 13], seq[:, 15])
        - geometry.angle_deg(seq[:, 12], seq[:, 14], seq[:, 16])
    ) / 2
    smooth_width = np.full(len(seq), np.nan)
    smooth_bend = np.full(len(seq), np.nan)
    for i in range(1, len(seq) - 1):
        if valid[i - 1 : i + 2].all():
            smooth_width[i] = np.median(width[i - 1 : i + 2])
            smooth_bend[i] = np.median(bend[i - 1 : i + 2])
    observed = valid & np.isfinite(smooth_width) & np.isfinite(smooth_bend)
    if observed.sum() < minimum * 2:
        return result
    peak = float(np.percentile(smooth_width[observed], 90))
    if peak < 1 or peak - np.nanmin(smooth_width) < 0.35 * np.nanmedian(length):
        return result
    wide = [
        (a, b)
        for a, b in ytw_phases._runs(
            observed & (smooth_width > peak * 0.7), minimum
        )
        if (smooth_width[a : b + 1] >= peak * 0.85).sum() >= minimum
    ]
    for a, b in wide:
        reference = float(np.median(smooth_bend[a : b + 1]))
        if reference > 35:
            continue
        stable = np.zeros(len(seq), dtype=bool)
        for i in range(b + 1, min(len(seq) - 1, b + int(1.5 * sample_hz) + 1)):
            low, high = max(0, i - 1), min(len(seq), i + 2)
            stable[i] = bool(
                observed[low:high].all()
                and smooth_width[i] <= peak * 0.7
                and smooth_bend[i] >= reference + 10
                and np.ptp(smooth_width[low:high]) <= peak * 0.15
                and np.ptp(smooth_bend[low:high]) <= 16
            )
        runs = ytw_phases._runs(stable, minimum)
        if not runs:
            continue
        start, end = runs[0]
        if start - b > sample_hz:
            continue
        result["phases"] = {}
        for name, (lo, hi) in zip(
            ("T_attempt", "W_attempt"), ((a, b), (start, end)), strict=True
        ):
            result["phases"][name] = dict(
                start_index=lo,
                end_index=hi,
                anchor_index=(lo + hi) // 2,
                samples=hi - lo + 1,
            )
        result.update(
            bend_fallback_used=True,
            complete_attempt_sequence=False,
            hold_reason="Y_unconfirmed_bend_dwell_TW_only",
            thresholds=dict(
                T_bend_max_deg=35,
                W_bend_rise_min_deg=10,
                width_exit_ratio=0.7,
                width_dwell_range_max_relative_peak=0.15,
                bend_three_sample_range_max_deg=16,
                search_after_T_seconds=1.5,
                min_samples=minimum,
            ),
        )
        return result
    result["bend_fallback_used"] = False
    return result
