"""T 다음 복귀와 실제 팔 굽힘이 분리된 경우 W 후보 시점을 보완한다."""

import numpy as np
import geometry
import ytw_partial_trial


def detect(sequence: np.ndarray, sample_hz: float = 10) -> dict:
    """Y 누락 보완 경로에서 T 뒤 굽힘이 가장 뚜렷한 짧은 유지 창을 찾는다.

    Args:
        sequence: N×33×2 픽셀 관절. 누락은 NaN이며 보간하지 않는다.
        sample_hz: 표본 빈도 Hz. 기본 10Hz.

    Returns:
        dict: 기존 단계와 W 보완 여부. 완전 YTW 경로는 변경하지 않는다.
        T 종료 뒤 2초 안에서 양팔 굽힘 중앙값이 15도 이상인 3표본 창을
        찾는다. 점수·정상 모양·직접 단계 라벨을 입력하지 않는다. 근거가
        없으면 기존 시점을 유지하고 새 Y나 전체 시도를 만들지 않는다.

    Raises:
        ValueError: 기존 단계 인식기의 형상·표본 빈도 계약 위반.
    """
    result = ytw_partial_trial.detect(sequence, sample_hz)
    result["w_refined"] = False
    if (
        not result.get("partial_fallback_used")
        or "T_attempt" not in result["phases"]
    ):
        return result
    seq = np.asarray(sequence)
    first = result["phases"]["T_attempt"]["end_index"] + 1
    last = min(len(seq), first + int(round(2 * sample_hz)))
    bend = np.minimum(
        180 - geometry.angle_deg(seq[:, 11], seq[:, 13], seq[:, 15]),
        180 - geometry.angle_deg(seq[:, 12], seq[:, 14], seq[:, 16]),
    )
    shoulder = (seq[:, 11] + seq[:, 12]) / 2
    down = (seq[:, 23] + seq[:, 24]) / 2 - shoulder
    length = np.linalg.norm(down, axis=1)
    vertical = np.divide(
        down,
        length[:, None],
        out=np.full_like(down, np.nan),
        where=length[:, None] > 1,
    )
    lateral = np.stack((-vertical[:, 1], vertical[:, 0]), axis=1)
    wrist = seq[:, [15, 16]] - shoulder[:, None]
    width = np.mean(abs(np.sum(wrist * lateral[:, None], axis=2)), axis=1)
    t = result["phases"]["T_attempt"]
    peak = np.nanmax(width[t["start_index"] : t["end_index"] + 1])
    candidates = [
        (float(np.median(bend[a : a + 3])), a)
        for a in range(first, last - 2)
        if np.isfinite(seq[a : a + 3, [11, 12, 13, 14, 15, 16, 23, 24]]).all()
        and np.isfinite(peak)
        and peak > 1
        and np.nanmedian(width[a : a + 3]) <= peak * 0.70
    ]
    if not candidates:
        return result
    strength, start = max(candidates, key=lambda x: (x[0], -x[1]))
    if strength < 15:
        return result
    result["original_W_attempt"] = dict(result["phases"]["W_attempt"])
    result["phases"]["W_attempt"] = dict(
        start_index=start,
        end_index=start + 2,
        anchor_index=start + 1,
        samples=3,
    )
    result.update(
        w_refined=True,
        w_refinement="bilateral_bend_dwell_after_T_v1",
        complete_attempt_sequence=False,
    )
    return result
