"""3종의 관절 각도 시계열에서 시작·중간·복귀가 있는 동작을 찾는다.

정상 자세 점수 경계를 구간 탐지에 사용하지 않아 얕은 오류 수행도 보존한다.
관절을 복원하지 않으며 짧은 결측 보간은 단계 후보 검출에만 사용한다.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import geometry


def detect(exercise: str, sequence: np.ndarray, frames: np.ndarray) -> dict:
    """10Hz 관절 표본에서 변화량 8도 이상의 복귀 동작 후보를 반환한다.

    Args:
        exercise: pushups·squats·shoulder 중 운동 코드.
        sequence: N×33×2 픽셀 관절. 미관측은 NaN.
        frames: 각 관절 표본과 대응하는 엄격히 증가하는 원본 프레임 인덱스.

    Returns:
        dict: 반복 후보의 시작·중간·끝 인덱스·원본 프레임과 실패 사유.
        검출은 수행 완료 후보이며 정상 수행 판정이나 점수 라벨이 아니다.

    Raises:
        ValueError: 운동·좌표·프레임 계약 불일치.
    """
    if exercise not in ("pushups", "squats", "shoulder"):
        raise ValueError("Unsupported active exercise")
    if len(sequence) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("Strictly increasing aligned frame indices required")
    values = geometry.channels(sequence)
    channel = {
        "pushups": "elbow_deg",
        "squats": "knee_deg",
        "shoulder": "elevation_deg",
    }[exercise]
    left, right = (values[side + "_" + channel] for side in ("left", "right"))
    signal = max((left, right), key=lambda a: np.isfinite(a).sum()).copy()
    valid = np.isfinite(signal)
    if len(signal) < 7 or valid.mean() < 0.8:
        return dict(repetitions=[], hold="insufficient_joint_observation")
    missing = np.flatnonzero(~valid)
    if len(missing):
        runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
        if any(len(run) > 2 for run in runs):
            return dict(repetitions=[], hold="long_phase_gap")
        signal = np.interp(
            np.arange(len(signal)), np.flatnonzero(valid), signal[valid]
        )
    signal = median_filter(signal, size=3, mode="nearest")
    # 푸시업·스쿼트는 굽힘 각도 최저점, 숄더 프레스는 팔 높이 최고점이다.
    signal = signal if exercise == "shoulder" else -signal
    peaks, properties = find_peaks(signal, prominence=8, distance=5)
    troughs, _ = find_peaks(-signal, prominence=4, distance=3)
    repetitions = []
    for peak, prominence in zip(peaks, properties["prominences"], strict=True):
        before = troughs[troughs < peak]
        after = troughs[troughs > peak]
        a = int(before[-1]) if len(before) else 0
        b = int(after[0]) if len(after) else len(signal) - 1
        if peak - a < 2 or b - peak < 2:
            continue
        excursion = min(signal[peak] - signal[a], signal[peak] - signal[b])
        if excursion < 8 or abs(signal[a] - signal[b]) > max(
            12, excursion * 0.4
        ):
            continue
        repetitions.append(
            dict(
                start_index=int(a),
                middle_index=int(peak),
                end_index=int(b),
                original_frames=[int(frames[i]) for i in (a, peak, b)],
                angular_excursion_deg=float(excursion),
                peak_prominence_deg=float(prominence),
            )
        )
    return dict(
        repetitions=repetitions,
        hold=None if repetitions else "no_complete_return_motion",
        phase_interpolation_samples=len(missing),
        scoring_landmarks_imputed=False,
    )
