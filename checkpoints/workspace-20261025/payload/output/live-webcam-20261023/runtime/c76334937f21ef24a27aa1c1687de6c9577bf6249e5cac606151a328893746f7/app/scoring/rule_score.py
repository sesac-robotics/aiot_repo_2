"""관측된 유한 기하 값의 백분위와 명시 등급만 계산한다."""

import numpy as np


def _stat(channel: np.ndarray, percentile: float) -> float | None:
    good = channel[np.isfinite(channel)]
    if len(channel) < 5 or len(good) / len(channel) < 0.8:
        return None
    return float(np.percentile(good, percentile))


def _grade(
    value: float | None, good: float, partial: float, higher: bool = True
) -> int | None:
    if value is None:
        return None
    if higher:
        return 2 if value >= good else 1 if value >= partial else 0
    return 2 if value <= good else 1 if value <= partial else 0
