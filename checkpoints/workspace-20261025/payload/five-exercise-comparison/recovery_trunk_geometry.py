"""측면 푸시업의 골반 처짐·들림 방향을 같은 관절에서 보존한다."""

import numpy as np


def signed_offsets(sequence: np.ndarray, side: str) -> np.ndarray:
    """어깨–발목 직선에 대한 골반의 수직 편차를 몸통 길이로 정규화한다.

    Args:
        sequence: N×33×2 픽셀 관절. 관측 누락은 NaN이다.
        side: left 또는 right. 기존 몸통 항목과 동일한 측면을 사용한다.

    Returns:
        np.ndarray: N개 무차원 편차. 영상 아래쪽 처짐은 양수, 위쪽 들림은
        음수다. 좌우 반전·평행 이동·균일 확대에 불변이다. 관절 누락 또는
        길이가 1픽셀 미만인 퇴화 표본은 NaN이며 보간하지 않는다.

    Raises:
        ValueError: 배열 형상 또는 측면 이름이 잘못된 경우.
    """
    if (
        sequence.ndim != 3
        or sequence.shape[1:] != (33, 2)
        or side not in ("left", "right")
    ):
        raise ValueError("Expected N x 33 x 2 and left/right side")
    offset = int(side == "right")
    shoulder, hip, ankle = (sequence[:, j + offset] for j in (11, 23, 27))
    axis = ankle - shoulder
    length = np.linalg.norm(axis, axis=1)
    torso = np.linalg.norm(hip - shoulder, axis=1)
    normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1)
    normal *= np.where(normal[:, 1] < 0, -1, 1)[:, None]
    valid = (
        np.isfinite(np.stack((shoulder, hip, ankle))).all((0, 2))
        & (length >= 1)
        & (torso >= 1)
        & (abs(axis[:, 0]) >= 1)
    )
    result = np.full(len(sequence), np.nan)
    result[valid] = (
        np.sum((hip - shoulder)[valid] * normal[valid], axis=1)
        / length[valid]
        / torso[valid]
    )
    return result
