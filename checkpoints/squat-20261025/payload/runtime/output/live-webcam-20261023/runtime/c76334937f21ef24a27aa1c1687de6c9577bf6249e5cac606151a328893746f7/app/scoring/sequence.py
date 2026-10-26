"""관절 배열의 좌표계·시간·가시성 계약을 서비스 외부 입력과 분리한다."""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class PoseSequence:
    """한 구간의 10Hz 픽셀 관절과 원본 프레임 시각을 소유한다.

    배열은 생성 시 복사하며 누락은 NaN이다. 3D 좌표를 받지 않는다.
    같은 객체의 배열을 호출자가 수정하거나 여러 스레드에서 쓰지 않는다.

    Attributes:
        image_xy: N×33×2 원본 영상 픽셀 좌표.
        frame_numbers: 엄격히 증가하는 원본 프레임 번호 N개.
        fps: 원본 영상 초당 프레임 수.
        width: 원본 가로 픽셀 수.
        height: 원본 세로 픽셀 수.
    """

    image_xy: np.ndarray
    frame_numbers: np.ndarray
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        """좌표·시각·크기를 검증하며 불일치는 ValueError로 반환한다."""
        xy = np.asarray(self.image_xy, dtype=float)
        frames = np.asarray(self.frame_numbers)
        if (
            xy.ndim != 3
            or xy.shape[1:] != (33, 2)
            or not len(xy)
            or np.isinf(xy).any()
            or frames.shape != (len(xy),)
            or not np.isfinite(frames).all()
            or not np.equal(frames, np.floor(frames)).all()
            or (frames < 0).any()
            or (np.diff(frames) <= 0).any()
            or not np.isfinite(self.fps)
            or self.fps < 10
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError(
                "Expected pixel N x 33 x 2, increasing frames, "
                "fps >= 10 and positive dimensions"
            )
        if (
            len(frames) > 1
            and np.max(abs(np.diff(frames) / self.fps - 0.1))
            > 1 / self.fps + 1e-6
        ):
            raise ValueError("Model requires approximately 10 Hz observations")
        xy = xy.copy()
        xy[~np.isfinite(xy).all(2)] = np.nan
        object.__setattr__(self, "image_xy", xy)
        object.__setattr__(
            self, "frame_numbers", frames.astype(np.int64).copy()
        )

    @property
    def duration_seconds(self) -> float:
        """실제로 관측한 첫·끝 표본 간 시간(초)을 반환한다."""
        return float(
            (self.frame_numbers[-1] - self.frame_numbers[0]) / self.fps
        )


def verify_pair(
    side: PoseSequence, front: PoseSequence, lag_frames: int
) -> None:
    """직접 확인한 두 구도의 원본 프레임 시차와 표본 정렬을 검사한다.

    Args:
        side: 측면 10Hz 시계열.
        front: 동일 수행 정면 시계열.
        lag_frames: 측면 프레임에 더할 정면 시차, 원본 프레임 단위.

    Raises:
        ValueError: FPS·길이 또는 시차가 1프레임 허용 오차를 넘은 경우.
    """
    if (
        side.fps != front.fps
        or len(side.frame_numbers) != len(front.frame_numbers)
        or np.max(abs(front.frame_numbers - side.frame_numbers - lag_frames))
        > 1
    ):
        raise ValueError(
            "Two views require equal fps/count and reviewed frame offset"
        )
