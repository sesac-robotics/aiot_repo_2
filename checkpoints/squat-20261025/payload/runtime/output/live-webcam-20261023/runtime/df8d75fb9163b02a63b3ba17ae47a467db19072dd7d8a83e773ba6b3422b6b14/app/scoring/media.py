"""OpenCV·MediaPipe Heavy와 픽셀 시계열 사이의 파일 기반 Adapter다."""

import hashlib
import pathlib
import time

import numpy as np

from app.scoring.sequence import PoseSequence

HEAVY_SHA256 = (
    "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
)


def extract_video(
    path: pathlib.Path,
    model_path: pathlib.Path,
    *,
    start_frame: int,
    end_frame: int,
    delegate: str = "CPU",
) -> tuple[PoseSequence, dict]:
    """명시 구간을 10Hz로 순차 디코딩하고 Heavy 관절을 추출한다.

    Args:
        path: 로컬 원본 영상. 다운로드·카메라 연결을 하지 않는다.
        model_path: 지정 SHA-256의 Heavy task 모델 경로.
        start_frame: 포함 시작 원본 프레임, 0 이상.
        end_frame: 포함 끝 원본 프레임. 최대 120초 구간만 허용한다.
        delegate: CPU 또는 GPU. 실패 시 다른 장치로 조용히 대체하지 않는다.

    Returns:
        tuple: visibility·presence 0.5 이상 관절만 남긴 픽셀 시계열과
        추출 초·모델 해시. 미검출 관절은 NaN이고 점수는 생성하지 않는다.

    Raises:
        ValueError: 모델·장치·영상 메타데이터·프레임 범위 불일치.
        OSError: 영상 또는 모델 접근 실패.
        RuntimeError: MediaPipe 실행·디코딩 실패.
    """
    import cv2
    import mediapipe as mp

    if delegate not in ("CPU", "GPU"):
        raise ValueError("Delegate must be CPU or GPU")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != HEAVY_SHA256:
        raise ValueError("Exact MediaPipe Heavy model required")
    cap = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    start = time.perf_counter()
    try:
        if not cap.isOpened():
            raise OSError("Video could not be opened")
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        if (
            not np.isfinite(fps)
            or fps < 10
            or type(start_frame) is not int
            or type(end_frame) is not int
            or not 0 <= start_frame < end_frame < count
            or (end_frame - start_frame) / fps > 120
        ):
            raise ValueError(
                "Valid <=120 second frame interval and fps >= 10 required"
            )
        width, height = int(cap.get(3)), int(cap.get(4))
        indices = np.unique(
            np.rint(np.arange(start_frame, end_frame + 0.01, fps / 10))
        ).astype(int)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=getattr(mp.tasks.BaseOptions.Delegate, delegate),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        sequence = []
        cap.set(1, start_frame)
        decoded = start_frame - 1
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as pose:
            for index in indices:
                while decoded < index:
                    if not cap.grab():
                        raise RuntimeError("Sequential frame decode failed")
                    decoded += 1
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError("Selected frame unavailable")
                scale = min(1.0, 640 / max(width, height))
                resized = cv2.resize(
                    frame, (round(width * scale), round(height * scale))
                )
                result = pose.detect_for_video(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(
                            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        ),
                    ),
                    round((int(index) - start_frame) * 1000 / fps),
                )
                pixels = np.full((33, 2), np.nan, dtype=np.float32)
                if result.pose_landmarks:
                    for j, landmark in enumerate(result.pose_landmarks[0]):
                        if (
                            landmark.visibility >= 0.5
                            and landmark.presence >= 0.5
                            and 0 <= landmark.x <= 1
                            and 0 <= landmark.y <= 1
                        ):
                            pixels[j] = landmark.x * width, landmark.y * height
                sequence.append(pixels)
        return PoseSequence(
            np.asarray(sequence), indices, fps, width, height
        ), dict(
            extraction_seconds=time.perf_counter() - start,
            delegate=delegate,
            heavy_sha256=HEAVY_SHA256,
            score_generated=False,
        )
    finally:
        cap.release()


def load_landmarks(path: pathlib.Path) -> PoseSequence:
    """수치형 NPZ를 읽고 좌표·시각 계약을 검사한다.

    Args:
        path: image_xy·frame_numbers·fps·width·height가 있는 로컬 NPZ.

    Returns:
        PoseSequence: 검증된 10Hz 원본 픽셀 관절 사본.

    Raises:
        OSError: 파일 접근 실패.
        ValueError: 필드·형상·시간·좌표 계약 불일치.
    """
    with np.load(path, allow_pickle=False) as data:
        return PoseSequence(
            data["image_xy"],
            data["frame_numbers"],
            float(data["fps"]),
            int(data["width"]),
            int(data["height"]),
        )


def save_landmarks(path: pathlib.Path, seq: PoseSequence) -> None:
    """시계열을 수치형 NPZ로 저장한다. 실패 시 OSError를 전달한다.

    Args:
        path: 출력 파일. 기존 파일을 덮어쓸 수 있으므로 호출자가 결정한다.
        seq: 검증된 픽셀·원본 프레임·fps·영상 크기.
    """
    np.savez_compressed(
        path,
        image_xy=seq.image_xy,
        frame_numbers=seq.frame_numbers,
        fps=seq.fps,
        width=seq.width,
        height=seq.height,
    )
