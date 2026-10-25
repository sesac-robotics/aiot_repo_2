"""MediaPipe 관절 어댑터와 클립 특징을 제공한다.

입력 영상에서 균등한 12시점을 사용한다. 완료 클립 평가용이며 실시간
반복 판정이 아니다. 추정 world 좌표는 미터 단위이고 특징은 몸통 길이로
정규화한다. 누락 관절은 NaN으로 유지한다.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def extract_clip(detector: object, path: Path) -> tuple:
    """한 클립에서 관절 시퀀스·특징·유효 비율을 추출한다.

    Args:
        detector: IMAGE 모드 PoseLandmarker. 모델별로 재사용한다.
        path: 디코딩할 영상의 절대 경로.
    Returns:
        특징 벡터 또는 None, 12개 관절 좌표, 프레임별 관측 비율.
        몸통 관절 관측이 부족하면 특징은 None이다.
    Raises:
        ValueError: 영상이 열리지 않거나 길이가 잘못된 경우.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or count < 12:
            raise ValueError(f"Invalid video: {path.name}")
        targets = set(np.linspace(0, count - 1, 12).astype(int).tolist())
        coords, coverages = [], []
        for index in range(count):
            if not cap.grab():
                break
            if index not in targets:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
            )
            world = np.full((33, 3), np.nan, dtype=np.float32)
            if result.pose_landmarks:
                local = result.pose_landmarks[0]
                points = result.pose_world_landmarks[0]
                raw = np.array([[p.x, p.y, p.z] for p in points])
                for k, point in enumerate(points):
                    if local[k].visibility >= 0.3 and local[k].presence >= 0.5:
                        world[k] = [point.x, point.y, point.z]
            else:
                raw = np.full((33, 3), np.nan)
            # 중심·크기에는 추정 좌표를 쓰되 관측 안 된 특징은 보충하지 않는다.
            torso = raw[[11, 12, 23, 24]]
            if np.isfinite(torso).all():
                hip = (raw[23] + raw[24]) / 2
                length = (
                    np.linalg.norm(raw[11] - raw[23])
                    + np.linalg.norm(raw[12] - raw[24])
                ) / 2
                if length > 0.01:
                    world = (world - hip) / length
                else:
                    world[:] = np.nan
            else:
                world[:] = np.nan
            coords.append(world[BODY])
            coverages.append(float(np.isfinite(world[BODY]).mean()))
        sequence = np.asarray(coords, dtype=np.float32)
        if (
            len(sequence) != 12
            or np.mean(coverages) < 0.35
            or sum(c >= 0.35 for c in coverages) < 6
        ):
            return None, sequence, coverages
        # 요약 통계와 함께 시간 순서를 보존한다.
        flat = sequence.reshape(12, -1)
        med = np.array(
            [
                np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
                for c in flat.T
            ]
        )
        spread = np.array(
            [
                np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
                for c in flat.T
            ]
        )
        features = np.concatenate([flat.ravel(), med, spread])
        return features, sequence, coverages
    finally:
        cap.release()


def make_detector(model: str) -> object:
    """지정 GPU 모델을 생성하며 호출자가 close로 자원을 해제한다.

    Args:
        model: heavy, full, lite 중 하나.
    Returns:
        IMAGE 모드 PoseLandmarker. GPU 실패를 CPU로 숨기지 않는다.
    Raises:
        ValueError: 지원하지 않는 모델 이름인 경우.
    """
    if model not in ("heavy", "full", "lite"):
        raise ValueError(model)
    path = Path(__file__).resolve().parents[1] / "models" / f"{model}.task"
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(path),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)
