"""정지 플랭크 개발 자료에 Heavy IMAGE GPU를 적용한다.

시간 간격을 알 수 없는 JPEG에 VIDEO 시각이나 FPS를 만들지 않는다. 기존
영상 캐시와 다른 스키마로 저장하며 제공자 관절·상태·직접 등급을 입력하지 않는다.
"""

import argparse
import time
import pathlib
import cv2
import mediapipe as mp
import numpy as np
import audit
from extract import MODEL
from resources import check_resources


def run(output_stem: str = "heavy_static") -> dict:
    """변경 없는 원본 이미지 56개의 관절·마스크·처리 시간을 저장한다.

    Args:
        output_stem: 캐시 이름. 최초 추출 또는 독립 재실행 보존용 이름.

    Returns:
        dict: IMAGE 전용 캐시·원본/모델 해시·이미지별 관측 및 처리 초.
        미관측 좌표는 NaN이며 실제 접촉·밴드 하중을 추정하지 않는다.

    Raises:
        OSError: 원본·모델 또는 출력 접근 실패.
        ValueError: 원본 해시·디코딩 오류.
        RuntimeError: Heavy 또는 자원 보호 오류.
    """
    if output_stem not in ("heavy_static", "heavy_static_repeat"):
        raise ValueError("Unsupported extraction output name")
    folder = audit.OUT / "plank_static_review"
    inputs = audit.read_json(folder / "inputs.json")
    records = inputs["records"]
    arrays, confidences, durations = [], [], []
    model_sha = audit.digest(MODEL)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=mp.tasks.BaseOptions.Delegate.GPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    check_resources()
    init = time.perf_counter()
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
        initialization_seconds = time.perf_counter() - init
        for row in records:
            check_resources()
            path = pathlib.Path(row["image_path"])
            if audit.digest(path) != row["image_sha256"]:
                raise ValueError("Source image changed")
            start = time.perf_counter()
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Source decode failed")
            height, width = frame.shape[:2]
            scale = min(1.0, 640 / max(width, height))
            resized = cv2.resize(
                frame, (round(width * scale), round(height * scale))
            )
            result = pose.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(
                        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    ),
                )
            )
            xy = np.full((33, 2), np.nan, dtype=np.float32)
            confidence = np.zeros((33, 2), dtype=np.float32)
            if result.pose_landmarks:
                for index, point in enumerate(result.pose_landmarks[0]):
                    confidence[index] = [point.visibility, point.presence]
                    if (
                        point.visibility >= 0.5
                        and point.presence >= 0.5
                        and 0 <= point.x <= 1
                        and 0 <= point.y <= 1
                    ):
                        xy[index] = [point.x * width, point.y * height]
            arrays.append(xy)
            confidences.append(confidence)
            durations.append(time.perf_counter() - start)
    target = folder / (output_stem + ".npz")
    np.savez_compressed(
        target,
        image_xy=np.asarray(arrays),
        confidence=np.asarray(confidences),
        ids=np.asarray([r["id"] for r in records]),
        source_sha256=np.asarray([r["image_sha256"] for r in records]),
        model_sha256=np.asarray(model_sha),
        schema=np.asarray("heavy-static-IMAGE-v1"),
    )
    report = dict(
        cache=str(target),
        cache_sha256=audit.digest(target),
        model_sha256=model_sha,
        mode="IMAGE",
        delegate="GPU",
        records=records,
        source_images=len(records),
        initialization_seconds=initialization_seconds,
        decode_resize_inference_seconds=durations,
        inference_label_inputs=False,
        provider_joint_inputs=False,
        FPS=None,
        temporal_score_supported=False,
        fully_observed_left_trunk=sum(
            np.isfinite(a[[11, 23, 27]]).all() for a in arrays
        ),
        fully_observed_right_trunk=sum(
            np.isfinite(a[[12, 24, 28]]).all() for a in arrays
        ),
    )
    # JSON 직렬화에 numpy 정수가 남지 않도록 변환한다.
    report["fully_observed_left_trunk"] = int(
        report["fully_observed_left_trunk"]
    )
    report["fully_observed_right_trunk"] = int(
        report["fully_observed_right_trunk"]
    )
    audit.save_json(folder / (output_stem + ".json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", action="store_true")
    arguments = parser.parse_args()
    report = run("heavy_static_repeat" if arguments.repeat else "heavy_static")
    print(
        {
            k: report[k]
            for k in [
                "source_images",
                "fully_observed_left_trunk",
                "fully_observed_right_trunk",
                "initialization_seconds",
            ]
        }
    )
