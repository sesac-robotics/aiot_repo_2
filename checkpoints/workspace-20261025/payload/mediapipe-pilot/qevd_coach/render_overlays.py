"""시험 영상의 모든 프레임에 Heavy 관절을 그려 재생용 MP4를 만든다.

학습 특징과 모델을 변경하지 않는다. 영상 좌표로 직접 그리며 가려진
관절을 임의 보간하지 않는다. 완료된 운동의 시험 사례만 처리한다.
"""

import json
import logging
from pathlib import Path
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np

import catalog
import pose_features

EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (27, 31),
    (28, 30),
    (30, 32),
    (28, 32),
]


def render_video(detector: object, item: dict, target: Path) -> dict:
    """원본과 같은 시간축으로 관절을 표시한 H.264 영상을 저장한다.

    Args:
        detector: Heavy IMAGE 모드 추론기.
        item: 시험 클립의 경로와 식별자.
        target: 새로 생성할 MP4 경로. 원본은 수정하지 않는다.
    Returns:
        전체·검출 프레임 수와 출력 파일명.
    Raises:
        ValueError: 원본 디코딩·임시 영상 생성이 실패한 경우.
        subprocess.CalledProcessError: H.264 변환 실패.
    """
    cap = cv2.VideoCapture(item["path"])
    writer = None
    raw = target.with_suffix(".raw.mp4")
    pending = target.with_suffix(".pending.mp4")
    frames, detected = 0, 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not fps > 0:
            raise ValueError("원본 영상을 열 수 없습니다: " + item["id"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
            frame = cv2.resize(frame, (w, h))
            result = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ),
                )
            )
            count = 0
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                points = {}
                for k in pose_features.BODY:
                    p = landmarks[k]
                    if (
                        p.visibility >= 0.3
                        and p.presence >= 0.5
                        and 0 <= p.x <= 1
                        and 0 <= p.y <= 1
                    ):
                        points[k] = (round(p.x * (w - 1)), round(p.y * (h - 1)))
                count = len(points)
                for a, b in EDGES:
                    if a in points and b in points:
                        cv2.line(
                            frame,
                            points[a],
                            points[b],
                            (70, 240, 80),
                            3,
                            cv2.LINE_AA,
                        )
                for point in points.values():
                    cv2.circle(frame, point, 5, (20, 20, 20), -1)
                    cv2.circle(frame, point, 3, (0, 220, 255), -1)
            detected += int(count > 0)
            label = f"Heavy | visible joints: {count}"
            cv2.rectangle(frame, (0, 0), (min(w, 350), 31), (15, 25, 35), -1)
            cv2.putText(
                frame,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
                if not writer.isOpened():
                    raise ValueError("임시 영상 저장 실패")
            writer.write(frame)
            frames += 1
        if frames != expected or frames == 0:
            raise ValueError("영상 전체를 디코딩하지 못했습니다")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(pending),
        ],
        check=True,
    )
    check = cv2.VideoCapture(str(pending))
    decoded = 0
    try:
        while check.read()[0]:
            decoded += 1
    finally:
        check.release()
    if decoded != frames:
        raise ValueError("인코딩 결과 프레임 수가 다릅니다")
    pending.replace(target)
    raw.unlink()
    return dict(
        frames=frames, detected_frames=detected, file=target.name, fps=fps
    )


def main() -> None:
    """학습 완료 운동을 감시해 관절 영상을 생성하고 완료 시 종료한다.

    Raises:
        OSError: 결과·원본 디스크에 접근할 수 없는 경우.
        RuntimeError: GPU 초기화·추론 실패.
    """
    logging.basicConfig(level=logging.INFO)
    root = catalog.ROOT
    out = root / "overlays"
    out.mkdir(exist_ok=True)
    status_file = out / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    with pose_features.make_detector("heavy") as detector:
        while True:
            result = root / "results.json"
            records = (
                json.loads(result.read_text()).get("heavy", {})
                if result.exists()
                else {}
            )
            for exercise in records.values():
                for item in exercise.get("demos", []):
                    target = out / (item["id"] + ".mp4")
                    if target.exists() and item["id"] in status:
                        continue
                    status[item["id"]] = render_video(detector, item, target)
                    temp = status_file.with_suffix(".tmp")
                    temp.write_text(
                        json.dumps(status, ensure_ascii=False, indent=2)
                    )
                    temp.replace(status_file)
                    logging.info("관절 영상 생성 완료 %s", item["id"])
            progress = root / "progress.json"
            stage = (
                json.loads(progress.read_text()).get("stage")
                if progress.exists()
                else None
            )
            if stage in ("complete", "failed"):
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
