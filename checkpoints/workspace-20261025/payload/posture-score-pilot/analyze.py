"""로컬 영상에서 Heavy 관절을 추출하고 반복 점수ㆍ표시 영상을 저장한다.

원본은 보존한다. 최대 120초 영상을 10fps 이하로 분석ㆍ렌더링한다.
점수는 규칙 초안이며 데이터셋 정답으로 학습한 평가가 아니다.
"""

import json
import hashlib
from pathlib import Path
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from scoring import RepetitionTracker, measure

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "mediapipe-pilot/models/heavy.task"
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
]


def analyze(source, folder, exercise, progress):
    """영상 한 개의 관절 표시 MP4와 반복별 JSON을 만든다.

    Args:
        source: 원본 영상 경로. 단일 인물의 측면 촬영을 전제로 한다.
        folder: 새 출력 폴더.
        exercise: squat 또는 pushup.
        progress: 진행 상태 사전을 받아 저장하는 콜백.
    Returns:
        반복 점수ㆍ유효 프레임 비율ㆍ배점 규칙을 담은 사전.
    Raises:
        ValueError: 디코딩ㆍ길이ㆍ출력 영상 검사가 실패한 경우.
        subprocess.CalledProcessError: H.264 변환 실패.
    """
    rules = json.loads((ROOT / "rules.json").read_text())
    tracker = RepetitionTracker(exercise, rules)
    cap = cv2.VideoCapture(str(source))
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not fps > 0 or count < 1:
        cap.release()
        raise ValueError("영상을 열 수 없습니다.")
    if count / fps > 120:
        cap.release()
        raise ValueError("120초 이하 영상을 사용해 주세요.")
    stride = max(1, int(np.ceil(fps / 10)))
    outfps = fps / stride
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )
    side = None
    valid = total = 0
    rows = []
    pose_rows = []
    visibility_rows = []
    idx = 0
    last_stamp = -1
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if (idx - 1) % stride:
                    continue
                t = (idx - 1) / fps
                h, w = frame.shape[:2]
                scale = min(1, 720 / max(h, w))
                w, h = max(2, int(w * scale) // 2 * 2), max(
                    2, int(h * scale) // 2 * 2
                )
                frame = cv2.resize(frame, (w, h))
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                stamp = max(last_stamp + 1, round(t * 1000))
                last_stamp = stamp
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp
                )
                points = {}
                vis = {}
                if result.pose_landmarks:
                    for i, p in enumerate(result.pose_landmarks[0]):
                        if (
                            p.visibility >= rules["visibility"]
                            and p.presence >= rules["presence"]
                            and 0 <= p.x <= 1
                            and 0 <= p.y <= 1
                        ):
                            points[i] = np.array([p.x * w, p.y * h])
                            vis[i] = p.visibility
                candidates = {}
                for name, indices in [
                    ("left", [11, 13, 15, 23, 25, 27]),
                    ("right", [12, 14, 16, 24, 26, 28]),
                ]:
                    ids = (
                        indices
                        if exercise == "pushup"
                        else [indices[i] for i in [0, 3, 4, 5]]
                    )
                    if all(i in points for i in ids):
                        candidates[name] = min(vis[i] for i in ids)
                values = None
                if side is None and candidates:
                    side = max(candidates, key=candidates.get)
                if side:
                    values = measure(points, side, exercise)
                # 양 어깨를 볼 수 있을 때 정면에 가까운 구도는 점수를 보류한다.
                if values and all(i in points for i in [11, 12, 23, 24]):
                    torso = (
                        np.linalg.norm(points[11] - points[23])
                        + np.linalg.norm(points[12] - points[24])
                    ) / 2
                    if np.linalg.norm(points[11] - points[12]) > max(
                        torso * rules["maximum_projected_shoulder_torso_ratio"],
                        15,
                    ):
                        values = None
                pose_array = np.full((33, 2), np.nan, dtype=np.float32)
                visibility_array = np.zeros(33, dtype=np.float32)
                for joint, point in points.items():
                    pose_array[joint] = point / np.array([w, h])
                    visibility_array[joint] = vis[joint]
                pose_rows.append(pose_array)
                visibility_rows.append(visibility_array)
                tracker.update(t, values)
                total += 1
                valid += int(values is not None)
                rows.append({"time": round(t, 3), "values": values})
                for a, b in EDGES:
                    if a in points and b in points:
                        cv2.line(
                            frame,
                            tuple(points[a].astype(int)),
                            tuple(points[b].astype(int)),
                            (70, 230, 100),
                            2,
                        )
                for i, p in points.items():
                    if i >= 11:
                        cv2.circle(
                            frame, tuple(p.astype(int)), 3, (0, 220, 255), -1
                        )
                label = f"Heavy | {len(tracker.results)} reps | " + (
                    "measurable" if values else "not measurable"
                )
                cv2.rectangle(frame, (0, 0), (w, 32), (20, 25, 30), -1)
                cv2.putText(
                    frame,
                    label,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(folder / "raw.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        outfps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise ValueError("영상 저장 실패")
                writer.write(frame)
                if total % 10 == 0:
                    progress(
                        {
                            "state": "processing",
                            "percent": min(95, round(idx / count * 95)),
                        }
                    )
        if idx < count - 1:
            raise ValueError("영상 전체를 읽지 못했습니다.")
    finally:
        cap.release()
        if writer:
            writer.release()
    if total == 0:
        raise ValueError("분석 가능한 영상 프레임이 없습니다.")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(folder / "raw.mp4"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(folder / "overlay.mp4"),
        ],
        check=True,
    )
    check = cv2.VideoCapture(str(folder / "overlay.mp4"))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != total:
        raise ValueError("출력 영상 프레임 검증 실패")
    (folder / "raw.mp4").unlink()
    result = {
        "exercise": exercise,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "coordinate_system": "image-normalized x,y; unavailable joints are NaN",
        "sampling_fps": outfps,
        "label_status": "pseudo_label_unreviewed",
        "human_reviewed": False,
        "side": side,
        "repetitions": tracker.results,
        "valid_ratio": round(valid / total, 3),
        "incomplete": tracker.discarded + int(tracker.active),
        "rules": rules,
        "note": "측면 영상용 실습 기준 점수입니다. 정면ㆍ관절 누락 구간은 측정 불가로 처리합니다. 실제 각도ㆍ카메라 구도는 사람이 확인해야 합니다.",
    }
    np.savez_compressed(
        folder / "landmarks.npz",
        timestamps=np.array([row["time"] for row in rows]),
        xy=np.array(pose_rows),
        visibility=np.array(visibility_rows),
    )
    (folder / "measurements.json").write_text(
        json.dumps(rows, ensure_ascii=False)
    )
    (folder / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    progress({"state": "done", "percent": 100, "result": result})
    return result
