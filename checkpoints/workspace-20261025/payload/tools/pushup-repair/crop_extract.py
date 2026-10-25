"""Heavy GPU로 동일 구간의 관절·관측 마스크·시각을 보존한다.

점수·제공자 문자열은 추출 입력에 사용하지 않는다. 스쿼트는 연결된
측면·정면을 각각 추출한다. 캐시는 모델·원본·설정 해시로 무효화한다.
mediapipe-pilot/run_nvidia.sh를 통해 실행해야 NVIDIA EGL이 선택된다.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import time
from collections.abc import Callable

import cv2
import mediapipe as mp
import numpy as np

import audit
from resources import check_resources

LOGGER = logging.getLogger(__name__)
SCHEMA = audit.CACHE_SCHEMA
MODEL = audit.BASE / "mediapipe-pilot/models/heavy.task"


def extract_view(
    row: dict,
    view: str,
    path: str,
    destination: pathlib.Path | None = None,
    *,
    delegate: str = "GPU",
    resource_checker: Callable[[], dict] | None = None,
) -> dict:
    """한 시점의 지정 구간을 10Hz 표본으로 추출하고 NPZ에 저장한다.

    Args:
        row: ID·프레임 구간이 있는 감사 manifest 레코드.
        view: side 또는 front 시점 이름.
        path: 실제 로컬 영상 절대 경로.
        destination: 캐시 상위 폴더. None이면 공통 heavy 폴더다.
        delegate: GPU 기본값 또는 CPU. 실행 장치는 캐시 계약에 기록한다.
        resource_checker: CPU 단기 실행 전용 자원 검사 함수. None이면
            기존 온도·메모리 보호를 사용한다. AC는 사용자 요청으로 기록만 한다. GPU에서 사용자 지정 검사는 금지한다.

    Returns:
        dict: 캐시 경로, 관측 비율, 프레임 수, 처리 초, 재사용 여부.
        image_xy는 원본 픽셀 좌표, world_xyz는 미터 단위 추정치다.
        관측 불가는 NaN이며 visibility·presence를 별도로 저장한다.

    Raises:
        ValueError: 영상 메타데이터·구간·디코딩 오류.
        RuntimeError: 관절 추출 또는 자원 보호 검사 실패.
        OSError: 원본·모델 읽기 또는 저장 실패.

    Notes:
        동기화는 기존 라벨의 같은 프레임 구간 계약을 따른다.
        추출만으로 카메라 동기화나 접지·대상 사람을 검증하지 않는다.
    """
    if delegate not in ("GPU", "CPU") or (
        delegate == "GPU" and resource_checker is not None
    ):
        raise ValueError("Custom resource profile is permitted only for CPU")
    ensure_resources = resource_checker or check_resources
    ensure_resources()
    source_hash = audit.digest(pathlib.Path(path))
    expected_hash = (
        row["front_sha256"] if view == "front" else row["video_sha256"]
    )
    if source_hash != expected_hash:
        raise ValueError("Source hash changed after audit")
    config = dict(
        schema="heavy-person-roi-v1",
        crop=row.get("crop"),
        model_sha256=audit.digest(MODEL),
        source_sha256=source_hash,
        segment=row["segment"],
        required_views=row["required_views"],
        sample_hz=10,
        max_edge=960,
        min_visibility=0.5,
        min_presence=0.5,
        delegate=delegate,
        num_poses=1,
        decoding="one_initial_seek_then_sequential_grab",
    )
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    folder = (destination or audit.OUT / "heavy") / row["id"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{view}.npz"
    metadata = folder / f"{view}.json"
    if target.exists() and metadata.exists():
        previous = audit.read_json(metadata)
        if previous["fingerprint"] == fingerprint:
            with np.load(target) as data:
                if str(data["fingerprint"]) == fingerprint:
                    return dict(previous, reused=True)
    capture = cv2.VideoCapture(
        path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
    )
    started = time.perf_counter()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=getattr(mp.tasks.BaseOptions.Delegate, delegate),
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    image_xy, world_xyz, confidence, frame_numbers = [], [], [], []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first = row["segment"]["start_frame"]
        last = row["segment"]["end_frame"]
        if (
            fps <= 0
            or not 0 <= first < last < count
            or abs(fps - row["segment"]["fps"]) > 0.1
        ):
            raise ValueError("Video metadata does not match label segment")
        indices = np.unique(np.rint(np.arange(first, last + 0.01, fps / 10)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        decoded_index = first - 1
        next_check = time.monotonic() + 2
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as pose:
            for index in indices.astype(int):
                if time.monotonic() >= next_check:
                    ensure_resources()
                    next_check = time.monotonic() + 2
                # 긴 원본의 GOP를 시점마다 다시 탐색하는 비용을 피한다.
                while decoded_index < index:
                    if not capture.grab():
                        raise ValueError("Sequential frame decoding failed")
                    decoded_index += 1
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError("Selected frame decoding failed")
                original_height, original_width = frame.shape[:2]
                x0,y0,x1,y1=row.get("crop",[0,0,original_width,original_height])
                frame=frame[y0:y1,x0:x1]
                height, width = frame.shape[:2]
                scale = min(1.0, 960 / max(height, width))
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
                    round((index - first) * 1000 / fps),
                )
                pixels = np.full((33, 2), np.nan, dtype=np.float32)
                world = np.full((33, 3), np.nan, dtype=np.float32)
                reliability = np.zeros((33, 2), dtype=np.float32)
                if result.pose_landmarks:
                    for joint, landmark in enumerate(result.pose_landmarks[0]):
                        reliability[joint] = (
                            landmark.visibility,
                            landmark.presence,
                        )
                        if (
                            landmark.visibility >= 0.5
                            and landmark.presence >= 0.5
                            and 0 <= landmark.x <= 1
                            and 0 <= landmark.y <= 1
                        ):
                            pixels[joint] = (
                                x0 + landmark.x * width,
                                y0 + landmark.y * height,
                            )
                            point = result.pose_world_landmarks[0][joint]
                            world[joint] = point.x, point.y, point.z
                image_xy.append(pixels)
                world_xyz.append(world)
                confidence.append(reliability)
                frame_numbers.append(index)
    finally:
        capture.release()
    sequence = np.asarray(image_xy)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        image_xy=sequence,
        world_xyz=np.asarray(world_xyz),
        confidence=np.asarray(confidence),
        frame_numbers=np.asarray(frame_numbers),
        time_seconds=np.asarray(frame_numbers) / fps,
        observed=np.isfinite(sequence).all(axis=2),
        fingerprint=fingerprint,
    )
    temporary.replace(target)
    report = dict(
        id=row["id"],
        view=view,
        config=config,
        fingerprint=fingerprint,
        cache=str(target),
        frames=len(sequence),
        observed_joint_fraction=float(np.isfinite(sequence).all(2).mean()),
        extract_seconds=time.perf_counter() - started,
        reused=False,
        synchronization_visually_verified=False,
        score_generated=False,
    )
    audit.save_json(metadata, report)
    return report


def run(limit: int, exercises: list[str] | None = None) -> None:
    """학습·개발 검증 자료를 순서대로 추출하고 진행 결과를 저장한다.

    Args:
        limit: 처리할 최대 레코드 수. 0이면 대상 전체를 처리한다.
        exercises: 처리할 운동 코드 목록. None이면 모든 운동이다.

    Returns:
        None: 영상·시점별 원자적 캐시와 처리 보고서를 저장한다.

    Raises:
        ValueError: limit가 음수이거나 영상 구간이 잘못된 경우.
        RuntimeError: GPU 추출 또는 자원 보호 검사 실패.
        OSError: 입력 읽기 또는 결과 저장 실패.
    """
    if limit < 0:
        raise ValueError("limit cannot be negative")
    rows = audit.read_json(audit.OUT / "development_manifest.json")
    selected = [
        row
        for row in rows
        if row["extraction_ready"]
        and row["original_split"] in ("train", "validation")
        and (exercises is None or row["exercise"] in exercises)
    ]
    if limit:
        selected = selected[:limit]
    report = []
    for row in selected:
        primary = (
            "side"
            if row["exercise"] in ("pushups", "squats", "plank")
            else "primary"
        )
        report.append(extract_view(row, primary, row["video_path"]))
        if row["exercise"] == "squats":
            report.append(extract_view(row, "front", row["front_path"]))
        report_name = (
            "extraction_report_" + "_".join(sorted(exercises)) + ".json"
            if exercises
            else "extraction_report.json"
        )
        audit.save_json(audit.OUT / report_name, report)
        LOGGER.info("Extracted %s (%d view caches)", row["id"], len(report))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exercises", nargs="+", choices=list(audit.VIEWS))
    args = parser.parse_args()
    run(args.limit, args.exercises)
