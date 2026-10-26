"""직접 확인한 합성 영상의 측면 패널을 분리하고 원본 계보를 보존한다.

같은 수행자의 두 패널을 두 사람·두 시범으로 세지 않는다. 점수는 만들지 않는다.
"""

import subprocess
import cv2
import numpy as np

import audit
from external_review import contact_sheet
from resources import check_resources


def run() -> dict:
    """원본과 프레임 번호가 같은 무음 측면 파생 영상·검토판을 저장한다.

    Returns:
        dict: 원본·파생 해시, 픽셀 자르기 범위, 표본 픽셀 일치 검사.

    Raises:
        OSError: 원본·파생 영상·결과 저장 파일 접근 실패.
        ValueError: 자르기 픽셀 검사 또는 영상 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
        subprocess.SubprocessError: ffmpeg 실패 또는 90초 제한 초과.
    """
    check_resources()
    original = audit.OUT / "sources/dvids_640237.mp4"
    output = audit.OUT / "sources/dvids_640237_side.mkv"
    crop = [266, 294, 790, 576]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(original),
            "-an",
            "-vf",
            "crop=524:282:266:294",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-threads",
            "1",
            str(output),
        ],
        check=True,
        timeout=90,
    )
    check_resources()
    src = cv2.VideoCapture(
        str(original), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    dst = cv2.VideoCapture(
        str(output), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    checks = []
    try:
        for index in [75, 100, 165]:
            src.set(1, index)
            dst.set(1, index)
            a, b = src.read(), dst.read()
            if not a[0] or not b[0]:
                raise ValueError("Derived frame decoding failed")
            patch = a[1][crop[1] : crop[3], crop[0] : crop[2]]
            diff = np.abs(patch.astype(int) - b[1].astype(int))
            checks.append(
                dict(
                    frame=index,
                    pixels_equal=bool(np.array_equal(patch, b[1])),
                    maximum_channel_difference=int(diff.max()),
                )
            )
    finally:
        src.release()
        dst.release()
    report = dict(
        original_video_path=str(original),
        original_sha256=audit.digest(original),
        derived_video_path=str(output),
        derived_sha256=audit.digest(output),
        spatial_crop_xyxy=crop,
        frame_numbers_preserved=True,
        audio_removed=True,
        codec="FFV1 bgr0",
        pixel_checks=checks,
        person="dvids_gysgt_gates",
        performance_group="dvids_640237_same_hold_both_views",
        split="train_candidate",
        score_generated=False,
    )
    audit.save_json(
        audit.OUT / "sources/dvids_640237_side_lineage.json", report
    )
    contact_sheet(
        output,
        audit.OUT / "review/dvids_640237_side_hold.jpg",
        75 / (30000 / 1001),
        165 / (30000 / 1001),
        24,
    )
    return report


if __name__ == "__main__":
    print(run())
