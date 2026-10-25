"""직접 표시한 구간을 원본 프레임과 연결된 무손실 검토 영상으로 분리한다.

정상·오류·변형은 별도 메타데이터로 보존한다. 같은 원본의 모든 시점과
시범은 같은 분할로 묶고 제공자 설명을 점수로 바꾸지 않는다.
"""

import collections
import argparse
import pathlib
import shutil

import cv2
import numpy as np

import audit
from resources import check_resources


def validate_segments(records: list[dict]) -> None:
    """원본·참여자 그룹의 분할 누출과 잘못된 프레임 범위를 거부한다.

    Args:
        records: ID·원본 SHA·참여자·split·시작/끝 프레임을 가진 목록.
        끝 프레임은 미포함이며 물리 시간은 원본 FPS를 따른다.

    Returns:
        None: 모든 구간과 그룹이 유효하면 반환한다.

    Raises:
        ValueError: 범위 오류, ID 중복 또는 같은 원본·참여자의 분할 충돌.
        KeyError: 필수 필드가 없는 경우.
    """
    ids, splits = set(), {}
    for row in records:
        if (
            row["id"] in ids
            or not 0 <= row["start_frame"] < row["end_frame_exclusive"]
        ):
            raise ValueError("Duplicate ID or invalid frame range")
        ids.add(row["id"])
        for key in (row["source_sha256"], row["person_group"]):
            previous = splits.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Source/person group crosses splits")


def run(plan_path: pathlib.Path | None = None) -> dict:
    """표시 구간을 순차 디코딩·FFV1 저장하고 3개 프레임 픽셀을 대조한다.

    Args:
        plan_path: 구간 계획 JSON 경로. None이면 기존 YouTube 계획을 쓴다.
        별도 계획의 결과는 계획 파일명에 _exports를 붙여 보존한다.

    Returns:
        dict: 파생 해시·프레임 수·원본 좌표 크롭·픽셀 일치 결과.
        3개 프레임 일치는 전체 프레임 검사로 표현하지 않는다.

    Raises:
        ValueError: 원본·범위·라이선스·픽셀 정합 오류.
        OSError: 원본·출력 접근 또는 인코더 실패.
        RuntimeError: 온도·전원·디스크 예산 보호 중단.
    """
    default_plan = audit.OUT / "youtube_segment_plan.json"
    plan_path = pathlib.Path(plan_path) if plan_path else default_plan
    plan = audit.read_json(plan_path)
    records = plan["records"]
    validate_segments(records)
    if shutil.disk_usage(audit.OUT).free < 10 * 2**30:
        raise RuntimeError("Insufficient disk reserve")
    folder = audit.OUT / "youtube_segments"
    folder.mkdir(exist_ok=True)
    grouped = collections.defaultdict(list)
    for row in records:
        grouped[row["source_path"]].append(row)
    result = []
    for source, rows in grouped.items():
        check_resources()
        if audit.digest(pathlib.Path(source)) != rows[0]["source_sha256"]:
            raise ValueError("Source changed")
        if any(
            r["license"]
            != "Creative Commons Attribution license (reuse allowed)"
            for r in rows
        ):
            raise ValueError("Reusable source license required")
        cap = cv2.VideoCapture(
            source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        fps, count = cap.get(5), int(cap.get(7))
        if max(r["end_frame_exclusive"] for r in rows) > count:
            raise ValueError("Segment outside source")
        writers, samples, written = {}, {}, {}
        try:
            for row in rows:
                crop = row.get("crop_xyxy") or [
                    0,
                    0,
                    int(cap.get(3)),
                    int(cap.get(4)),
                ]
                x1, y1, x2, y2 = crop
                if not 0 <= x1 < x2 <= cap.get(
                    3
                ) or not 0 <= y1 < y2 <= cap.get(4):
                    raise ValueError("Invalid source crop")
                row["crop_xyxy"] = crop
                target = folder / (row["id"] + ".mkv")
                writer = cv2.VideoWriter(
                    str(target),
                    cv2.VideoWriter_fourcc(*"FFV1"),
                    fps,
                    (x2 - x1, y2 - y1),
                )
                if not writer.isOpened():
                    raise OSError("FFV1 encoder unavailable")
                writers[row["id"]] = writer
                samples[row["id"]] = {}
                written[row["id"]] = 0
            for frame_index in range(
                max(r["end_frame_exclusive"] for r in rows)
            ):
                if frame_index % 150 == 0:
                    check_resources()
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError("Sequential source decode failed")
                for row in rows:
                    start, end = row["start_frame"], row["end_frame_exclusive"]
                    if not start <= frame_index < end:
                        continue
                    x1, y1, x2, y2 = row["crop_xyxy"]
                    cropped = np.ascontiguousarray(pixels[y1:y2, x1:x2])
                    writers[row["id"]].write(cropped)
                    local = frame_index - start
                    written[row["id"]] += 1
                    if local in (0, (end - start) // 2, end - start - 1):
                        samples[row["id"]][local] = cropped.copy()
        finally:
            cap.release()
            for writer in writers.values():
                writer.release()
        for row in rows:
            target = folder / (row["id"] + ".mkv")
            cap = cv2.VideoCapture(
                str(target), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            checks = []
            for local, expected in samples[row["id"]].items():
                cap.set(1, local)
                ok, actual = cap.read()
                match = ok and np.array_equal(expected, actual)
                checks.append(
                    dict(
                        clip_frame=local,
                        source_frame=local + row["start_frame"],
                        exact_pixels_equal=bool(match),
                    )
                )
                if not match:
                    raise ValueError("Export/source crop pixel mismatch")
            cap.release()
            result.append(
                dict(
                    row,
                    video_path=str(target),
                    video_sha256=audit.digest(target),
                    frames=written[row["id"]],
                    fps=fps,
                    pixel_checks=checks,
                    full_score=None,
                )
            )
    size = sum((folder / (r["id"] + ".mkv")).stat().st_size for r in records)
    if size > 2**30:
        raise RuntimeError("Selective extraction exceeds 1 GiB budget")
    report = dict(
        records=result,
        bytes=size,
        full_score_labels_created=0,
        independent_final_test=False,
        input_plan_sha256=audit.digest(plan_path),
    )
    report_path = (
        audit.OUT / "youtube_segment_exports.json"
        if plan_path == default_plan
        else plan_path.with_name(plan_path.stem + "_exports.json")
    )
    audit.save_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path, nargs="?")
    print(
        {
            k: v
            for k, v in run(parser.parse_args().plan).items()
            if k != "records"
        }
    )
