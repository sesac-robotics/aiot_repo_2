"""배경 추적 실패 사례의 필요한 신체 영역을 잘라 원본 좌표로 재추출한다.

개발 검증 실패를 보고 설계한 입력 전처리 후보다. 실제 가림 아래 관절을
복원하지 않으며 AI와 규칙에 같은 입력을 제공한다. 원본은 변경하지 않는다.
"""

import hashlib
import json
import pathlib

import cv2
import numpy as np

import audit
import extract
from multiview_squat_prepare import OUT
from resources import check_resources


def extract_masked(row: dict) -> dict:
    """필요한 어깨·골반·무릎·발을 보존한 영역에서 Heavy를 재추출한다.

    Args:
        row: 원본 경로·SHA·원본 프레임 구간이 있는 직접 초안 레코드.

    Returns:
        dict: 원본 좌표·프레임으로 되돌린 관절 캐시와 전처리 계보.
        좌표 단위는 픽셀이며 관절 누락은 NaN이다. 점수는 계산하지 않는다.

    Raises:
        ValueError: 원본 해시·파생 영상·프레임 디코딩 계약 불일치.
        RuntimeError: 자원 보호 기준 위반.
        OSError: 파생 영상 또는 캐시 저장 실패.
    """
    if audit.digest(pathlib.Path(row["video_path"])) != row["video_sha256"]:
        raise ValueError("Original source changed")
    folder = OUT / "roi" / row["id"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "target_roi.mp4"
    meta = folder / "preprocessing.json"
    contract = dict(
        original_source_sha256=row["video_sha256"],
        original_segment=row["segment"],
        crop_fraction=[0.10, 0.30, 0.90, 0.98],
        coordinate_transform="translate_crop_pixels_to_original",
        development_failure_informed=True,
    )
    old = audit.read_json(meta) if meta.exists() else {}
    if (
        not target.exists()
        or old.get("contract") != contract
        or old.get("derived_sha256") != audit.digest(target)
    ):
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        writer = None
        try:
            first, last = (
                row["segment"]["start_frame"],
                row["segment"]["end_frame"],
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            for index in range(first, last + 1):
                check_resources()
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Derived frame missing")
                height, width = frame.shape[:2]
                x0, y0 = round(width * 0.10), round(height * 0.30)
                x1, y1 = round(width * 0.90), round(height * 0.98)
                x1 -= (x1 - x0) % 2
                y1 -= (y1 - y0) % 2
                frame = frame[y0:y1, x0:x1]
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(target),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        row["segment"]["fps"],
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise ValueError("Derived writer unavailable")
                writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        audit.save_json(
            meta, dict(contract=contract, derived_sha256=audit.digest(target))
        )
    derived_sha = audit.digest(target)
    segment = row["segment"]
    local = dict(
        start_frame=0,
        end_frame=segment["end_frame"] - segment["start_frame"],
        fps=segment["fps"],
    )
    derived = dict(
        row,
        id=row["id"] + "_target_roi",
        video_sha256=derived_sha,
        segment=local,
        required_views=["side", "synchronized_front"],
    )
    raw = extract.extract_view(derived, "primary", str(target), OUT / "heavy")
    mapping = dict(
        contract,
        derived_sha256=derived_sha,
        derived_fingerprint=raw["fingerprint"],
        derived_cache=raw["cache"],
    )
    config = dict(
        raw["config"],
        source_sha256=row["video_sha256"],
        segment=segment,
        preprocessing=mapping,
    )
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    with np.load(raw["cache"]) as data:
        arrays = {k: data[k].copy() for k in data.files}
    cap = cv2.VideoCapture(row["video_path"])
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    arrays["image_xy"] += np.array([round(width * 0.10), round(height * 0.30)])
    arrays["observed"] = np.isfinite(arrays["image_xy"]).all(2)
    arrays["frame_numbers"] += segment["start_frame"]
    arrays["time_seconds"] = arrays["frame_numbers"] / segment["fps"]
    arrays["fingerprint"] = np.array(fingerprint)
    cache = folder / "original_coordinates.npz"
    np.savez_compressed(cache, **arrays)
    receipt = dict(
        raw,
        id=row["id"],
        cache=str(cache),
        fingerprint=fingerprint,
        config=config,
        preprocessing=mapping,
    )
    audit.save_json(folder / "mapped_receipt.json", receipt)
    return receipt
