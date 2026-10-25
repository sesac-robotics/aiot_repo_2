"""순차 디코딩 최적화가 선택 프레임의 픽셀을 바꾸지 않는지 검사한다.

같은 원본·구간에서 기존 반복 탐색과 새 순차 읽기의 전체 프레임 해시와
읽기 시간을 비교한다. 관절 추론·점수 계산 성능 측정은 포함하지 않는다.
"""

import hashlib
import time

import cv2
import numpy as np

import audit


def verify() -> list[dict]:
    """짧은 원본·긴 원본 후반·15FPS 구간의 픽셀 일치를 확인한다.

    Returns:
        list[dict]: 대표 세 구간의 프레임 수·해시 일치·디코딩 초.

    Raises:
        ValueError: 대표 구간이 없거나 프레임을 읽을 수 없는 경우.
        RuntimeError: 두 방식이 다른 픽셀 프레임을 반환한 경우.
        OSError: manifest 읽기 또는 결과 저장 실패.
    """
    rows = audit.read_json(audit.OUT / "development_manifest.json")
    ids = (
        "subject_001_push_up_good_side",
        "rehab_Mima_2010_2078",
        "subject_019_shoulder_good_front",
    )
    report = []
    for ident in ids:
        row = next(r for r in rows if r["id"] == ident)
        first = row["segment"]["start_frame"]
        last = row["segment"]["end_frame"]
        fps = row["segment"]["fps"]
        indices = np.unique(
            np.rint(np.arange(first, last + 0.01, fps / 10))
        ).astype(int)
        modes = {}
        for mode in ("repeated_seek", "sequential"):
            capture = cv2.VideoCapture(
                row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
            )
            hashes = []
            started = time.perf_counter()
            try:
                current = first - 1
                if mode == "sequential":
                    capture.set(1, first)
                for index in indices:
                    if mode == "repeated_seek":
                        capture.set(1, int(index))
                        ok, frame = capture.read()
                    else:
                        while current < index:
                            if not capture.grab():
                                raise ValueError(
                                    "Sequential verification decode failed"
                                )
                            current += 1
                        ok, frame = capture.retrieve()
                    if not ok:
                        raise ValueError("Verification frame unavailable")
                    hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            finally:
                capture.release()
            modes[mode] = dict(
                seconds=time.perf_counter() - started, hashes=hashes
            )
        equal = (
            modes["repeated_seek"]["hashes"] == modes["sequential"]["hashes"]
        )
        report.append(
            dict(
                id=ident,
                selected_frames=len(indices),
                all_frame_pixels_identical=equal,
                repeated_seek_seconds=modes["repeated_seek"]["seconds"],
                sequential_seconds=modes["sequential"]["seconds"],
            )
        )
        audit.save_json(audit.OUT / "decode_verification.json", report)
        if not equal:
            raise RuntimeError("Decoder optimization changed selected frames")
    return report


if __name__ == "__main__":
    print(verify())
