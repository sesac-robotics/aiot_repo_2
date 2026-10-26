"""새 외부 DASH 영상의 탐색 색인을 무손실 재패키징하고 프레임 대응을 검사한다.

원본은 보존한다. 재인코딩·점수 생성 없이 MP4 색인을 다시 쓰며, 원본의
처음부터 연속 디코딩한 표본과 새 파일의 임의 탐색 표본을 직접 비교한다.
"""

import pathlib
import subprocess

import cv2
import numpy as np

import audit
import external_review
import three_protocol
from resources import check_resources


def run() -> dict:
    """문제가 관찰된 긴 외부 영상의 무손실 사본과 검증판을 준비한다.

    Returns:
        dict: 원본·사본 해시, 연속/임의 탐색 픽셀 오차와 검토판 목록.
        검증이 실패한 사본은 학습용으로 승인하지 않는다.

    Raises:
        OSError: 원본·사본·결과 접근 실패.
        ValueError: 프레임 디코딩·색인 대응 불일치.
        subprocess.SubprocessError: ffmpeg 실패 또는 시간 제한 초과.
        RuntimeError: 온도·전원·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    output = []
    for id, window in [
        ("0xbZWGGEK3Q", (30, 85)),
        ("tH1C1J-0n3w", (175, 215)),
        ("OOe_HrNnQWw", (85, 121)),
    ]:
        check_resources()
        original = root / f"sources/youtube_{id}.mp4"
        target = root / f"sources/indexed_{id}.mp4"
        if not target.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(original),
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
                check=True,
                timeout=60,
            )
        sequential = cv2.VideoCapture(
            str(original), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        indexed = cv2.VideoCapture(
            str(target), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        fps = sequential.get(5)
        indices = sorted(
            {round(t * fps) for t in [0, window[0], sum(window) / 2, window[1]]}
        )
        errors = []
        try:
            for index in range(indices[-1] + 1):
                if index % 120 == 0:
                    check_resources()
                if not sequential.grab():
                    raise ValueError("Sequential source decode failed")
                if index not in indices:
                    continue
                ok, a = sequential.retrieve()
                indexed.set(1, index)
                ok2, b = indexed.read()
                if not ok or not ok2:
                    raise ValueError("Index audit frame missing")
                error = float(
                    np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)))
                )
                errors.append(dict(frame=index, max_pixel_error=error))
        finally:
            sequential.release()
            indexed.release()
        passed = all(r["max_pixel_error"] == 0 for r in errors)
        record = dict(
            id=id,
            original_path=str(original),
            indexed_path=str(target),
            original_sha256=audit.digest(original),
            indexed_sha256=audit.digest(target),
            sampled_seek_checks=errors,
            passed=passed,
            not_all_frames_compared=True,
        )
        if passed:
            record["sheet"] = external_review.contact_sheet(
                target, root / f"review/indexed_{id}.jpg", 0, None, 48
            )
            record["window"] = external_review.contact_sheet(
                target,
                root / f"review/indexed_{id}_{window[0]}_{window[1]}.jpg",
                *window,
                48,
            )
        output.append(record)
        audit.save_json(
            root / "media_index_verification.json",
            dict(records=output, completed=False),
        )
        if not passed:
            raise ValueError(
                "Remuxed index does not match sequential source frames"
            )
    result = dict(records=output, completed=True, score_generated=False)
    audit.save_json(root / "media_index_verification.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
