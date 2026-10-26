"""외부 후보 영상의 시간순 검토판을 만든다. 점수는 생성하지 않는다.

로컬 원본을 읽어 검토판·프레임 번호·해시를 실험 출력 폴더에 보존한다.
설명 자막과 제공자 판정은 채점값으로 바꾸지 않는다.
"""

import argparse
import pathlib
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def contact_sheet(
    path: pathlib.Path,
    output: pathlib.Path,
    start_seconds: float,
    end_seconds: float | None,
    samples: int = 42,
) -> dict:
    """초 단위 구간의 등간격 장면과 원본 연결 정보를 저장한다.

    Args:
        path: 읽을 로컬 영상 경로.
        output: 저장할 JPG 경로.
        start_seconds: 시작 시각, 0 이상 초 단위.
        end_seconds: 종료 시각, 초 단위. None이면 영상 끝이다.
        samples: 표본 장면 수. 6의 배수를 권장한다.

    Returns:
        dict: FPS·프레임 범위·표본 번호·원본 해시·근거 경로.

    Raises:
        ValueError: 구간·표본 수·영상 메타데이터 또는 디코딩 오류.
        OSError: 원본 읽기 또는 결과 저장 실패.
    """
    if samples < 2 or start_seconds < 0:
        raise ValueError("Invalid sampling request")
    check_resources()
    capture = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    try:
        fps = capture.get(5)
        count = int(capture.get(7))
        if fps <= 0 or count < 2:
            raise ValueError("Invalid video metadata")
        start = round(start_seconds * fps)
        end = count - 1 if end_seconds is None else round(end_seconds * fps)
        if not 0 <= start < end < count:
            raise ValueError("Review segment outside video")
        indices = np.unique(
            np.linspace(start, end, samples).round().astype(int)
        )
        canvas = Image.new(
            "RGB", (1536, ((len(indices) + 5) // 6) * 220 + 32), "#14212a"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 7), path.stem + " | AUTOMATIC SAMPLE PREPARATION", fill="white"
        )
        sequential = (end - start) / fps < 20
        decoded_index = start - 1
        if sequential:
            capture.set(1, start)
        last_check = time.monotonic()
        for j, index in enumerate(indices):
            if time.monotonic() - last_check >= 2:
                check_resources()
                last_check = time.monotonic()
            if sequential:
                while decoded_index < int(index):
                    if not capture.grab():
                        raise ValueError("Sequential review decoding failed")
                    decoded_index += 1
                ok, frame = capture.retrieve()
            else:
                capture.set(1, int(index))
                ok, frame = capture.read()
            if not ok:
                raise ValueError("Review frame decoding failed")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((256, 192))
            x, y = (j % 6) * 256, 32 + (j // 6) * 220
            canvas.paste(image, (x + (256 - image.width) // 2, y))
            draw.text(
                (x + 3, y + 196), f"f{index} {index/fps:.3f}s", fill="white"
            )
    finally:
        capture.release()
    canvas.save(output, quality=96)
    result = dict(
        video_path=str(path.resolve()),
        video_sha256=audit.digest(path),
        fps=fps,
        start_frame=start,
        end_frame=end,
        sample_frames=[int(i) for i in indices],
        evidence_image=str(output.resolve()),
        status="automatically_prepared_not_directly_reviewed",
    )
    audit.save_json(output.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--samples", type=int, default=42)
    args = parser.parse_args()
    print(
        contact_sheet(
            args.video, args.output, args.start, args.end, args.samples
        )
    )
