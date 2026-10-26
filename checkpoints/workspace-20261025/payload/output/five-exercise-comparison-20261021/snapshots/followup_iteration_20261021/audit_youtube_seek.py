"""YouTube 원본의 순차 디코딩과 임의 탐색 프레임 정합을 검사한다."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from resources import check_resources


def run() -> dict:
    """YTW 원본을 처음부터 한 번 읽어 36개 시점과 탐색 차이를 저장한다.

    Returns:
        dict: 실제 순차 프레임의 표시 시각·해시와 임의 탐색 픽셀 일치.
        단계·점수 정답은 만들지 않는다.

    Raises:
        OSError: 출력 접근 실패.
        ValueError: 순차 또는 탐색 디코딩 실패.
        RuntimeError: 자원 보호 중단.
    """
    path = audit.OUT / "sources/youtube_I7_A0R_9z-0.mp4"
    cap = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    count, fps = int(cap.get(7)), cap.get(5)
    sample = np.linspace(0, count - 1, 36).round().astype(int)
    special = [240, 1079, 1379, 2997, 3566, 4346]
    targets = set(sample.tolist() + special)
    canvas = Image.new("RGB", (1536, 6 * 220), "#14212c")
    draw = ImageDraw.Draw(canvas)
    arrays, rows = {}, []
    for index in range(count):
        if index % 180 == 0:
            check_resources()
        if not cap.grab():
            raise ValueError("Sequential decode ended early")
        if index not in targets:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            raise ValueError("Sequential retrieval failed")
        if index in special:
            arrays[index] = frame.copy()
            cv2.imwrite(
                str(audit.OUT / f"review/paul_sequential_f{index}.png"), frame
            )
        if index in sample:
            j = sample.tolist().index(index)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((256, 180))
            x, y = (j % 6) * 256, (j // 6) * 220
            canvas.paste(image, (x, y))
            draw.text(
                (x, y + 184),
                f"SEQUENTIAL f{index} {index/fps:.2f}s",
                fill="white",
            )
        rows.append(
            dict(
                frame=index,
                expected_seconds=index / fps,
                reported_seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
            )
        )
    cap.release()
    checks = []
    for index, expected in arrays.items():
        check_resources()
        cap = cv2.VideoCapture(
            str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, index)
        ok, actual = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Random decode failed")
        checks.append(
            dict(
                frame=index,
                exact_pixels_equal=bool(np.array_equal(expected, actual)),
                mean_absolute_pixel_delta=float(
                    np.mean(
                        np.abs(expected.astype(float) - actual.astype(float))
                    )
                ),
            )
        )
    canvas.save(audit.OUT / "review/paul_sequential_overview.jpg", quality=95)
    report = dict(
        source_sha256=audit.digest(path),
        rows=rows,
        random_seek_checks=checks,
        scores_created=0,
    )
    audit.save_json(audit.OUT / "youtube_seek_audit.json", report)
    return report


if __name__ == "__main__":
    print(run()["random_seek_checks"])
