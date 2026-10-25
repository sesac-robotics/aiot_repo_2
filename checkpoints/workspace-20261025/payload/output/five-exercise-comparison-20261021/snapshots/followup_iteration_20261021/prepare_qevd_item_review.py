"""새 QEVD 자동 선별 후보의 실제 원본 장면을 직접 검토판으로 준비한다."""

import pathlib
import cv2
from PIL import Image, ImageDraw
import audit
from resources import check_resources


def run() -> dict:
    """자동 선별 16개 이하의 세 표본 장면을 제공자 판정 없이 저장한다.

    Returns:
        dict: 검토판·원본 프레임·출처 해시. 점수와 직접 검토를 자동 생성하지 않는다.

    Raises:
        OSError: 원본·이미지·결과 접근 실패.
        ValueError: 원본 프레임 디코딩 실패.
        RuntimeError: 자원 보호 실패.
    """
    rows = audit.read_json(audit.OUT / "plank_auto_screen_items/results.json")
    records = []
    for start in range(0, min(len(rows), 16), 8):
        group = rows[start : start + 8]
        canvas = Image.new("RGB", (1152, len(group) * 240 + 30), "#152333")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(group):
            check_resources()
            cap = cv2.VideoCapture(
                row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            for column, frame in enumerate(row["sample_frames"]):
                cap.set(1, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError("Screening raw frame unavailable")
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image.thumbnail((384, 210))
                x = column * 384
                y = 30 + index * 240
                canvas.paste(image, (x + (384 - image.width) // 2, y))
                draw.text(
                    (x + 5, y + 211),
                    f'{row["id"]} f{frame} {frame/row["fps"]:.2f}s',
                    fill="white",
                )
            cap.release()
        target = audit.OUT / f"review/qevd_item_new_{start//8+1}.jpg"
        canvas.save(target, quality=96)
        records.append(
            dict(
                sheet=str(target),
                ids=[r["id"] for r in group],
                sample_frames=[r["sample_frames"] for r in group],
            )
        )
    report = dict(records=records, preparation_only=True, labels_created=0)
    audit.save_json(audit.OUT / "qevd_item_review_preparation.json", report)
    return report


if __name__ == "__main__":
    print(run())
