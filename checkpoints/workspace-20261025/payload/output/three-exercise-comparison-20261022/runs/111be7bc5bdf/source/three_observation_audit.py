"""최대 도달 관측이 불확실한 숄더 프레스 입력을 큰 시간순 원본 장면으로 준비한다.

표본 준비는 직접 검토·등급 판정이 아니다. 이전 개발 예측에서 확인한
불일치를 감사하며 독립 시험이나 블라인드 검토로 취급하지 않는다.
"""

import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import three_protocol
from resources import check_resources

IDS = [
    "subject_010_shoulder_good_front",
    "subject_011_shoulder_bad_front",
    "subject_011_shoulder_good_front",
    "subject_013_shoulder_bad_front",
    "subject_018_shoulder_bad_front",
    "subject_018_shoulder_good_front",
    "subject_026_shoulder_bad_front",
    "subject_026_shoulder_good_front",
]


def run() -> dict:
    """여덟 기존 개발 입력에서 원본 전체 화면 16장을 시간순 배치한다.

    Returns:
        dict: ID·원본 SHA256·구간·프레임·준비한 이미지와 직접 미검토 상태.

    Raises:
        OSError: 원본·출력 접근 실패.
        ValueError: 영상 디코딩 또는 명시 구간 불일치.
        RuntimeError: 전원·온도·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    rows = {
        r["id"]: r for r in audit.read_json(root / "score_annotations.json")
    }
    results = []
    for id in IDS:
        check_resources()
        row = rows[id]
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        first, last = (row["segment"][k] for k in ("start_frame", "end_frame"))
        selected = np.unique(np.rint(np.linspace(first, last, 16)).astype(int))
        canvas = Image.new("RGB", (1280, 4 * 420 + 36), "#14212a")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), id + " | RAW INPUT AUDIT / NO SCORES", fill="white")
        cap.set(1, first)
        index = first - 1
        try:
            for j, wanted in enumerate(selected):
                while index < wanted:
                    if not cap.grab():
                        raise ValueError("Source sequential decode failed")
                    index += 1
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Source frame missing")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((320, 390))
                x, y = j % 4 * 320, 36 + j // 4 * 420
                canvas.paste(image, (x + (320 - image.width) // 2, y))
                draw.text((x + 4, y + 394), f"frame {wanted}", fill="white")
        finally:
            cap.release()
        target = root / "review" / ("observation_audit_" + id + ".jpg")
        canvas.save(target, quality=94)
        results.append(
            dict(
                id=id,
                video_sha256=row["video_sha256"],
                segment=row["segment"],
                frames=selected.tolist(),
                evidence_image=str(target),
                directly_reviewed=False,
                score_generated=False,
            )
        )
    report = dict(
        records=results,
        independent_test=False,
        selected_after_development_predictions=True,
    )
    audit.save_json(root / "observation_audit_preparation.json", report)
    return report


if __name__ == "__main__":
    print(len(run()["records"]))
