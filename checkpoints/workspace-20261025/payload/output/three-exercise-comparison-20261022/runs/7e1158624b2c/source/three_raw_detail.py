"""개발 불일치 사례의 원본 확대 근거를 점수·관절 표시 없이 준비한다.

이 재검토는 개발 예측을 본 뒤의 자료 감사이며 독립 전문가 검증이 아니다.
고정된 카메라 관심 영역만 잘라 원본 하단 장면을 두 시점으로 보여 준다.
"""

import pathlib

import cv2
from PIL import Image, ImageDraw

import audit
import three_protocol
from resources import check_resources

CASES = [
    "rehab24_PM_043_rep3",
    "rehab24_PM_043_rep6",
    "rehab24_PM_043_rep8",
    "rehab24_PM_043_rep10",
    "rehab24_PM_038_rep2",
    "rehab24_PM_105_rep21",
]


def run() -> dict:
    """여섯 개발 사례의 원본 중간 프레임 확대판을 저장한다.

    Returns:
        dict: 카메라별 원본·프레임·확대 영역·검토판 목록. 등급은 생성하지 않는다.

    Raises:
        OSError: 원본·결과 접근 실패.
        ValueError: 프레임 디코딩 실패.
        RuntimeError: 자원 보호 위반.
    """
    root = three_protocol.OUT
    rows = {
        r["id"]: r
        for r in audit.read_json(root / "additional_score_annotations.json")
    }
    exact = {
        (r["id"], r["view"]): r["png"]
        for r in audit.read_json(root / "rehab24_exact_preparation.json")[
            "records"
        ]
    }
    results = []
    for id in CASES:
        check_resources()
        row = rows[id]
        segment = row["segment"]
        middle = (segment["start_frame"] + segment["end_frame"]) // 2
        canvas = Image.new("RGB", (1300, 880), "#14212a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            id + " | RAW SOURCE ONLY | NO SCORE / NO POSE",
            fill="white",
        )
        records = []
        for column, view, path in [
            (0, "side", row["video_path"]),
            (1, "front", row["front_path"]),
        ]:
            known = exact.get((id, view))
            if known:
                frame = cv2.imread(known)
            else:
                cap = cv2.VideoCapture(
                    path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
                )
                cap.set(1, segment["start_frame"])
                try:
                    for _ in range(middle - segment["start_frame"] + 1):
                        if not cap.grab():
                            raise ValueError(
                                "Raw evidence sequential decode failed"
                            )
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError("Raw evidence retrieve failed")
                finally:
                    cap.release()
            h, w = frame.shape[:2]
            box = (
                (0, round(0.38 * h), round(0.9 * w), round(0.98 * h))
                if view == "side"
                else (
                    round(0.25 * w),
                    round(0.30 * h),
                    round(0.64 * w),
                    round(0.99 * h),
                )
            )
            a, b, c, d = box
            image = Image.fromarray(
                cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2RGB)
            )
            image.thumbnail((640, 800))
            canvas.paste(image, (column * 650, 55))
            draw.text(
                (column * 650 + 10, 35),
                f"{view} original frame {middle}",
                fill="white",
            )
            records.append(
                dict(
                    view=view,
                    source=path,
                    original_frame=middle,
                    crop_box_pixels=box,
                )
            )
        target = root / f"review/raw_detail_{id}.jpg"
        canvas.save(target, quality=98)
        results.append(dict(id=id, evidence_image=str(target), views=records))
    output = dict(
        records=results,
        directly_reviewed=False,
        score_generated=False,
        review_after_development_analysis=True,
        independent_expert_review=False,
    )
    audit.save_json(root / "raw_detail_preparation.json", output)
    return output


if __name__ == "__main__":
    print(len(run()["records"]))
