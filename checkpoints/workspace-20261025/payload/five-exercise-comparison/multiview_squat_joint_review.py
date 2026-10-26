"""각 시점의 추출 관절을 원본 시간순 표본에 겹쳐 검토판으로 준비한다."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import render
from multiview_squat_prepare import OUT
from resources import check_resources


def run() -> dict:
    """관절 캐시의 첫·중간·마지막을 포함한 12장/시점 표본을 저장한다.

    Returns:
        dict: 자동 준비된 10시점 검토판·프레임·캐시 지문. 사람 선택 승인과
        전체 프레임 추적 검증은 하지 않는다.

    Raises:
        ValueError: 캐시 지문 또는 원본 프레임 디코딩 불일치.
        RuntimeError: 자원 보호 기준 위반.
        OSError: 검토판 저장 실패.
    """
    cv2.setNumThreads(1)
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    receipts = []
    for row in audit.read_json(OUT / "extraction.json")["records"]:
        label = annotations[row["id"]]
        for view in ("side", "front"):
            check_resources()
            rec = row[view]
            with np.load(rec["cache"]) as data:
                if str(data["fingerprint"]) != rec["fingerprint"]:
                    raise ValueError("Joint cache fingerprint changed")
                frames = data["frame_numbers"].copy()
                points = data["image_xy"].copy()
            chosen = np.rint(np.linspace(0, len(frames) - 1, 12)).astype(int)
            source = (
                label["video_path"] if view == "side" else label["front_path"]
            )
            cap = cv2.VideoCapture(
                source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            board = Image.new("RGB", (960, 1380), "white")
            draw = ImageDraw.Draw(board)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
                decoded = int(frames[0]) - 1
                for k, idx in enumerate(chosen):
                    while decoded < frames[idx]:
                        if not cap.grab():
                            raise ValueError("Joint audit frame unavailable")
                        decoded += 1
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError("Joint audit retrieve failed")
                    for a, b in render.CONNECTIONS:
                        if np.isfinite(points[idx, [a, b]]).all():
                            pa, pb = np.rint(points[idx, [a, b]]).astype(int)
                            cv2.line(
                                frame, tuple(pa), tuple(pb), (0, 255, 0), 4
                            )
                    for pt in points[idx]:
                        if np.isfinite(pt).all():
                            cv2.circle(
                                frame,
                                tuple(np.rint(pt).astype(int)),
                                5,
                                (0, 255, 255),
                                -1,
                            )
                    image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    image.thumbnail((238, 430))
                    x, y = (k % 4) * 240, (k // 4) * 460
                    board.paste(image, (x, y + 24))
                    draw.text(
                        (x + 2, y + 3),
                        f"{view} frame {frames[idx]}",
                        fill="black",
                    )
            finally:
                cap.release()
            path = OUT / "review" / (row["id"] + "_" + view + "_joints.jpg")
            board.save(path, quality=92)
            receipts.append(
                dict(
                    id=row["id"],
                    view=view,
                    evidence_image=str(path),
                    frames=frames[chosen].tolist(),
                    fingerprint=rec["fingerprint"],
                    automatically_prepared_not_direct_review=True,
                )
            )
    audit.save_json(
        OUT / "joint_review_preparation.json", dict(records=receipts)
    )
    return dict(prepared_views=len(receipts), direct_reviews=0)


if __name__ == "__main__":
    print(json.dumps(run()))
