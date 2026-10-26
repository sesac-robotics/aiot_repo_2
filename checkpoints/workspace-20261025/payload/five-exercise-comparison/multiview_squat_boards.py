"""시차 후보를 적용한 두 시점 표본을 나란히 저장한다.

영상 표본 준비이며 직접 검토·동기화 승인·점수 부여는 수행하지 않는다.
"""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
from multiview_squat_prepare import OUT
from resources import check_resources


def run() -> dict:
    """선별된 시도의 경계와 후속 반복을 같은 시차로 비교한다.

    Returns:
        dict: 검토판 경로·원본 프레임 번호. 동기화 승인 값은 항상 False다.

    Raises:
        ValueError: 프레임 디코딩 실패 또는 구간 범위 이탈.
        RuntimeError: 자원 보호 기준 위반.
        OSError: 검토판 저장 실패.
    """
    cv2.setNumThreads(1)
    source = audit.read_json(OUT / "preparation.json")["records"]
    windows = {
        "subject_001_squat_good": (0, 30, 3),
        "subject_001_squat_bad": (12, 43, 13),
        "subject_002_squat_good": (3, 34, -2),
        "subject_002_squat_bad": (24, 70, -14),
        "subject_004_squat_good": (25, 73, 13),
        "subject_004_squat_bad": (20, 61, 4),
    }
    receipts = []
    for row in source:
        if row["id"] not in windows:
            continue
        a, b, lag = windows[row["id"]]
        indices = np.rint(np.linspace(a, b, 7)).astype(int).tolist()
        indices += [80, 95, 105]
        board = Image.new("RGB", (260 * len(indices), 1020), "white")
        draw = ImageDraw.Draw(board)
        for v, view in enumerate(("side", "front")):
            check_resources()
            cap = cv2.VideoCapture(
                row["views"][view]["video_path"],
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_N_THREADS, 1],
            )
            try:
                for j, side_index in enumerate(indices):
                    index = side_index + (lag if v else 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("Paired sample missing")
                    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    im.thumbnail((258, 480))
                    board.paste(im, (j * 260, v * 510 + 25))
                    draw.text(
                        (j * 260 + 4, v * 510 + 4),
                        f"{view}: {index}",
                        fill="black",
                    )
            finally:
                cap.release()
        path = OUT / "review" / (row["id"] + "_aligned.jpg")
        board.save(path, quality=92)
        receipts.append(
            dict(
                id=row["id"],
                side_frames=indices,
                front_frames=[i + lag for i in indices],
                lag_frames=lag,
                evidence_image=str(path),
                synchronization_verified=False,
            )
        )
    audit.save_json(OUT / "paired_sample_receipts.json", dict(records=receipts))
    return dict(boards=len(receipts), direct_reviews=0)


if __name__ == "__main__":
    print(json.dumps(run()))
