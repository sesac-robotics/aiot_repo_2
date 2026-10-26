"""기존 W 미수행 실패의 원본과 관절을 같은 프레임에서 표시한다."""

import cv2
import numpy as np
import audit
from resources import check_resources


def run() -> None:
    """원본 표본 4장과 Heavy 좌표를 PNG에 저장한다.

    Returns:
        None: 출력 w_review 폴더에 같은 시각 진단판을 저장한다.

    Raises:
        OSError: 영상 또는 캐시를 읽지 못한 경우.
        RuntimeError: 자원 보호 조건을 위반한 경우.
    """
    row = audit.read_json(audit.OUT / "w_prior_negative_review.json")
    result = audit.read_json(audit.OUT / "w_prior_negative_results.json")
    data = np.load(result["extraction"]["cache"])
    cap = cv2.VideoCapture(
        row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    images = []
    for idx in (10, 14, 16, 18):
        check_resources()
        frame = int(data["frame_numbers"][idx])
        cap.set(1, frame)
        ok, raw = cap.read()
        if not ok:
            raise OSError("Cannot read diagnostic frame")
        overlay = raw.copy()
        xy = data["image_xy"][idx]
        for j in (11, 12, 13, 14, 15, 16, 23, 24):
            if np.isfinite(xy[j]).all():
                point = tuple(xy[j].astype(int))
                cv2.circle(overlay, point, 4, (255, 255, 0), -1)
                cv2.putText(
                    overlay,
                    str(j),
                    point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
        tile = np.hstack([raw, overlay])
        cv2.putText(
            tile,
            f"frame {frame}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        images.append(cv2.resize(tile, (960, 270)))
    cap.release()
    cv2.imwrite(
        str(audit.OUT / "w_review/prior_negative_same_frame.png"),
        np.vstack(images),
    )


if __name__ == "__main__":
    run()
