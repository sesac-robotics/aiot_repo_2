"""W 원인 분석용 원본 대표 장면을 저장한다. 등급을 만들지 않는다."""

import cv2

import audit
from resources import check_resources


def run() -> None:
    """6개 개발 영상의 직접 W 핵심 구간 중앙 장면을 저장한다.

    Returns:
        None: 원본 PNG와 출처 JSON을 w_review 폴더에 저장한다.

    Raises:
        ValueError: 해당 프레임을 디코딩하지 못한 경우.
        RuntimeError: 자원 보호 조건에 걸린 경우.
    """
    out = audit.OUT / "w_review"
    out.mkdir(exist_ok=True)
    records = audit.read_json(audit.OUT / "ytw_phase_annotations.json")[
        "records"
    ]
    for row in records:
        window = row["phase_windows"].get("W_attempt")
        if not window or row["id"] == "qevd_00105254":
            continue
        check_resources()
        frame = (window["start_frame"] + window["end_frame"]) // 2
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Cannot decode review frame")
        path = out / f'{row["id"]}_f{frame}_raw.png'
        cv2.imwrite(str(path), pixels)
        audit.save_json(
            path.with_suffix(".json"),
            dict(
                video_sha256=row["video_sha256"],
                frame=frame,
                image=str(path),
                labels_or_landmarks_displayed=False,
                independent_final_test=False,
            ),
        )


if __name__ == "__main__":
    run()
