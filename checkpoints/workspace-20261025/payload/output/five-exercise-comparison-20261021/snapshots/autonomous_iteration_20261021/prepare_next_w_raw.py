"""새 W 후보의 직접 선택 시점 원본 PNG를 관절 표시 없이 준비한다."""

import cv2

import audit
from resources import check_resources


def run() -> None:
    """세 후보의 W 시도 후보 중앙 프레임을 원본 크기로 저장한다.

    Returns:
        None: 출력 PNG와 원본 SHA·프레임 연결 기록을 저장한다.

    Raises:
        ValueError: 프레임 디코딩 실패.
        OSError: 원본·결과 접근 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    frames = {"qevd_00280148": 94, "qevd_00148625": 82, "qevd_00099692": 53}
    for row in audit.read_json(audit.OUT / "next_view_candidates.json")[
        "records"
    ]:
        if row["id"] not in frames:
            continue
        check_resources()
        frame = frames[row["id"]]
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Raw candidate decode failed")
        target = audit.OUT / f'review/next_w_{row["id"]}_f{frame}.png'
        if not cv2.imwrite(str(target), pixels):
            raise OSError("Raw image write failed")
        audit.save_json(
            target.with_suffix(".json"),
            dict(
                source_sha256=row["video_sha256"],
                original_frame=frame,
                landmarks_or_rule_displayed=False,
            ),
        )


if __name__ == "__main__":
    run()
