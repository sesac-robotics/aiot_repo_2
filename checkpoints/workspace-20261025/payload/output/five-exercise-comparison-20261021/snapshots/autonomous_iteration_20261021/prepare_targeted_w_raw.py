"""W 오류 후보의 원본 크기 대표 장면을 저장한다. 점수는 계산하지 않는다."""

import cv2
import audit
from resources import check_resources


def run() -> None:
    """W 직접 핵심 구간 중앙의 원본 PNG를 후보별로 저장한다.

    Returns:
        None: 관절·규칙·제공자 문자열 없는 원본 장면을 보존한다.

    Raises:
        ValueError: 영상 프레임 디코딩 실패.
        OSError: 원본 읽기 또는 PNG 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    for row in audit.read_json(audit.OUT / "w_targeted_direct_review.json")[
        "reviews"
    ]:
        check_resources()
        frame = sum(row["direct_W_core_frames"]) // 2
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame)
        ok, pixels = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Raw W frame unavailable")
        target = audit.OUT / f'review/w_targeted_{row["id"]}_raw_f{frame}.png'
        cv2.imwrite(str(target), pixels)
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
