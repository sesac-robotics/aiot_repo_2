"""해군 플랭크 후보의 원본 크기 장면·짧은 구간을 준비한다."""

import cv2
import audit
from resources import check_resources
from external_review import contact_sheet


def run() -> None:
    """서로 다른 자세 시점의 원본 PNG와 3초 시간순 표본을 저장한다.

    Returns:
        None: 장면과 원본 프레임 연결 기록. 자동 채점·관절 표시는 없다.

    Raises:
        ValueError: 원본 디코딩 오류.
        OSError: 원본 읽기·검토판 저장 실패.
        RuntimeError: 자원 보호 중단.
    """
    source = audit.OUT / "sources/dvids_773312.mp4"
    cap = cv2.VideoCapture(
        str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    fps = cap.get(5)
    for second in (100, 124, 180, 188, 194, 198):
        check_resources()
        frame = round(second * fps)
        cap.set(1, frame)
        ok, pixels = cap.read()
        if not ok:
            raise ValueError("Raw Navy frame unavailable")
        target = audit.OUT / f"review/navy773312_raw_{second}.png"
        cv2.imwrite(str(target), pixels)
        audit.save_json(
            target.with_suffix(".json"),
            dict(
                source_sha256=audit.digest(source),
                frame=frame,
                fps=fps,
                landmarks_or_rule_displayed=False,
            ),
        )
    cap.release()
    for start in (100, 122, 193, 197):
        contact_sheet(
            source,
            audit.OUT / f"review/navy773312_core_{start}.jpg",
            start,
            start + 3,
            36,
        )


if __name__ == "__main__":
    run()
