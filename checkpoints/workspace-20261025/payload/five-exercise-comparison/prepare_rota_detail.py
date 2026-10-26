"""새 플랭크 후보의 전신·접지·오류 구분을 위해 큰 시간순 장면을 준비한다."""

import cv2
import numpy as np
import audit
from resources import check_resources


def run() -> None:
    """서로 다른 시점 6장을 540×960 크기로 보존한다.

    Returns:
        None: 구도 검토용 PNG와 프레임·해시 JSON을 저장한다. 점수는 없다.

    Raises:
        OSError: 영상 디코딩·파일 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    path = audit.OUT / "sources/dvids_986473.mp4"
    cap = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    fps = cap.get(5)
    frames = [315, 345, 920, 975, 1006, 1037]
    images = []
    for frame in frames:
        check_resources()
        cap.set(1, frame)
        ok, pixels = cap.read()
        if not ok:
            raise OSError("Cannot decode candidate frame")
        image = cv2.resize(pixels, (540, 960))
        cv2.putText(
            image,
            f"f{frame} {frame/fps:.3f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        images.append(image)
    cap.release()
    cv2.imwrite(
        str(audit.OUT / "review/dvids_986473_detail.png"),
        np.vstack([np.hstack(images[:3]), np.hstack(images[3:])]),
    )
    audit.save_json(
        audit.OUT / "review/dvids_986473_detail.json",
        dict(
            frames=frames,
            fps=fps,
            video_sha256=audit.digest(path),
            status="automatically_prepared_not_directly_reviewed",
        ),
    )


if __name__ == "__main__":
    run()
