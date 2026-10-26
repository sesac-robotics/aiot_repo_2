"""스쿼트 검토판의 작은 발·무릎을 확인할 원본 해상도 중간 장면을 준비한다."""

import cv2

import audit
import three_protocol
from resources import check_resources


def run() -> dict:
    """직접 라벨 생성 없이 같은 원본 구간의 중간 프레임을 PNG로 저장한다.

    Returns:
        dict: 원본·프레임·PNG 해시·자동 준비 수. 직접 관찰 기록은 별도다.

    Raises:
        OSError: 결과 읽기·쓰기 실패.
        ValueError: 원본 프레임 디코딩·PNG 저장 실패.
        RuntimeError: 전원·온도·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    folder = root / "review/rehab24_exact"
    folder.mkdir(exist_ok=True)
    prepared = audit.read_json(root / "rehab24_review_preparation.json")
    rows = []
    for row in prepared["records"]:
        a, b = row["segment"]["start_frame"], row["segment"]["end_frame"]
        middle = (a + b) // 2
        for view, path in [
            ("side", row["video_path"]),
            ("front", row["front_path"]),
        ]:
            check_resources()
            target = folder / f"{row['id']}_{view}_{middle}.png"
            if not target.exists():
                cap = cv2.VideoCapture(
                    path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
                )
                try:
                    cap.set(1, a)
                    for _ in range(a, middle + 1):
                        if not cap.grab():
                            raise ValueError("Sequential frame decode failed")
                    ok, pixels = cap.retrieve()
                    if not ok or not cv2.imwrite(str(target), pixels):
                        raise ValueError("Cannot preserve exact review frame")
                finally:
                    cap.release()
            rows.append(
                dict(
                    id=row["id"],
                    view=view,
                    source=path,
                    original_frame=middle,
                    png=str(target),
                    sha256=audit.digest(target),
                    directly_reviewed=False,
                )
            )
    report = dict(
        records=rows,
        automatically_prepared=len(rows),
        direct_reviews=0,
        score_labels_created=0,
    )
    audit.save_json(root / "rehab24_exact_preparation.json", report)
    return report


if __name__ == "__main__":
    print(run()["automatically_prepared"])
