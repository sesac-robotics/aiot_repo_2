"""추가 CC 플랭크 후보의 시간순 직접 검토판과 원본 장면을 준비한다."""

import cv2

import audit
from external_review import contact_sheet
from resources import check_resources


def run() -> dict:
    """짧은 오류 후보 구간을 준비하며 점수·정상 여부는 결정하지 않는다.

    Returns:
        dict: 원본 해시, 초·프레임 위치, 검토판 경로. 모두 개발 후보이며
        같은 원본의 정상·오류 시범은 같은 참여자 그룹으로 유지한다.

    Raises:
        ValueError: 원본 장면 디코딩 실패.
        OSError: 파일 접근 또는 PNG 저장 실패.
        RuntimeError: 자원 보호 중단.
    """
    records = []
    for name, source_id, start, end in [
        ("pannell_setup", "hR38KqFSe7A", 12, 18),
        ("pannell_errors", "hR38KqFSe7A", 19, 27),
        ("colossus_opening", "KbcsJX1DfRs", 20, 36),
        ("colossus_position", "KbcsJX1DfRs", 56, 78),
        ("colossus_tension", "KbcsJX1DfRs", 88, 114),
    ]:
        check_resources()
        source = audit.OUT / f"sources/youtube_{source_id}.mp4"
        sheet = contact_sheet(
            source, audit.OUT / f"review/{name}.jpg", start, end, 36
        )
        cap = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        images = []
        for second in [start + 1, (start + end) / 2, end - 1]:
            frame = round(second * sheet["fps"])
            cap.set(1, frame)
            ok, pixels = cap.read()
            if not ok:
                raise ValueError("Source frame unavailable")
            path = audit.OUT / f"review/{name}_{frame}.png"
            if not cv2.imwrite(str(path), pixels):
                raise OSError("PNG write failed")
            images.append(dict(frame=frame, path=str(path)))
        cap.release()
        records.append(
            dict(
                id=name,
                source_id=source_id,
                start=start,
                end=end,
                sheet=sheet,
                raw_images=images,
                source_sha256=audit.digest(source),
            )
        )
    report = dict(
        records=records,
        preparation_only=True,
        labels_created=0,
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "plank_error_window_preparation.json", report)
    return report


if __name__ == "__main__":
    print(run())
