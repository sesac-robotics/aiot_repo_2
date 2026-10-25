"""CC YouTube 자료의 운동·변형·시점별 직접 검토 구간을 준비한다."""

import cv2

import audit
from external_review import contact_sheet
from resources import check_resources


def run() -> dict:
    """Y·T·W 단계별 측면·정면과 플랭크 유지·변형 구간을 준비한다.

    Returns:
        dict: 초 단위 후보 범위, 원본 프레임·해시, 원본 PNG 위치.
        시점이 달라도 같은 원본·수행자 그룹이며 동기화된 시도는 아니다.

    Raises:
        ValueError: 시간 구간·메타데이터·프레임 디코딩 오류.
        OSError: 입력·출력 접근 실패.
        RuntimeError: 자원 보호 중단.
    """
    specifications = [
        ("paul_side", "I7_A0R_9z-0", 4, 24, [8, 18]),
        ("paul_front", "I7_A0R_9z-0", 32, 51, [36, 46]),
        ("paul_t_side", "I7_A0R_9z-0", 59, 80, [70]),
        ("paul_t_front", "I7_A0R_9z-0", 91, 109, [100]),
        ("paul_w_side", "I7_A0R_9z-0", 118, 140, [125, 135]),
        ("paul_w_front", "I7_A0R_9z-0", 150, 170, [155, 165]),
        ("mark_basic", "dXX3cNj9a0o", 5, 8, [6]),
        ("mark_kneeling_variant", "dXX3cNj9a0o", 27, 30, [28]),
        ("chris_basic", "3UNXPzP0QaY", 29, 32, [30]),
    ]
    rows = []
    for name, source_id, start, end, raw_times in specifications:
        check_resources()
        source = audit.OUT / f"sources/youtube_{source_id}.mp4"
        sheet = contact_sheet(
            source, audit.OUT / f"review/youtube_{name}.jpg", start, end, 36
        )
        capture = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        raw_images = []
        for second in raw_times:
            frame = round(second * sheet["fps"])
            capture.set(1, frame)
            ok, pixels = capture.read()
            if not ok:
                raise ValueError("Raw source decode failed")
            target = audit.OUT / f"review/youtube_{name}_f{frame}.png"
            if not cv2.imwrite(str(target), pixels):
                raise OSError("Raw PNG write failed")
            raw_images.append(dict(image=str(target), source_frame=frame))
        capture.release()
        rows.append(
            dict(
                id=name,
                source_id=source_id,
                source_group=f"youtube_{source_id}_all_segments",
                start_seconds=start,
                end_seconds=end,
                sheet=sheet,
                raw_images=raw_images,
                score_labels_created=0,
                independent_final_test=False,
                synchronized_multiview=False,
            )
        )
    result = dict(
        records=rows,
        no_provider_strings_converted_to_scores=True,
        preparation_only=True,
    )
    audit.save_json(audit.OUT / "youtube_window_preparation.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
