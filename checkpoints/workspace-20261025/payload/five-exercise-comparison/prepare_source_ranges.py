"""JSON에 지정한 공개 원본 후보의 시간순 검토판·원본 장면을 준비한다."""

import argparse
import pathlib
import re

import cv2

import audit
from external_review import contact_sheet
from resources import check_resources


def run(plan_path: pathlib.Path) -> dict:
    """초 단위 후보 구간을 직접 검토용 이미지로 준비하고 계보를 기록한다.

    Args:
        plan_path: id·source_id·start·end 목록 records를 가진 JSON 절대 경로.
        시간 단위는 초이며 결과명은 계획 파일 이름에 _prepared를 붙인다.
        video_path가 있으면 프로젝트 내부의 로컬 원본 경로를 사용한다.

    Returns:
        dict: 원본 해시, 프레임·FPS, 검토판과 원본 PNG 경로. 정상·오류나
        점수는 자동 생성하지 않는다. 전체 영상 프레임 검토를 뜻하지 않는다.

    Raises:
        ValueError: ID·시간 범위·원본 디코딩 오류.
        OSError: 파일 접근·PNG 저장 실패.
        RuntimeError: 자원 보호 중단.
    """
    rows = []
    for row in audit.read_json(plan_path)["records"]:
        check_resources()
        name, source_id = row["id"], row["source_id"]
        source_pattern = (
            r"[a-zA-Z0-9_-]+" if row.get("video_path") else r"[a-zA-Z0-9_-]{11}"
        )
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not re.fullmatch(
            source_pattern, source_id
        ):
            raise ValueError("Invalid source or range ID")
        start, end = row["start"], row["end"]
        if not 0 <= start < end:
            raise ValueError("Invalid seconds range")
        source = (
            pathlib.Path(row["video_path"]).resolve()
            if row.get("video_path")
            else audit.OUT / f"sources/youtube_{source_id}.mp4"
        )
        if not source.is_relative_to(audit.BASE):
            raise ValueError(
                "Local review source must stay inside this project"
            )
        sheet = contact_sheet(
            source, audit.OUT / f"review/{name}.jpg", start, end, 36
        )
        cap = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        images = []
        for fraction in [0.2, 0.5, 0.8]:
            frame = round((start + (end - start) * fraction) * sheet["fps"])
            cap.set(1, frame)
            ok, pixels = cap.read()
            if not ok:
                raise ValueError("Source frame unavailable")
            target = audit.OUT / f"review/{name}_{frame}.png"
            if not cv2.imwrite(str(target), pixels):
                raise OSError("PNG save failed")
            images.append(dict(frame=frame, path=str(target)))
        cap.release()
        rows.append(
            dict(
                row,
                sheet=sheet,
                raw_images=images,
                source_sha256=audit.digest(source),
            )
        )
    report = dict(records=rows, preparation_only=True, labels_created=0)
    audit.save_json(
        plan_path.with_name(plan_path.stem + "_prepared.json"), report
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path)
    print(run(parser.parse_args().plan))
