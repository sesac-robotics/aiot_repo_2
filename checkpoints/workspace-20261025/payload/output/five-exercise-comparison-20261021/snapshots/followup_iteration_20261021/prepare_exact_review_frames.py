"""지정 원본 프레임을 원래 해상도의 검토 이미지로 저장한다."""

import argparse
import pathlib

import cv2

import audit
from external_review import contact_sheet
from resources import check_resources


def run(plan: pathlib.Path) -> dict:
    """프레임 인덱스를 그대로 사용하며 관절·등급·점수를 만들지 않는다.

    Args:
        plan: records에 id, video_path, frames 목록이 있는 JSON 경로.
            frames는 0부터 시작하는 원본 프레임 인덱스다.

    Returns:
        dict: 원본 해시와 읽은 프레임·PNG 경로. overview가 True인 행은
        추가로 36시점 검토판을 생성한다. 이미지 생성은 직접 검토가 아니다.

    Raises:
        OSError: 계획·영상·출력 접근 실패.
        ValueError: 프로젝트 밖 경로·프레임 범위·디코딩 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    records = []
    for row in audit.read_json(plan)["records"]:
        check_resources()
        source = pathlib.Path(row["video_path"]).resolve()
        if not source.is_relative_to(audit.BASE):
            raise ValueError("Review source must be inside project")
        cap = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        images = []
        try:
            for frame in row["frames"]:
                if type(frame) is not int or not 0 <= frame < cap.get(7):
                    raise ValueError("Frame outside source")
                cap.set(1, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError("Exact frame unavailable")
                target = audit.OUT / f'review/{row["id"]}_raw_f{frame}.png'
                if not cv2.imwrite(str(target), pixels):
                    raise OSError("PNG write failed")
                images.append(dict(frame=frame, path=str(target)))
        finally:
            cap.release()
        record = dict(row, images=images, source_sha256=audit.digest(source))
        if row.get("overview"):
            record["sheet"] = contact_sheet(
                source,
                audit.OUT / f'review/{row["id"]}_overview.jpg',
                0,
                None,
                36,
            )
        records.append(record)
    report = dict(records=records, direct_review=False, labels_created=0)
    audit.save_json(plan.with_name(plan.stem + "_prepared.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=pathlib.Path)
    print(len(run(parser.parse_args().plan)["records"]))
