"""재사용 라이선스를 확인한 YouTube 후보를 로컬 개발 검토용으로 확보한다."""

import argparse
import os
import re
import subprocess
import sys

import audit
from external_review import contact_sheet
from resources import check_resources


def run(source_id: str, max_height: int = 720) -> dict:
    """기존 공개 메타데이터의 CC BY 표기를 확인하고 720p 이하 AVC를 저장한다.

    Args:
        source_id: 이미 조회한 YouTube 영상의 11자리 ID. 단위 없음.
        max_height: 저장 영상 최대 높이, 144~720 픽셀 정수.

    Returns:
        dict: 제작자·원본 URL·CC 표기·원본 SHA·로컬 영상·검토판 경로.
        점수·정상·오류 라벨은 만들지 않고 모든 구간을 같은 원본 그룹으로
        묶는다. 학습 허용 조건은 개별 영상의 추가 제한과 함께 확인한다.

    Raises:
        ValueError: ID 형식·라이선스 또는 길이 조건 불일치.
        OSError: 메타데이터·결과 접근 실패.
        subprocess.SubprocessError: 다운로드 실패 또는 90초 제한 초과.
        RuntimeError: 자원 보호 조건 위반.
    """
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]{11}", source_id)
        or not 144 <= max_height <= 720
    ):
        raise ValueError("Invalid YouTube ID")
    check_resources()
    folder = audit.OUT / "sources"
    metadata = folder / f"youtube_{source_id}.metadata.json"
    info = audit.read_json(metadata)
    if (
        info.get("license")
        != "Creative Commons Attribution license (reuse allowed)"
    ):
        raise ValueError("Verified reusable CC license required")
    if not 0 < info.get("duration", 0) < 600:
        raise ValueError("Candidate exceeds short review acquisition scope")
    target = folder / f"youtube_{source_id}.mp4"
    if not target.exists():
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(audit.BASE / "data/tools/yt-dlp")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--load-info-json",
                str(metadata),
                "--match-filters",
                'license~="Creative Commons"',
                "-f",
                f"bestvideo[height<={max_height}][filesize<=80M][ext=mp4][vcodec^=avc1]",
                "--max-filesize",
                "80M",
                "--socket-timeout",
                "15",
                "--retries",
                "1",
                "--no-progress",
                "--no-continue",
                "-o",
                str(target),
            ],
            env=environment,
            timeout=90,
            check=True,
        )
    sheet_path = audit.OUT / f"review/youtube_{source_id}.jpg"
    sheet = contact_sheet(target, sheet_path, 0, None, 48)
    result = dict(
        source_id=source_id,
        source_url=f"https://www.youtube.com/watch?v={source_id}",
        title=info["title"],
        creator=info["uploader"],
        channel_id=info.get("channel_id"),
        license=info["license"],
        license_metadata_sha256=audit.digest(metadata),
        source_group=f"youtube_{source_id}_all_segments",
        person_group=info.get("channel_id"),
        split="development_candidate",
        independent_final_test=False,
        video_path=str(target),
        video_sha256=audit.digest(target),
        bytes=target.stat().st_size,
        fps=sheet["fps"],
        video_format=f"video only, at most {max_height}p; AVC; known file size at most 80 MiB",
        review_sheet=sheet,
        direct_review=False,
        score_labels_created=0,
    )
    audit.save_json(folder / f"youtube_{source_id}_acquisition.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument("--max-height", type=int, default=720)
    args = parser.parse_args()
    print(run(args.source_id, args.max_height))
