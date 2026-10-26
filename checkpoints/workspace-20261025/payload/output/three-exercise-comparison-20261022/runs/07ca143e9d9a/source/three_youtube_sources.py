"""공개 검색에서 찾은 3종 보충 후보의 재사용 조건과 원본 그룹을 보존한다.

메타데이터 조회는 영상 직접 검토가 아니다. CC 표기를 확인한 짧은 원본만
내려받고 재업로드·변형 운동 여부는 별도 직접 검토 후 결정한다.
"""

import argparse
import os
import subprocess
import sys

import audit
import external_review
import three_protocol
from resources import check_resources

CANDIDATES = {
    "gpAwfWZBd1o": "development_train",
    "vlFGTI5JzjI": "development_train",
    "OOe_HrNnQWw": "development_validation",
    "P5_YMQC5tvE": "development_validation",
    "NxK8nYNPdb4": "prospective_reserved_not_yet_reviewed",
}


def run(metadata_only: bool = False) -> dict:
    """검색에서 실제 확인한 후보의 메타데이터·조건·선택 시 검토판을 저장한다.

    Args:
        metadata_only: True면 메타데이터만 조회하고 영상은 처리하지 않는다.

    Returns:
        dict: 출처·채널·라이선스·사전 역할·다운로드 또는 제외 사유 목록.
        점수 라벨은 만들지 않으며 같은 채널 후보를 다른 분할로 나누지 않는다.

    Raises:
        OSError: 파일 접근 실패.
        subprocess.SubprocessError: 90초 메타데이터·다운로드 제한 또는 명령 실패.
        RuntimeError: 전원·자원 보호 조건 위반.
    """
    folder = three_protocol.OUT / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ, PYTHONPATH=str(audit.BASE / "data/tools/yt-dlp")
    )
    report = dict(
        records=[],
        metadata_only=metadata_only,
        direct_reviews=0,
        labels_created=0,
    )
    for id, role in CANDIDATES.items():
        check_resources()
        metadata = folder / f"youtube_{id}.metadata.json"
        url = "https://www.youtube.com/watch?v=" + id
        if not metadata.exists():
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "yt_dlp",
                    "--dump-single-json",
                    "--skip-download",
                    "--no-warnings",
                    "--socket-timeout",
                    "15",
                    "--retries",
                    "1",
                    url,
                ],
                env=environment,
                timeout=90,
                check=True,
                capture_output=True,
                text=True,
            )
            metadata.write_text(result.stdout, encoding="utf-8")
        info = audit.read_json(metadata)
        item = dict(
            id=id,
            source_url=url,
            title=info.get("title"),
            creator=info.get("uploader"),
            channel_id=info.get("channel_id"),
            license=info.get("license"),
            duration=info.get("duration"),
            evaluation_role=role,
            metadata_sha256=audit.digest(metadata),
            directly_reviewed=False,
        )
        reusable = (
            info.get("license")
            == "Creative Commons Attribution license (reuse allowed)"
            and 0 < info.get("duration", 0) < 600
        )
        item["reusable_metadata_gate"] = reusable
        if not metadata_only and reusable:
            target = folder / f"youtube_{id}.mp4"
            if not target.exists():
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
                        "bestvideo[height<=360][filesize<=80M][ext=mp4][vcodec^=avc1]",
                        "--max-filesize",
                        "80M",
                        "--socket-timeout",
                        "15",
                        "--retries",
                        "1",
                        "--no-progress",
                        "-o",
                        str(target),
                    ],
                    env=environment,
                    timeout=90,
                    check=True,
                )
            item.update(
                video_path=str(target), video_sha256=audit.digest(target)
            )
            # 사전 예약 원본은 후보 모델을 동결하기 전 직접 표본·예측을 열지 않는다.
            if not role.startswith("prospective"):
                (three_protocol.OUT / "review").mkdir(exist_ok=True)
                item["sheet"] = external_review.contact_sheet(
                    target,
                    three_protocol.OUT / f"review/youtube_{id}.jpg",
                    0,
                    None,
                    48,
                )
        report["records"].append(item)
        audit.save_json(
            three_protocol.OUT
            / (
                "youtube_metadata_candidates.json"
                if metadata_only
                else "youtube_candidates.json"
            ),
            report,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    print(len(run(parser.parse_args().metadata_only)["records"]))
