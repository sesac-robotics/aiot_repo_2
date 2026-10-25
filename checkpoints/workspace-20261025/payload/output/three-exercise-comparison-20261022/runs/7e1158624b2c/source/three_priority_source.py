"""기존 메타데이터에서 고른 스쿼트 비틀림 후보 하나의 원본·개요를 준비한다."""

import os
import subprocess
import sys

import audit
import external_review
import three_protocol
from resources import check_resources


def run(id: str = "g_U121IYD1Y") -> dict:
    """이미 학습 역할로 분류한 원제작자 후보를 작은 해상도로 확보한다.

    Args:
        id: 이미 검색 명세에 있으며 사전 역할이 확정된 YouTube 원본 ID.

    Returns:
        dict: 원본·라이선스·채널·역할·해시·개요 검토판. 점수는 생성하지 않는다.

    Raises:
        OSError: 메타데이터·원본·결과 파일 접근 실패.
        ValueError: 재사용 조건 또는 예약 역할 불일치.
        subprocess.SubprocessError: 90초 제한 또는 다운로드 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    check_resources()
    row = next(
        r
        for r in audit.read_json(root / "external_discovery_errors.json")[
            "records"
        ]
        if r["id"] == id
    )
    if not row["reusable_metadata_gate"] or row["evaluation_role"].startswith(
        "prospective"
    ):
        raise ValueError("Reuse condition or development role missing")
    metadata = root / f"sources/youtube_{id}.metadata.json"
    target = root / f"sources/youtube_{id}.mp4"
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
                "12",
                "--retries",
                "1",
                "--no-progress",
                "-o",
                str(target),
            ],
            env=dict(
                os.environ, PYTHONPATH=str(audit.BASE / "data/tools/yt-dlp")
            ),
            check=True,
            timeout=90,
            capture_output=True,
        )
    row.update(video_path=str(target), video_sha256=audit.digest(target))
    row["sheet"] = external_review.contact_sheet(
        target, root / f"review/priority_{id}.jpg", 0, None, 48
    )
    row["directly_reviewed"] = False
    receipt = root / "priority_source.json"
    previous = audit.read_json(receipt)["records"] if receipt.exists() else []
    audit.save_json(
        receipt,
        dict(
            records=[r for r in previous if r["id"] != id] + [row],
            score_labels_created=0,
        ),
    )
    return row


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else "g_U121IYD1Y")["id"])
