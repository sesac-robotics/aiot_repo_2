"""부족한 오류 장면의 외부 후보를 검색하고 재사용 조건이 확인된 원본만 준비한다.

YouTube 검색 필터 자체를 이용 허락으로 간주하지 않는다. 개별 원본의
CC 표기·길이·기존 채널 분할을 확인하며 영상 검토나 점수는 생성하지 않는다.
"""

import os
import re
import subprocess
import sys
import urllib.parse

import audit
import three_protocol
from resources import check_resources


def metadata_matches(exercise: str, title: str) -> bool:
    """검색 추천에 섞인 다른 주제를 제목으로 걸러 다운로드 낭비를 줄인다.

    Args:
        exercise: pushups, shoulder 또는 squats 검색 범위.
        title: 원본 제공 제목. 정상·오류 점수로 변환하지 않는다.

    Returns:
        bool: 다운로드 후보로 남길 최소 종목 관련성. 실제 동작·시점·오류
        적합성은 별도 직접 검토가 필요하다.
    """
    text = title.casefold()
    if exercise == "pushups":
        return bool(re.search(r"push[\s-]?ups?|press[\s-]?ups?", text))
    if exercise == "squats":
        return "squat" in text
    if exercise == "shoulder":
        return (
            "press" in text
            and any(
                token in text for token in ("shoulder", "overhead", "military")
            )
            and not any(token in text for token in ("bench", "chest", "floor"))
        )
    return False


def run(
    queries: list[tuple[str, str]] | None = None,
    report_name: str = "external_discovery_next",
) -> dict:
    """운동별 검색 상위 다섯 후보의 조건을 검사하고 최대 두 원본을 확보한다.

    Args:
        queries: 운동 코드·검색어 목록. None이면 기존 세 검색어를 사용한다.
        report_name: 독립 결과 파일 접두사. 이전 검색 결과를 덮어쓰지 않는다.

    Returns:
        dict: 검색 URL·후보·원문 라이선스·채널 역할·준비 또는 실패 사유.
        검색·다운로드는 직접 검토나 학습 가능한 라벨 수로 세지 않는다.

    Raises:
        OSError: 후보 명세·메타데이터·결과 파일 접근 실패.
        RuntimeError: 자원 보호 조건 위반. 개별 네트워크 실패는 기록한다.
    """
    root = three_protocol.OUT
    folder = root / "sources"
    env = dict(os.environ, PYTHONPATH=str(audit.BASE / "data/tools/yt-dlp"))
    previous = audit.read_json(root / "youtube_download_receipts.json")[
        "records"
    ]
    for filename in [
        "external_discovery_next.json",
        "external_discovery_errors.json",
    ]:
        if (root / filename).exists():
            previous += audit.read_json(root / filename)["records"]
    known = {r["id"] for r in previous}
    roles = {
        r["channel_id"]: r["evaluation_role"]
        for r in previous
        if r.get("channel_id")
    }
    report = dict(records=[], searches=[], direct_reviews=0, labels_created=0)

    def invoke(arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-warnings",
                "--socket-timeout",
                "12",
                "--retries",
                "1",
                *arguments,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )

    for exercise, query in queries or [
        ("pushups", "push up mistakes"),
        ("shoulder", "standing dumbbell shoulder press mistakes"),
        ("squats", "bodyweight squat mistakes"),
    ]:
        check_resources()
        url = "https://www.youtube.com/results?" + urllib.parse.urlencode(
            dict(search_query=query, sp="EgIwAQ==")
        )
        downloaded = 0
        try:
            found = invoke(
                [
                    "--flat-playlist",
                    "--playlist-end",
                    "5",
                    "--dump-single-json",
                    url,
                ]
            )
            search_file = root / (
                report_name
                + "_search_"
                + str(len(report["searches"]))
                + ".json"
            )
            search_file.write_text(found.stdout, encoding="utf-8")
            entries = audit.read_json(search_file).get("entries", [])
            report["searches"].append(
                dict(exercise=exercise, url=url, returned=len(entries))
            )
        except (subprocess.SubprocessError, ValueError) as error:
            report["searches"].append(
                dict(exercise=exercise, url=url, error=str(error)[:400])
            )
            continue
        for entry in entries:
            id = entry.get("id")
            if not id or id in known:
                continue
            known.add(id)
            row = dict(
                id=id,
                exercise_search=exercise,
                source_url="https://www.youtube.com/watch?v=" + id,
                directly_reviewed=False,
                score_generated=False,
            )
            try:
                metadata = folder / ("youtube_" + id + ".metadata.json")
                if not metadata.exists():
                    result = invoke(
                        [
                            "--skip-download",
                            "--dump-single-json",
                            row["source_url"],
                        ]
                    )
                    metadata.write_text(result.stdout, encoding="utf-8")
                info = audit.read_json(metadata)
                role = roles.get(info.get("channel_id"), "development_train")
                row.update(
                    title=info.get("title"),
                    creator=info.get("uploader"),
                    channel_id=info.get("channel_id"),
                    license=info.get("license"),
                    duration=info.get("duration"),
                    metadata_sha256=audit.digest(metadata),
                    evaluation_role=role,
                )
                reusable = (
                    info.get("license")
                    == "Creative Commons Attribution license (reuse allowed)"
                    and 0 < info.get("duration", 0) < 600
                )
                row["reusable_metadata_gate"] = reusable
                row["metadata_exercise_match"] = metadata_matches(
                    exercise, info.get("title", "")
                )
                if (
                    reusable
                    and row["metadata_exercise_match"]
                    and downloaded < 2
                    and not role.startswith("prospective")
                ):
                    target = folder / ("youtube_" + id + ".mp4")
                    if not target.exists():
                        invoke(
                            [
                                "--load-info-json",
                                str(metadata),
                                "--match-filters",
                                'license~="Creative Commons"',
                                "-f",
                                "bestvideo[height<=360][filesize<=80M][ext=mp4][vcodec^=avc1]",
                                "--max-filesize",
                                "80M",
                                "--no-progress",
                                "-o",
                                str(target),
                            ]
                        )
                    row.update(
                        video_path=str(target),
                        video_sha256=audit.digest(target),
                    )
                    downloaded += 1
            except (subprocess.SubprocessError, ValueError, OSError) as error:
                row["error"] = str(error)[:400]
            report["records"].append(row)
            audit.save_json(root / (report_name + ".json"), report)
    report["completed"] = True
    audit.save_json(root / (report_name + ".json"), report)
    return report


if __name__ == "__main__":
    print(len(run()["records"]))
