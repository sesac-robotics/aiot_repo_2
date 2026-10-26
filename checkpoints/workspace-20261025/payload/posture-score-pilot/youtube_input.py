"""유튜브 공개 영상의 선택 구간을 로컬 분석용 파일로 가져오는 어댑터.

인증 쿠키를 사용하지 않으며 URL은 유튜브 영상 ID로 정규화한다.
다운로드·변환은 별도 프로세스에서 수행하고 최대 240초 후 종료한다.
"""

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 300 * 1024 * 1024


def validate_request(payload: dict) -> dict:
    """영상 주소·운동·시간을 검증하여 다운로드 요청을 정규화한다.

    Args:
        payload: url, exercise, start_seconds, duration_seconds 입력 사전.
            시간 단위는 초이며 길이는 1~120초, 시작은 0초 이상이다.

    Returns:
        정규화한 유튜브 주소와 검증된 운동·시간을 담은 새 사전.

    Raises:
        ValueError: 지원하지 않는 주소·운동·시간 또는 입력 형식인 경우.
    """
    if not isinstance(payload, dict):
        raise ValueError("요청 형식을 확인해 주세요.")
    exercise = payload.get("exercise")
    if exercise not in ("squat", "pushup"):
        raise ValueError("스쿼트 또는 푸시업을 선택해 주세요.")
    url = payload.get("url", "")
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("유튜브 영상 주소를 확인해 주세요.")
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname
        if parsed.scheme not in ("https", "http"):
            raise ValueError
        if parsed.username or parsed.password or parsed.port:
            raise ValueError
        parts = parsed.path.strip("/").split("/")
        if host == "youtu.be" and len(parts) == 1:
            ident = parts[0]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                ident = parse_qs(parsed.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
                ident = parts[1]
            else:
                raise ValueError
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", ident):
            raise ValueError
    except ValueError as error:
        raise ValueError("유튜브의 개별 영상 링크를 입력해 주세요.") from error
    try:
        start = float(payload.get("start_seconds", 0))
        duration = float(payload.get("duration_seconds", 30))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "시작 시간과 분석 길이를 숫자로 입력하세요."
        ) from error
    if not math.isfinite(start) or not 0 <= start <= 86400:
        raise ValueError("시작 시간은 0~86400초로 입력하세요.")
    if not math.isfinite(duration) or not 1 <= duration <= 120:
        raise ValueError("분석 길이는 1~120초로 입력하세요.")
    return {
        "url": "https://www.youtube.com/watch?v=" + ident,
        "exercise": exercise,
        "start_seconds": start,
        "duration_seconds": duration,
    }


def download_clip(request: dict, folder: Path) -> dict:
    """선택 구간을 내려받아 input.video와 간단한 출처 정보를 남긴다.

    Args:
        request: validate_request로 검증한 입력 사전. 시간 단위는 초.
        folder: 서버가 생성한 작업 폴더의 절대 경로.

    Returns:
        원본 주소·제목·채널과 선택 구간(초)을 담은 출처 사전.

    Raises:
        ValueError: 다운로드 거부·시간 초과·잘못된 구간·용량 초과.
        OSError: 작업 폴더나 파일에 접근할 수 없는 경우.
    """
    start = request["start_seconds"]
    end = start + request["duration_seconds"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "tools-dependencies")
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-playlist",
        "--no-plugin-dirs",
        "--no-cache-dir",
        "--no-progress",
        "--no-warnings",
        "--socket-timeout",
        "15",
        "--retries",
        "1",
        "--extractor-retries",
        "1",
        "--max-filesize",
        str(MAX_BYTES),
        "--format",
        "bestvideo[height<=720][ext=mp4]/best[height<=720]",
        "--download-sections",
        f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--write-info-json",
        "--output",
        str(folder / "youtube.%(ext)s"),
        request["url"],
    ]
    with (folder / "download.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=240)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise ValueError(
                "유튜브 영상 다운로드 시간이 초과되었습니다. "
                "네트워크를 확인하거나 영상 파일로 업로드해 주세요."
            ) from error
    if code:
        raise ValueError(
            "유튜브 영상을 가져오지 못했습니다. 네트워크와 공개 여부를 "
            "확인하거나 영상 파일로 업로드해 주세요."
        )
    metadata_path = folder / "youtube.info.json"
    videos = [
        p
        for p in folder.glob("youtube.*")
        if p.suffix in (".mp4", ".webm", ".mkv")
    ]
    if len(videos) != 1 or not metadata_path.is_file():
        raise ValueError("선택 구간의 영상을 받지 못했습니다.")
    info = json.loads(metadata_path.read_text(encoding="utf-8"))
    duration = info.get("duration")
    if info.get("is_live") or info.get("live_status") == "is_live":
        raise ValueError("진행 중인 실시간 방송은 분석할 수 없습니다.")
    if duration is not None and start >= duration:
        raise ValueError("시작 시간이 원본 영상 길이보다 깁니다.")
    video = videos[0]
    if not 0 < video.stat().st_size <= MAX_BYTES:
        raise ValueError("다운로드한 영상은 300MB 이하여야 합니다.")
    video.replace(folder / "input.video")
    metadata_path.unlink()
    return {
        "url": request["url"],
        "title": info.get("title") or "YouTube video",
        "channel": info.get("channel") or info.get("uploader") or "YouTube",
        "original_start_seconds": start,
        "original_end_seconds": min(end, duration) if duration else end,
    }
