"""이용 조건을 확인한 새 플랭크 후보에서 짧은 구간만 확보한다.

원본 전체를 저장하지 않고 공개 플레이어 원본의 190~202초를
재인코딩한다. 독립 시험·오류·점수 판정은 생성하지 않는다.
"""

import pathlib
import subprocess
import time

import audit
from external_review import contact_sheet
from resources import check_resources


def run() -> dict:
    """공개 원본의 12초 파생 영상과 시간순 구도 검토판을 저장한다.

    Returns:
        dict: 출처·이용 표기·원본 URL·파생 SHA·변환·개발 분할 정보.
        프레임은 파생 영상 기준이며 원본 프레임 정합 검증 전이다.

    Raises:
        OSError: 디렉터리·파일 접근 실패.
        RuntimeError: 자원 보호 또는 120초 처리 제한 위반.
        subprocess.CalledProcessError: 영상 다운로드·변환 실패.
        ValueError: 검토판 디코딩 실패.
    """
    check_resources()
    source_url = "https://www.dvidshub.net/video/961882/v-corps-bsc-2025-acft"
    media_url = "https://d34w7g4gy10iej.cloudfront.net/video/2505/DOD_110985742/DOD_110985742.mp4"
    target = audit.OUT / "sources/dvids_961882_plank_preview.mp4"
    temporary = target.with_name(target.stem + ".partial.mp4")
    if not target.exists():
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-ss",
            "190",
            "-i",
            media_url,
            "-t",
            "12",
            "-map",
            "0:v:0",
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            "scale=1280:-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-threads",
            "1",
            "-y",
            str(temporary),
        ]
        process = subprocess.Popen(command)
        started = time.monotonic()
        try:
            while process.poll() is None:
                check_resources()
                if time.monotonic() - started > 120:
                    raise RuntimeError("Candidate transfer timeout")
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            temporary.replace(target)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    result = dict(
        source_url=source_url,
        media_url=media_url,
        credit="1st Sgt. Jacob Connor / U.S. Army 5th Mobile Public Affairs Detachment",
        terms_url="https://www.dvidshub.net/about/copyright",
        public_domain_marker=True,
        individual_copyright_restriction_found=False,
        original_full_file_bytes=None,
        original_full_file_downloaded=False,
        source_seek_seconds=190,
        requested_duration_seconds=12,
        transform="ffmpeg accurate input seek; H264 CRF18, 1280 wide; no audio",
        original_frame_pixel_parity_verified=False,
        video_path=str(target),
        sha256=audit.digest(target),
        split="development_candidate",
        independent_final_test=False,
        score_labels_created=0,
        disclaimer="The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement.",
    )
    audit.save_json(audit.OUT / "sources/dvids_961882_acquisition.json", result)
    contact_sheet(
        target, audit.OUT / "review/dvids_961882_plank.jpg", 0, None, 36
    )
    return result


if __name__ == "__main__":
    print(run())
