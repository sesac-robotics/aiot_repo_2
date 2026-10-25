"""다인 4K 영상의 측면 플랭크 후보를 원본 픽셀 크롭으로 분리한다."""

import subprocess
import time
import audit
from resources import check_resources
from external_review import contact_sheet


def run() -> dict:
    """원본 f5430~5520의 1700×650 공간 크롭과 선별판을 저장한다.

    Returns:
        dict: 원본·파생 해시, 원본 프레임 연결, 크롭 픽셀 좌표. 점수는 없다.
        FFV1 무손실 인코딩이며 공간 리사이즈는 하지 않는다. 크롭은 같은
        원본·참여자·수행 그룹으로 유지해야 한다. 원본 정합은 별도 확인한다.

    Raises:
        OSError: 파일 읽기·쓰기·검토판 저장 실패.
        RuntimeError: 자원 보호 조건 위반 또는 변환 실패.
    """
    check_resources()
    source = audit.OUT / "sources/dvids_947417.mp4"
    target = audit.OUT / "sources/dvids_947417_background_candidate.mkv"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-ss",
        "181.181",
        "-c:v",
        "h264_cuvid",
        "-i",
        str(source),
        "-frames:v",
        "91",
        "-an",
        "-vf",
        "crop=1700:650:1680:1000",
        "-c:v",
        "ffv1",
        "-threads",
        "1",
        "-level",
        "3",
        str(target),
    ]
    with (audit.OUT / "fao_candidate_derivation.log").open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=stream)
        try:
            while proc.poll() is None:
                check_resources()
                time.sleep(0.5)
            if proc.returncode:
                raise RuntimeError("Candidate crop conversion failed")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
    record = dict(
        source_path=str(source),
        source_sha256=audit.digest(source),
        derived_path=str(target),
        derived_sha256=audit.digest(target),
        source_start_frame=5430,
        source_end_frame=5520,
        fps=30000 / 1001,
        crop_xywh=[1680, 1000, 1700, 650],
        resized=False,
        codec="FFV1",
        exact_source_pixel_parity_verified=False,
        participant_group="dvids_947417_background_black_beanie",
        source_group="dvids_947417",
        split="train_development_candidate",
        independent_final_test=False,
        score_generated=False,
        command=command,
    )
    audit.save_json(audit.OUT / "fao_candidate_lineage.json", record)
    contact_sheet(
        target, audit.OUT / "review/dvids_947417_candidate.jpg", 0, None, 36
    )
    return record


if __name__ == "__main__":
    print(run())
