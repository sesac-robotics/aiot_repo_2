"""FAO 크롭의 원본 시간·무손실 픽셀 정합을 세 지점에서 검사한다."""

import subprocess
import cv2
import audit
from resources import check_resources


def run() -> dict:
    """원본과 파생 영상의 f0·30·60 YUV 픽셀 일치를 검사한다.

    Returns:
        dict: 검사 지점·동일 여부·원본/파생 해시. 전체 프레임 일치 증명은
        아니며 세 프레임의 정확한 시간·크롭 연결 회귀 검사이다.

    Raises:
        ValueError: 픽셀 불일치 또는 영상 디코딩 실패.
        OSError: 원본·파생·검사 결과 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
        subprocess.SubprocessError: 하드웨어 디코딩 명령 실패.
    """
    row = audit.read_json(audit.OUT / "fao_candidate_lineage.json")
    checks = []
    for index in (0, 30, 60):
        check_resources()
        prefix = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
        ]
        source = prefix + [
            "-ss",
            f"{(5430+index)*1001/30000:.6f}",
            "-c:v",
            "h264_cuvid",
            "-i",
            row["source_path"],
            "-frames:v",
            "1",
            "-an",
            "-vf",
            "crop=1700:650:1680:1000",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        derived = prefix + [
            "-i",
            row["derived_path"],
            "-frames:v",
            "1",
            "-an",
            "-vf",
            f"select=eq(n\\,{index})",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        a = subprocess.run(
            source, check=True, capture_output=True, timeout=15
        ).stdout
        b = subprocess.run(
            derived, check=True, capture_output=True, timeout=15
        ).stdout
        same = a == b and len(a) == 1700 * 650 * 3 // 2
        checks.append(
            dict(
                derived_frame=index,
                source_frame=5430 + index,
                equal=same,
                bytes=len(a),
            )
        )
        if not same:
            audit.save_json(
                audit.OUT / "fao_pixel_parity.json",
                dict(checks=checks, passed=False),
            )
            raise ValueError("Source/derived pixel or frame parity mismatch")
    cap = cv2.VideoCapture(
        row["derived_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    ok, pixels = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot decode raw review frame")
    cv2.imwrite(
        str(audit.OUT / "review/dvids_947417_candidate_raw_f0.png"), pixels
    )
    result = dict(
        checks=checks,
        passed=True,
        source_sha256=row["source_sha256"],
        derived_sha256=row["derived_sha256"],
        scope="three_exact_YUV_frames_not_all_frames",
    )
    audit.save_json(audit.OUT / "fao_pixel_parity.json", result)
    row["exact_source_pixel_parity_verified"] = True
    row["pixel_parity_scope"] = "frames_0_30_60_only"
    audit.save_json(audit.OUT / "fao_candidate_lineage.json", row)
    return result


if __name__ == "__main__":
    print(run())
