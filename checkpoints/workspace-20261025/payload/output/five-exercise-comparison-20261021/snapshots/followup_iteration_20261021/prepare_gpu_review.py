"""4K 후보를 NVIDIA 디코딩하여 저부하 구도 검토판만 준비한다."""

import subprocess
import time
from PIL import Image, ImageDraw
import audit
from resources import check_resources


def run() -> dict:
    """FAO 영상의 178~197초를 3Hz·1024×540 검토 표본으로 만든다.

    Returns:
        dict: 원본 해시·변환 명령·표본 경로. 리사이즈·빈도 변환을 한
        구도 선별 자료이며 원본 프레임 단위 점수 정답·모델 입력이 아니다.

    Raises:
        RuntimeError: 자원 보호 조건 위반 또는 하드웨어 디코딩 실패.
        OSError: 원본·검토판 저장 실패.
    """
    check_resources()
    folder = audit.OUT / "review/fao_gpu_samples"
    folder.mkdir(exist_ok=True)
    source = audit.OUT / "sources/dvids_947417.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-ss",
        "178",
        "-c:v",
        "h264_cuvid",
        "-resize",
        "1024x540",
        "-i",
        str(source),
        "-t",
        "19",
        "-an",
        "-vf",
        "fps=3",
        "-threads",
        "1",
        "-q:v",
        "3",
        str(folder / "fao_%03d.jpg"),
    ]
    log = audit.OUT / "fao_gpu_review_decoder.log"
    with log.open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=stream)
        try:
            while proc.poll() is None:
                check_resources()
                time.sleep(0.5)
            if proc.returncode:
                raise RuntimeError("Hardware decoder failed; see saved log")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
    paths = sorted(folder.glob("fao_*.jpg"))
    canvas = Image.new("RGB", (1536, ((len(paths) + 5) // 6) * 165), "#14212a")
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image.thumbnail((256, 135))
            x, y = (i % 6) * 256, (i // 6) * 165
            canvas.paste(image, (x, y))
            draw.text((x + 3, y + 138), f"proxy t~{178+i/3:.3f}s", fill="white")
    target = audit.OUT / "review/dvids_947417_gpu_review.jpg"
    canvas.save(target, quality=96)
    result = dict(
        source_video=str(source),
        source_sha256=audit.digest(source),
        command=command,
        hardware_decoder="h264_cuvid",
        samples=len(paths),
        review_image=str(target),
        directly_reviewed=False,
        model_input=False,
        exact_original_frame_labels=False,
    )
    audit.save_json(audit.OUT / "review/dvids_947417_gpu_review.json", result)
    return result


if __name__ == "__main__":
    print(run())
