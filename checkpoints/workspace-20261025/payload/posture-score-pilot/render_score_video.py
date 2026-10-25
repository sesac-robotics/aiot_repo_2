"""완료된 반복의 규칙 점수와 출처를 관절 표시 영상에 합성한다.

영상 시작 시점에는 미래 반복의 점수를 표시하지 않는다. 각 반복 종료
이후 해당 점수를 표시하고 끝에는 3초 요약 화면을 추가한다.
"""

import json
from pathlib import Path
import subprocess
import cv2


def render(folder):
    """기존 분석 영상에서 점수 표시 결과 MP4를 새로 만든다.

    Args:
        folder: result.json과 overlay.mp4가 있는 작업 폴더.
    Returns:
        score-video.mp4의 절대 경로. 단위는 영상 시간 초, 점수는 0~100점.
    Raises:
        ValueError: 입력 또는 출력 영상을 디코딩하지 못한 경우.
        subprocess.CalledProcessError: H.264 변환에 실패한 경우.
    """
    folder = Path(folder)
    result = json.loads((folder / "result.json").read_text())
    cap = cv2.VideoCapture(str(folder / "overlay.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0:
        raise ValueError("분석 영상을 열 수 없습니다.")
    raw = folder / "score-raw.mp4"
    out = folder / "score-video.mp4"
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 100)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("점수 영상 저장 실패")
    import numpy as np

    count = 0
    canvas = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = count / fps
            done = [r for r in result["repetitions"] if r["end"] <= t]
            canvas = np.zeros((h + 100, w, 3), dtype=np.uint8)
            canvas[:h] = frame
            label = (
                f"Rep {done[-1]['number']} - {done[-1]['score']:.1f}/100"
                if done
                else "Waiting for a completed repetition"
            )
            for i, text in enumerate(
                [
                    "DRAFT RULE SCORE - not expert validated",
                    label,
                    "Source - " + result["source"]["channel"],
                ]
            ):
                cv2.putText(
                    canvas,
                    text,
                    (12, h + 24 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (100, 235, 210) if i == 1 else (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
            count += 1
        if count != expected or canvas is None:
            raise ValueError("입력 영상 전체를 읽지 못했습니다.")
        reps = result["repetitions"]
        average = sum(r["score"] for r in reps) / len(reps) if reps else 0
        card = np.zeros_like(canvas)
        texts = [
            result["exercise"].upper() + " - RULE SCORE SUMMARY",
            f"Completed repetitions - {len(reps)}",
            f"Mean draft score - {average:.1f}/100",
            "",
            *[f"Rep {r['number']} - {r['score']:.1f}/100" for r in reps],
            "",
            "Not expert validated. Not training ground truth.",
            "Source - " + result["source"]["channel"],
        ]
        for i, text in enumerate(texts):
            cv2.putText(
                card,
                text,
                (20, 45 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (220, 240, 235),
                1,
                cv2.LINE_AA,
            )
        for _ in range(round(3 * fps)):
            writer.write(card)
    finally:
        cap.release()
        writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(raw),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    check = cv2.VideoCapture(str(out))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != count + round(3 * fps):
        raise ValueError("출력 프레임 검증 실패")
    raw.unlink()
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    jobs = json.loads((root / "youtube-demo-jobs.json").read_text())
    for ident in jobs.values():
        print(render(root / "jobs" / ident))
