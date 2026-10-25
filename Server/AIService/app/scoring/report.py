"""두 점수와 판정 근거를 독립 HTML·관절 MP4로 표시한다."""

import html
import pathlib
import subprocess

from app.scoring.sequence import PoseSequence

CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
)


def overlay(
    video: pathlib.Path,
    seq: PoseSequence,
    result: dict,
    destination: pathlib.Path,
) -> pathlib.Path:
    """원본 영상에 추출한 10Hz 관절과 이번 두 점수를 표시한다.

    Args:
        video: 해당 관절을 추출한 원본 경로.
        seq: 원본 픽셀 관절과 원본 프레임 번호.
        result: 이번 엔진의 0~100 점수 또는 None.
        destination: 결과 디렉터리. ffmpeg가 PATH에 있어야 한다.

    Returns:
        pathlib.Path: 브라우저 재생용 H.264 MP4. 총점 미판정은 N/A로 표시한다.

    Raises:
        OSError: 파일·ffmpeg 접근 실패.
        RuntimeError: 원본·표시 프레임 디코딩 또는 기록 실패.
        subprocess.SubprocessError: MP4 변환 실패·60초 시간 초과.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(
        str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    temporary, target = (
        destination / "review_raw.mp4",
        destination / "review.mp4",
    )
    writer = None
    try:
        cap.set(1, int(seq.frame_numbers[0]))
        decoded = int(seq.frame_numbers[0]) - 1
        for index, points in zip(seq.frame_numbers, seq.image_xy):
            while decoded < index:
                if not cap.grab():
                    raise RuntimeError("Overlay source frame unavailable")
                decoded += 1
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError("Overlay source decode failed")
            scale = min(1.0, 800 / max(frame.shape[:2]))
            frame = cv2.resize(
                frame,
                (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(temporary),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    10,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("Overlay output unavailable")
            for a, b in CONNECTIONS:
                if np.isfinite(points[[a, b]]).all():
                    aa, bb = (points[[a, b]] * scale).round().astype(int)
                    cv2.line(frame, tuple(aa), tuple(bb), (80, 240, 100), 2)

            def label(value: float | None) -> str:
                return "N/A" if value is None else f"{value:.1f}"

            text = (
                f"AI {label(result['ai_score'])} | "
                f"Rule {label(result['rule_score'])} | EXPERIMENTAL"
            )
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (20, 20, 20), -1)
            cv2.putText(
                frame,
                text,
                (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
        timeout=60,
    )
    temporary.unlink()
    return target


def write_html(
    result: dict, destination: pathlib.Path, video_name: str | None = None
) -> None:
    """점수·미판정·항목 근거·제약을 UTF-8 HTML로 저장한다.

    Args:
        result: 엔진이 반환한 이번 실행 결과.
        destination: 이미 생성한 결과 폴더.
        video_name: 같은 폴더의 관절 MP4 파일명 또는 None.

    Raises:
        OSError: 출력 파일 저장 실패.
    """

    def cell(value: object) -> str:
        return html.escape("판정 불가" if value is None else str(value))

    rows = "".join(
        "<tr>"
        + "".join(
            "<td>" + cell(x) + "</td>"
            for x in [
                r["id"],
                r["weight"],
                r["ai_grade"],
                r["rule_grade"],
                r["ai_minus_rule"],
                ", ".join(r["reasons"]),
                r.get("training_grade_counts", "기록 없음"),
                r.get("unseen_training_grades", "기록 없음"),
            ]
        )
        + "</tr>"
        for r in result["items"]
    )
    text = (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        "<title>자세 점수 비교</title><style>"
        "body{font:16px sans-serif;margin:30px;line-height:1.6}"
        "td,th{border:1px solid #aaa;padding:8px}"
        "table{border-collapse:collapse}</style>"
    )
    text += "<h1>" + cell(result["exercise"]) + " · 실험용 점수 비교</h1>"
    text += (
        "<p>AI: "
        + cell(result["ai_score"])
        + " / Rule: "
        + cell(result["rule_score"])
        + " / 차이: "
        + cell(result["ai_minus_rule"])
        + "</p>"
    )
    text += (
        "<p>검토된 구간·구도 기준입니다. 전문가 정답·독립 최종 시험 정확도가 아니며 "
        "미관측 항목은 환산하지 않습니다.</p>"
        "<p>학습 등급별 수는 구간 수이며 참여자 수·예측 신뢰도가 아닙니다. "
        "미학습 등급의 오류 구분 성능은 검증되지 않았습니다.</p>"
    )
    if video_name:
        text += (
            '<video controls width="640" src="'
            + html.escape(video_name, quote=True)
            + '"></video>'
        )
    text += (
        "<table><tr><th>항목</th><th>배점</th><th>AI /2</th>"
        "<th>Rule /2</th><th>차이</th><th>판정 근거·학습 제약</th>"
        "<th>학습 등급별 구간 수</th><th>미학습 등급</th></tr>"
        + rows
        + "</table><ul>"
    )
    text += (
        "".join("<li>" + cell(x) + "</li>" for x in result["limitations"])
        + '</ul><p><a href="result.json">전체 수치·관측 근거 JSON</a></p></html>'
    )
    (destination / "index.html").write_text(text, encoding="utf-8")
