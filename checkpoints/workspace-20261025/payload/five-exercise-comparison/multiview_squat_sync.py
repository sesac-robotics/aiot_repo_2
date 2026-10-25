"""기존 스쿼트 시점의 영상 움직임으로 시간 차이 후보를 찾는다."""

import json
import pathlib

import cv2
import numpy as np

import audit
from multiview_squat_prepare import OUT
from resources import check_resources


def motion_trace(path: str, last: int) -> list[float]:
    """원본 초기 구간의 몸통·하체 중앙 영역에서 수직 광류를 측정한다.

    Args:
        path: 실제 보유 MP4 절대 경로.
        last: 포함할 마지막 원본 프레임 번호, 양수.

    Returns:
        list[float]: 프레임 간 수직 움직임, 축소 영상 픽셀/프레임 단위.
        관절·점수·동일인 또는 같은 수행 인증이 아닌 자동 정렬 보조값이다.

    Raises:
        ValueError: 초기 프레임·중간 영상 디코딩 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    previous = None
    values = []
    try:
        for i in range(last + 1):
            if i % 30 == 0:
                check_resources()
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Motion trace frame missing")
            image = cv2.cvtColor(
                cv2.resize(frame, (96, 160)), cv2.COLOR_BGR2GRAY
            )
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous, image, None, 0.5, 2, 15, 2, 5, 1.1, 0
                )[48:144, 24:72]
                magnitude = np.linalg.norm(flow, axis=2)
                mask = (
                    magnitude >= max(0.10, float(np.percentile(magnitude, 75)))
                ) & (abs(flow[:, :, 1]) >= abs(flow[:, :, 0]) * 0.4)
                values.append(
                    float(np.median(flow[:, :, 1][mask]))
                    if mask.sum() >= 20
                    else 0.0
                )
            previous = image
    finally:
        cap.release()
    return values


def lag_scores(
    side: list[float], front: list[float], maximum: int = 24
) -> list[dict]:
    """측면 프레임에 더할 정면 지연 후보의 상관을 구한다.

    Args:
        side: 측면 프레임 간 수직 움직임.
        front: 정면 프레임 간 수직 움직임.
        maximum: 양·음 방향 탐색 상한, 원본 프레임 단위.

    Returns:
        list[dict]: 상관 내림차순의 lag_frames·correlation. 정면 원본 번호는
        측면 번호+lag_frames다. 반복 동작은 높은 상관이어도 오정렬 가능하다.

    Raises:
        ValueError: 음수 탐색 상한 또는 너무 짧은 입력.
    """
    if maximum < 0 or min(len(side), len(front)) < 12:
        raise ValueError("Longer traces required")
    a = np.convolve(side, np.ones(3) / 3, mode="same")
    b = np.convolve(front, np.ones(3) / 3, mode="same")
    scores = []
    for lag in range(-maximum, maximum + 1):
        first = max(0, -lag)
        last = min(len(a), len(b) - lag)
        x, y = a[first:last], b[first + lag : last + lag]
        if (
            len(x) < max(12, min(len(a), len(b)) // 2)
            or np.std(x) < 1e-5
            or np.std(y) < 1e-5
        ):
            continue
        scores.append(
            dict(lag_frames=lag, correlation=float(np.corrcoef(x, y)[0, 1]))
        )
    return sorted(
        scores, key=lambda x: (-x["correlation"], abs(x["lag_frames"]))
    )


def run() -> dict:
    """준비된 8개 묶음의 전후 절반 지연 차이와 자동 후보를 저장한다.

    Returns:
        dict: 자동 정렬 후보와 드리프트. 사람이 직접 검토한 결과나 동기화
        승인으로 기록하지 않으며 점수 라벨을 만들지 않는다.

    Raises:
        OSError: 준비 기록·결과 접근 실패.
        ValueError: FPS·추적 또는 정렬 계약 위반.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cv2.setNumThreads(1)
    rows = []
    for row in audit.read_json(OUT / "preparation.json")["records"]:
        check_resources()
        views = row["views"]
        if views["side"]["fps"] != views["front"]["fps"]:
            raise ValueError("Different source FPS")
        traces = {
            v: motion_trace(views[v]["video_path"], views[v]["end_frame"])
            for v in ("side", "front")
        }
        scores = lag_scores(traces["side"], traces["front"])
        half = min(map(len, traces.values())) // 2
        first = lag_scores(traces["side"][:half], traces["front"][:half], 18)
        second = lag_scores(traces["side"][half:], traces["front"][half:], 18)
        rows.append(
            dict(
                id=row["id"],
                fps=views["side"]["fps"],
                source_sha256={v: views[v]["video_sha256"] for v in views},
                trace=traces,
                best=scores[0] if scores else None,
                first_half=first[0] if first else None,
                second_half=second[0] if second else None,
                top_alternatives=[
                    x
                    for x in scores
                    if not scores
                    or abs(x["lag_frames"] - scores[0]["lag_frames"]) >= 5
                ][:3],
                automatic_candidate_only=True,
                synchronization_verified=False,
            )
        )
        audit.save_json(
            OUT / "motion_alignment.json",
            dict(records=rows, completed=False, score_labels_created=0),
        )
    audit.save_json(
        OUT / "motion_alignment.json",
        dict(records=rows, completed=True, score_labels_created=0),
    )
    return {
        r["id"]: dict(
            best=r["best"],
            first_half=r["first_half"],
            second_half=r["second_half"],
        )
        for r in rows
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
