"""라벨을 쓰지 않는 상체 픽셀 크롭으로 Heavy W 관측 개선을 시험한다.

같은 원본·참여자·프레임의 개발 실험이다. 크롭은 기존 관절 전체 구간의
상체 좌표 범위와 여백만 사용하며 점수·직접 핵심 구간을 사용하지 않는다.
"""

import pathlib
import subprocess
import cv2
import numpy as np

import audit
import w_morphology
import ytw_phases
from extract import extract_view
from resources import check_resources


def crop_bounds(sequence: np.ndarray, width: int, height: int) -> list[int]:
    """전체 표본의 팔·몸통 관측 범위에 여백을 둔 픽셀 상자를 반환한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 누락은 NaN이다.
        width: 원본 가로 픽셀 수.
        height: 원본 세로 픽셀 수.

    Returns:
        list[int]: FFV1 크롭의 x·y·가로·세로 픽셀. 배율 변경은 없다.
        15% 여백과 최소 16픽셀 여백을 사용하며 관측 밖 팔의 포함을 보장하지
        않으므로 후속 원본 대조가 필요하다. 짝수 경계로 색차 샘플을 보존한다.

    Raises:
        ValueError: 좌표 형상·영상 크기·관측 표본이 잘못된 경우.
    """
    seq = np.asarray(sequence)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or width < 2 or height < 2:
        raise ValueError("Invalid ROI input")
    points = seq[:, [11, 12, 13, 14, 15, 16, 23, 24]].reshape(-1, 2)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 8:
        raise ValueError("Too few observed upper-body points")
    low, high = points.min(0), points.max(0)
    pad = np.maximum((high - low) * 0.15, 16)
    low = np.maximum(0, np.floor((low - pad) / 2) * 2).astype(int)
    high = np.minimum(
        [width // 2 * 2, height // 2 * 2], np.ceil((high + pad) / 2) * 2
    ).astype(int)
    if np.any(high - low < 32):
        raise ValueError("Degenerate ROI")
    return [
        int(low[0]),
        int(low[1]),
        int(high[0] - low[0]),
        int(high[1] - low[1]),
    ]


def run() -> dict:
    """오류 유형 12개에 같은 규칙을 적용하고 원본 입력과 비교한다.

    Returns:
        dict: 크롭 연결 정보·캐시·동일 직접 구간 W 등급·자동 단계 진단.
        독립 시험·전문가 정확도가 아니며 공통 입력·학습 라벨은 바꾸지 않는다.

    Raises:
        ValueError: 원본·캐시·픽셀 또는 구간 계약 오류.
        OSError: 입력 또는 결과 접근 실패.
        RuntimeError: 자원 보호 또는 Heavy 추출 실패.
        subprocess.SubprocessError: FFV1 크롭 변환 실패.
    """
    direct = audit.read_json(audit.OUT / "w_targeted_direct_review.json")[
        "reviews"
    ]
    previous = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "w_targeted_results.json")[
            "results"
        ]
    }
    folder = audit.OUT / "w_roi_trial"
    folder.mkdir(exist_ok=True)
    results = []
    for row in direct:
        check_resources()
        if audit.digest(pathlib.Path(row["video_path"])) != row["video_sha256"]:
            raise ValueError("Original source changed")
        original = audit.OUT / "w_targeted_cache" / row["id"] / "primary.npz"
        with np.load(original) as data:
            seq, frames = data["image_xy"], data["frame_numbers"]
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        width, height = int(cap.get(3)), int(cap.get(4))
        cap.release()
        x, y, w, h = crop_bounds(seq, width, height)
        target = folder / f'{row["id"]}.mkv'
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            row["video_path"],
            "-an",
            "-vf",
            f"crop={w}:{h}:{x}:{y}",
            "-c:v",
            "ffv1",
            "-threads",
            "1",
            str(target),
        ]
        subprocess.run(command, check=True, timeout=60)
        # 원본과 파생 영상의 세 표본을 원본 픽셀 단위로 대조한다.
        src = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        dst = cv2.VideoCapture(
            str(target), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        parity = []
        for frame in (
            int(frames[0]),
            int(frames[len(frames) // 2]),
            int(frames[-1]),
        ):
            src.set(1, frame)
            dst.set(1, frame)
            ok, a = src.read()
            good, b = dst.read()
            # YUV 크롭 후 BGR 변환의 색차 경계는 비교에서 2픽셀 제외한다.
            equal = bool(
                ok
                and good
                and np.array_equal(
                    a[y + 2 : y + h - 2, x + 2 : x + w - 2],
                    b[2 : h - 2, 2 : w - 2],
                )
            )
            parity.append(dict(frame=frame, BGR_interior_equal=equal))
        src.release()
        dst.release()
        if not all(p["BGR_interior_equal"] for p in parity):
            raise ValueError("Crop pixel parity failed")
        prepared = dict(
            row,
            id=row["id"] + "_roi",
            video_sha256=audit.digest(target),
            required_views=["upper_body_ROI_development_only"],
        )
        extracted = extract_view(
            prepared, "primary", str(target), audit.OUT / "w_roi_cache"
        )
        with np.load(extracted["cache"]) as data:
            roi, selected = data["image_xy"], data["frame_numbers"]
        if not np.array_equal(frames, selected):
            raise ValueError("ROI and original sampling differs")
        core = row["direct_W_core_frames"]
        indices = np.flatnonzero((selected >= core[0]) & (selected <= core[1]))
        manual = (
            w_morphology.evaluate(roi, int(indices[0]), int(indices[-1]))
            if len(indices)
            else None
        )
        phases = ytw_phases.detect(roi)
        window = phases["phases"].get("W_attempt")
        auto = (
            w_morphology.evaluate(
                roi, window["start_index"], window["end_index"]
            )
            if window
            else dict(grade=None, hold_reason="W_attempt_not_detected")
        )
        results.append(
            dict(
                id=row["id"],
                draft=row["W_grade_draft"],
                source_sha256=row["video_sha256"],
                derived_sha256=prepared["video_sha256"],
                derived_path=str(target),
                crop_xywh=[x, y, w, h],
                resized=False,
                pixel_checks=parity,
                crop_uses_score_or_manual_window=False,
                extraction=extracted,
                baseline_manual=previous[row["id"]]["manual_window_diagnostic"],
                roi_manual=manual,
                baseline_auto=previous[row["id"]]["automatic"],
                roi_auto=auto,
                roi_phases=phases,
                independent_final_test=False,
            )
        )
        audit.save_json(
            audit.OUT / "w_roi_trial_results.json",
            dict(results=results, complete=False, adopted=False),
        )
    report = dict(
        results=results,
        complete=True,
        adopted=False,
        independent_final_test=False,
        labels_modified=False,
        common_input_modified=False,
    )
    for variant in ("baseline", "roi"):
        known = [r for r in results if r["draft"] is not None]
        assessed = [
            r
            for r in known
            if r[f"{variant}_manual"] is not None
            and r[f"{variant}_manual"]["grade"] is not None
        ]
        report[variant + "_summary"] = dict(
            known=len(known),
            manual_window_assessed=len(assessed),
            manual_window_MAE=(
                float(
                    np.mean(
                        [
                            abs(r[f"{variant}_manual"]["grade"] - r["draft"])
                            for r in assessed
                        ]
                    )
                )
                if assessed
                else None
            ),
            grade_on_human_unknown=sum(
                r["draft"] is None
                and r[f"{variant}_manual"] is not None
                and r[f"{variant}_manual"]["grade"] is not None
                for r in results
            ),
        )
    audit.save_json(audit.OUT / "w_roi_trial_results.json", report)
    return report


if __name__ == "__main__":
    r = run()
    print(r["baseline_summary"], r["roi_summary"])
