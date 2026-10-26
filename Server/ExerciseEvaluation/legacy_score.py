"""스쿼트 보존본의 구간 점수 계산을 새 실행 경로에서 재사용합니다.

출처: checkpoints/squat-20261025/payload/runtime/output/live-webcam-20261023/server.py.
세 함수의 계산은 보존본에서 복사했으며 모델·원본 파일을 변경하지 않습니다.
의존 모듈은 worker 실행 환경의 PYTHONPATH로 보존 경로에서 불러옵니다.
"""

import threading

import numpy as np
from accepted_items import evaluate as evaluate_accepted_items
from front_candidate import predict as predict_front_candidate
from front_policy import apply_front_policy, frontal_motion_observations
from live_observation import mask_frame_border
from method_comparison import compare_candidate
from rubric100 import assessment as assess100
from squat_coach import completed_feedback
from unified_assessment import apply as unify_assessment

from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence

MODEL_LOCK = threading.Lock()
ENGINE = ScoringEngine()


def safe(x):
    """배열·비유한 값을 JSON 허용 값으로 변환한다."""
    if isinstance(x, dict):
        return {k: safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [safe(v) for v in x]
    if isinstance(x, np.generic):
        return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def score_samples(
    samples,
    width,
    height,
    exercise,
    view,
    complete=False,
    phase="complete",
    standing_reference=None,
    return_samples=None,
):
    """실제 표본의 누락을 NaN으로 유지해 동일 모델로 구간을 평가한다.

    Args:
        samples: 10Hz 슬롯 번호와 픽셀 관절 배열의 목록.
        width: 영상 너비, 픽셀.
        height: 영상 높이, 픽셀.
        exercise: 사용자가 선택한 운동 코드.
        view: 사용자가 확인한 촬영 방향.
        complete: 사용자가 완결 수행을 확인한 경우만 True.

    Returns:
        dict: 모델 결과. 완결 미확인 실시간 구간의 총점은 None이다.

    Raises:
        ValueError: 표본 부족·잘못된 구도 또는 입력 계약 불일치.
    """
    if len(samples) < 10:
        raise ValueError("최소 1초 이상 수집하세요")
    ticks = np.arange(samples[0][0], samples[-1][0] + 1)
    xy = np.full((len(ticks), 33, 2), np.nan)
    for tick, points in samples:
        xy[tick - ticks[0]] = points
    xy = mask_frame_border(xy, width, height)
    return_xy = None
    if return_samples:
        return_xy = np.full(
            (return_samples[-1][0] - return_samples[0][0] + 1, 33, 2), np.nan
        )
        for tick, points in return_samples:
            return_xy[tick - return_samples[0][0]] = points
        return_xy = mask_frame_border(return_xy, width, height)
    seq = PoseSequence(xy, ticks - ticks[0], 10.0, width, height)
    with MODEL_LOCK:
        result = apply_front_policy(
            safe(
                ENGINE.score(
                    exercise, seq, view=view, complete_attempt=complete
                )
            ),
            exercise,
            view,
        )
        if exercise == "squats" and view == "front":
            result["frontal_motion_observations"] = frontal_motion_observations(
                xy
            )
            try:
                result["front_candidate"] = predict_front_candidate(xy)
            except ValueError as error:
                result["front_candidate"] = {
                    "error": str(error),
                    "experimental": True,
                }
            result["method_comparison"] = compare_candidate(
                result.get("front_candidate"),
                xy,
                width=width,
                height=height,
                standing_reference=standing_reference,
                phase=phase,
                return_xy=return_xy,
            )
            result["coaching"] = completed_feedback(
                xy, result.get("front_candidate")
            )
            result["accepted_items"] = evaluate_accepted_items(
                xy, width, height
            )
            for item in result["accepted_items"]["items"]:
                for check in result["coaching"].get("checks", []):
                    if check["id"] == item["id"]:
                        check["status"] = item.get("guidance", item["status"])
                if item.get("reason") or item.get("disagreement"):
                    result["coaching"]["messages"].append(item["guidance"])
        result["score_100"] = assess100(result, exercise)
        result = unify_assessment(result, exercise, view, phase)
        if exercise == "shoulder":
            titles = dict(
                overhead_reach="양팔 상단 도달",
                lower_return="하단 복귀",
                bilateral_timing="양팔 타이밍",
                frontal_trunk_hold="몸통 좌우 안정",
            )
            source = {item["id"]: item for item in result["items"]}
            reasons = {
                "required_joints_not_simultaneously_observed_in_80_percent": "필요 관절의 동시 관측이 동작 구간의 80% 미만입니다.",
                "wrists_not_visible_at_overhead_peak": "상단에서 양손목을 확인하지 못했습니다.",
                "no_observed_lift_motion": "팔을 올린 구간을 확인하지 못했습니다.",
                "bilateral_peak_not_fully_visible": "양팔 상단 구간의 관측이 부족합니다.",
                "required_measurement_unavailable": "관측이 부족해 필요한 측정값을 계산하지 못했습니다.",
                "unsupported_model_view_bank": "이 항목의 현재 구도 모델이 없습니다.",
                "frontal_trunk_requires_front_view": "몸통 좌우 항목은 정면 구도가 필요합니다.",
            }
            for item in result["score_100"]["items"]:
                item["title"] = titles.get(item["id"], item["title"])
                for reason in source.get(item["id"], {}).get("reasons", []):
                    if (
                        reason in reasons
                        and reasons[reason] not in item["reasons"]
                    ):
                        item["reasons"].append(reasons[reason])
            result["score_100"][
                "scope"
            ] = "숄더 프레스 정면 관측의 항목별 실험 점수입니다. 미평가 항목을 채우거나 100점으로 환산하지 않습니다. 정상만 학습한 AI 항목의 오류 판별은 미검증입니다."
        return result


def finish_observation_window(samples, standing_reference, available):
    """Include up to .3 seconds of observed standing, never the next descent."""
    end = samples[-1][0]
    out = list(samples)
    for tick, points in available:
        if tick <= end:
            continue
        if tick > end + 3:
            break
        if (
            standing_reference is not None
            and np.isfinite(points[[23, 24]]).all()
        ):
            hip = points[[23, 24]].mean(0)
            drop = (
                hip[1] - standing_reference["points"][[23, 24]].mean(0)[1]
            ) / standing_reference["scale"]
            if drop > 0.09:
                break
        out.append((tick, points.copy()))
    return out
