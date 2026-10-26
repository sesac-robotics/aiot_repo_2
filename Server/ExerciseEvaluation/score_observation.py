"""실제 관절의 품질·준비 상태·항목별 관측 조건을 수치로 확인합니다."""

import numpy as np

JOINTS = {
    11: "왼쪽 어깨",
    12: "오른쪽 어깨",
    23: "왼쪽 골반",
    24: "오른쪽 골반",
    25: "왼쪽 무릎",
    26: "오른쪽 무릎",
}
REQUIRED = {
    "descent_range": [11, 12, 23, 24, 25, 26],
    "standing_return": [11, 12, 23, 24],
    "knee_inward": [23, 24, 25, 26],
    "trunk_balance": [11, 12, 23, 24],
}
VERSION = "score-observation-v2"


def inspect_pose(raw, width, height):
    """raw 순서는 정규화 x/y/z/visibility/presence이며 기준은 기존과 같습니다."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (33, 5):
        raise ValueError("Expected 33 x 5 raw landmarks")
    pixel = raw[:, :2] * [width, height]
    xy = pixel.copy()
    reasons = []
    for j in range(33):
        why = []
        if not np.isfinite(raw[j]).all():
            why.append("not_detected")
        else:
            if raw[j, 3] < 0.5:
                why.append("low_visibility")
            if raw[j, 4] < 0.5:
                why.append("low_presence")
            if np.any(pixel[j] < [0, 0]) or np.any(pixel[j] >= [width, height]):
                why.append("outside_frame")
            elif np.any(pixel[j] < np.array([width, height]) * 0.02) or np.any(
                pixel[j] > np.array([width, height]) * 0.98
            ):
                why.append("frame_border")
        if why:
            xy[j] = np.nan
        reasons.append(why)
    valid = np.isfinite(xy).all(1)
    targets = {
        key: bool(valid[indices].all()) for key, indices in REQUIRED.items()
    }
    return xy, dict(
        version=VERSION,
        item_observed=targets,
        all_items_observed=all(targets.values()),
        joint_rejection_reasons=reasons,
        joints={
            str(j): dict(
                name=name,
                valid=bool(valid[j]),
                visibility=float(raw[j, 3]) if np.isfinite(raw[j, 3]) else None,
                presence=float(raw[j, 4]) if np.isfinite(raw[j, 4]) else None,
                pixel_xy=(
                    pixel[j].tolist() if np.isfinite(pixel[j]).all() else None
                ),
                reasons=reasons[j],
            )
            for j, name in JOINTS.items()
        },
    )


def choose_recovery(primary, retry, width, height):
    """더 낮은 기준이나 관절 혼합 없이, 같은 프레임의 완전 관측 자세만 채택합니다."""
    original_xy, original = inspect_pose(primary, width, height)
    if original["all_items_observed"] or retry is None:
        return primary, "video", original_xy, original
    retry_xy, recovered = inspect_pose(retry, width, height)
    if not recovered["all_items_observed"]:
        return primary, "video", original_xy, original
    core = [11, 12, 23, 24]
    if np.isfinite(original_xy[core]).all():
        scale = max(
            30.0,
            float(
                np.linalg.norm(
                    original_xy[[11, 12]].mean(0)
                    - original_xy[[23, 24]].mean(0)
                )
            ),
        )
        if (
            np.max(np.linalg.norm(retry_xy[core] - original_xy[core], axis=1))
            > 0.25 * scale
        ):
            return primary, "video", original_xy, original
    return retry, "independent_image_recovery", retry_xy, recovered


class ScoringReadyGate:
    def __init__(self):
        self.streak = 0
        self.last_tick = None
        self.prepared = False

    def step(self, tick, quality):
        consecutive = self.last_tick is not None and tick == self.last_tick + 1
        self.last_tick = tick
        if quality["all_items_observed"]:
            self.streak = self.streak + 1 if consecutive else 1
        else:
            self.streak = 0
        self.prepared = self.prepared or self.streak >= 6
        return dict(
            prepared=self.prepared,
            ready_frames=min(self.streak, 6),
            required_frames=6,
            current_all_items_observed=quality["all_items_observed"],
            message=(
                "전체 항목 관절 준비 완료"
                if self.streak >= 6
                else "양쪽 무릎·골반·어깨가 화면 안쪽에서 함께 추적되도록 서십시오."
            ),
        )


def segment_observation(samples, width, height):
    """점수 계산과 동일한 시간 슬롯·경계·최소 표본·초기 참조 조건입니다."""
    from live_observation import mask_frame_border

    ticks = np.arange(samples[0][0], samples[-1][0] + 1)
    xy = np.full((len(ticks), 33, 2), np.nan)
    for tick, points in samples:
        xy[tick - ticks[0]] = points
    xy = mask_frame_border(xy, width, height)
    prefix = max(3, min(10, int(np.ceil(len(xy) * 0.2))))
    items = {}
    for key, joints in REQUIRED.items():
        mask = np.isfinite(xy[:, joints]).all((1, 2))
        count, initial = int(mask.sum()), int(mask[:prefix].sum())
        reasons = []
        if count < 10:
            reasons.append("fewer_than_10_observed_frames")
        if mask.mean() < 0.8:
            reasons.append("simultaneous_coverage_below_80_percent")
        if initial < 3:
            reasons.append("initial_reference_fewer_than_3_frames")
        items[key] = dict(
            eligible=not reasons,
            observed_frames=count,
            total_slots=len(ticks),
            coverage=float(mask.mean()),
            reference_frames=initial,
            reference_window_slots=prefix,
            reasons=reasons,
            per_joint_coverage={
                str(j): float(np.isfinite(xy[:, j]).all(1).mean())
                for j in joints
            },
        )
    return dict(
        version=VERSION,
        items=items,
        all_items_eligible=all(i["eligible"] for i in items.values()),
    )


def retry_due(raw, stamp, last_retry):
    """새 실제 슬롯의 추적 실패는 즉시 재검출하고 빈 영상만 제한합니다."""
    return bool(np.isfinite(raw).any()) or stamp - last_retry >= 0.5
