"""측면 Forearm Plank의 유효 연속 유지 시간. Score/자세 오류 모델과는 별개입니다."""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from uuid import uuid4

from app.analysis.exercise import AnalysisResult, HoldCompleted


@dataclass(frozen=True)
class PlankConfig:
    min_body_alignment_deg: float = 155.0
    max_hip_deviation_ratio: float = 0.12
    max_body_tilt_deg: float = 35.0
    # 진입보다 넓은 유지 범위로 미세 흔들림에 의한 반복 이탈을 줄입니다.
    min_hold_alignment_deg: float = 145.0
    max_hold_body_tilt_deg: float = 40.0
    max_hold_hip_deviation_ratio: float = 0.18
    posture_grace_sec: float = 2.0
    min_visibility: float = 0.6
    min_presence: float = 0.5
    min_elbow_drop_ratio: float = 0.05
    max_elbow_drop_ratio: float = 0.45
    max_elbow_horizontal_ratio: float = 0.15
    min_elbow_angle_deg: float = 55.0
    max_elbow_angle_deg: float = 125.0
    max_forearm_vertical_ratio: float = 0.1
    stabilization_sec: float = 0.3
    stabilization_samples: int = 2
    occlusion_grace_sec: float = 0.5
    max_frame_gap_sec: float = 0.75

    def __post_init__(self):
        values = [
            v for k, v in vars(self).items() if k != "stabilization_samples"
        ]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values
        ):
            raise ValueError("플랭크 설정은 유한한 숫자여야 합니다.")
        if not (
            0 < self.min_body_alignment_deg <= 180
            and 0 < self.max_body_tilt_deg < 90
            and 0 < self.min_elbow_angle_deg < self.max_elbow_angle_deg < 180
            and 0 <= self.min_visibility <= 1
            and 0 <= self.min_presence <= 1
            and 0 < self.min_elbow_drop_ratio < self.max_elbow_drop_ratio
            and self.max_hip_deviation_ratio > 0
            and self.max_elbow_horizontal_ratio > 0
            and self.max_forearm_vertical_ratio > 0
            and self.stabilization_sec > 0
            and 0 < self.min_hold_alignment_deg <= self.min_body_alignment_deg
            and self.max_body_tilt_deg <= self.max_hold_body_tilt_deg < 90
            and self.max_hold_hip_deviation_ratio
            >= self.max_hip_deviation_ratio
            and self.posture_grace_sec > 0
            and type(self.stabilization_samples) is int
            and self.stabilization_samples >= 2
            and 0 < self.occlusion_grace_sec <= self.max_frame_gap_sec
        ):
            raise ValueError(
                "플랭크 임계값 또는 시간 설정이 올바르지 않습니다."
            )


_SIDES = {"left": (11, 13, 23, 27, 15), "right": (12, 14, 24, 28, 16)}


class PlankAnalyzer:
    def __init__(
        self,
        exercise_name="plank",
        exercise_type="hold",
        target_duration_sec=None,
        *,
        config=None,
    ):
        if exercise_name != "plank" or exercise_type != "hold":
            raise ValueError("plank/hold만 지원합니다.")
        if target_duration_sec is not None and (
            type(target_duration_sec) not in (int, float)
            or not math.isfinite(target_duration_sec)
            or target_duration_sec <= 0
        ):
            raise ValueError("목표 시간은 양의 유한한 초 값이어야 합니다.")
        self.target_duration_sec = target_duration_sec
        self.config = config if config is not None else PlankConfig()
        self._instance_id, self._completion_number = uuid4().hex, 0
        self._debug = os.getenv("PLANK_DEBUG") == "1"
        self._debug_last, self._debug_change = -math.inf, None
        self.reset()

    def _clear_segment(self):
        self.phase = "waiting"
        self.selected_side = None
        self.current_hold_sec = 0.0
        self._stable_since = None
        self._stable_samples = 0
        self._last_valid = None
        self._interrupted_since = None
        self._resume_holding = False
        self._posture_invalid_since = None
        self._occlusion_since = None
        self._last_joint_observation = None

    def reset(self):
        self._clear_segment()
        self._context = None
        self._last_timestamp = None
        self._last_frame_id = None

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _joint(self, points, ident):
        found = [
            p
            for p in points
            if isinstance(p, dict)
            and type(p.get("id")) is int
            and p["id"] == ident
        ]
        point = found[0] if len(found) == 1 else {}
        reason = "missing_or_duplicate" if len(found) != 1 else None
        if reason is None and any(
            not self._finite(point.get(a)) or not 0 <= point[a] <= 1
            for a in ("x", "y")
        ):
            reason = "coordinates_invalid"
        if reason is None and (
            not self._finite(point.get("visibility"))
            or not self.config.min_visibility <= point["visibility"] <= 1
        ):
            reason = "visibility_invalid_or_low"
        if (
            reason is None
            and point.get("presence") is not None
            and (
                not self._finite(point["presence"])
                or not self.config.min_presence <= point["presence"] <= 1
            )
        ):
            reason = "presence_invalid_or_low"
        return (point if reason is None else None), dict(
            visibility=point.get("visibility"),
            presence=point.get("presence"),
            invalid_reason=reason,
        )

    def _observe(self, points, side, size):
        ids = _SIDES[side]
        observed = [self._joint(points, i) for i in ids]
        joints = [p for p, _ in observed]
        detail = {
            "joints": {str(i): d for i, (_, d) in zip(ids, observed)},
            "joint_observation_valid": False,
        }
        if any(p is None for p in joints[:4]):
            detail.update(
                posture_valid=False,
                posture_invalid_reason="required_joint_invalid",
            )
            return detail
        width, height = size

        def pixel(p):
            return (p["x"] * width, p["y"] * height)

        def sub(a, b):
            return (a[0] - b[0], a[1] - b[1])

        def angle(a, b, c):
            u, v = sub(a, b), sub(c, b)
            norm = math.hypot(*u) * math.hypot(*v)
            if norm <= 1e-9:
                raise ValueError("degenerate_joint_vectors")
            return math.degrees(
                math.acos(
                    max(-1.0, min(1.0, sum(x * y for x, y in zip(u, v)) / norm))
                )
            )

        s, e, h, a = [pixel(p) for p in joints[:4]]
        axis = sub(a, s)
        length = math.hypot(*axis)
        if length <= 1e-6:
            detail.update(
                posture_valid=False,
                posture_invalid_reason="degenerate_body_axis",
            )
            return detail
        try:
            alignment = angle(s, h, a)
        except ValueError as error:
            detail.update(
                posture_valid=False, posture_invalid_reason=str(error)
            )
            return detail
        detail["joint_observation_valid"] = True
        # 화면 위쪽으로 향하는 수직축을 사용해 골반 상승/처짐을 구분합니다.
        normal = (-axis[1] / length, axis[0] / length)
        if normal[1] > 0:
            normal = (-normal[0], -normal[1])
        signed = sum(x * y for x, y in zip(sub(h, s), normal)) / length
        tilt = math.degrees(math.atan2(abs(axis[1]), abs(axis[0])))
        drop = (e[1] - s[1]) / length
        horizontal = abs(e[0] - s[0]) / length
        cfg = self.config
        reasons = []
        maintaining = (
            self.phase == "holding"
            or self.phase == "interrupted"
            and self._resume_holding
        )
        alignment_min = (
            cfg.min_hold_alignment_deg
            if maintaining
            else cfg.min_body_alignment_deg
        )
        hip_max = (
            cfg.max_hold_hip_deviation_ratio
            if maintaining
            else cfg.max_hip_deviation_ratio
        )
        tilt_max = (
            cfg.max_hold_body_tilt_deg if maintaining else cfg.max_body_tilt_deg
        )
        if abs(signed) > hip_max:
            reasons.append("hip_too_high" if signed > 0 else "hip_sagging")
        if alignment < alignment_min:
            reasons.append("body_alignment_below_minimum")
        if tilt > tilt_max:
            reasons.append("body_not_horizontal")
        detail["threshold_mode"] = "hold" if maintaining else "entry"
        if not cfg.min_elbow_drop_ratio <= drop <= cfg.max_elbow_drop_ratio:
            reasons.append("elbow_not_below_shoulder")
        if horizontal > cfg.max_elbow_horizontal_ratio:
            reasons.append("elbow_too_far_from_shoulder")
        elbow_angle = None
        forearm_vertical = None
        if joints[4] is not None:
            wrist = pixel(joints[4])
            forearm_vertical = abs(wrist[1] - e[1]) / length
            try:
                elbow_angle = angle(s, e, wrist)
            except ValueError:
                reasons.append("degenerate_forearm")
            if (
                elbow_angle is not None
                and not cfg.min_elbow_angle_deg
                <= elbow_angle
                <= cfg.max_elbow_angle_deg
            ):
                reasons.append("not_forearm_support")
            if forearm_vertical > cfg.max_forearm_vertical_ratio:
                reasons.append("forearm_not_horizontal")
        quality = min(
            min(
                p["visibility"],
                p.get("presence") if p.get("presence") is not None else 1.0,
            )
            for p in joints[:4]
        )
        detail.update(
            posture_valid=not reasons,
            posture_invalid_reason=reasons[0] if reasons else None,
            posture_failures=reasons,
            quality=quality,
            body_alignment_deg=alignment,
            hip_deviation_ratio=abs(signed),
            hip_signed_deviation_ratio=signed,
            body_tilt_deg=tilt,
            shoulder_elbow_drop_ratio=drop,
            shoulder_elbow_horizontal_ratio=horizontal,
            elbow_angle_deg=elbow_angle,
            forearm_vertical_ratio=forearm_vertical,
            wrist_observed=joints[4] is not None,
        )
        return detail

    def _result(self, ts, event=None):
        return AnalysisResult(
            "plank",
            "hold",
            self.phase,
            ts,
            target_duration_sec=self.target_duration_sec,
            hold_duration_sec=self.current_hold_sec,
            hold_completion_event=event,
        )

    def _interruption_expiry(self, ts):
        """자세 이탈과 관절 누락의 독립된 시계 중 먼저 만료되는 것을 적용합니다."""
        remaining = []
        if self._posture_invalid_since is not None:
            elapsed = ts - self._posture_invalid_since
            self._row["posture_invalid_elapsed_sec"] = elapsed
            remaining.append(self.config.posture_grace_sec - elapsed)
        if self._occlusion_since is not None:
            elapsed = ts - self._occlusion_since
            self._row["occlusion_elapsed_sec"] = elapsed
            remaining.append(self.config.occlusion_grace_sec - elapsed)
        self._row["grace_remaining_sec"] = (
            max(0.0, min(remaining)) if remaining else 0.0
        )
        if (
            self._occlusion_since is not None
            and ts - self._occlusion_since + 1e-9
            >= self.config.occlusion_grace_sec
        ):
            return "occlusion_timeout"
        if (
            self._posture_invalid_since is not None
            and ts - self._posture_invalid_since + 1e-9
            >= self.config.posture_grace_sec
        ):
            return "posture_grace_exceeded"
        return None

    def _analyze(self, points, context):
        if context != self._context:
            self.reset()
            self._context = context
            self._row["reset_reason"] = "context_changed"
        ts, size = getattr(points, "timestamp", None), getattr(
            points, "frame_size", None
        )
        if not self._finite(ts):
            self._clear_segment()
            self._row.update(
                reset_reason="invalid_timestamp",
                posture_invalid_reason="invalid_timestamp",
            )
            return self._result(time.monotonic())
        if self._last_timestamp is not None and ts <= self._last_timestamp:
            self._row["posture_invalid_reason"] = (
                "duplicate_timestamp"
                if ts == self._last_timestamp
                else "timestamp_reversed"
            )
            return self._result(ts)
        ident = getattr(points, "frame_id", None)
        if ident is not None and ident == self._last_frame_id:
            self._row["posture_invalid_reason"] = "duplicate_frame_id"
            return self._result(ts)
        gap = (
            ts - self._last_timestamp
            if self._last_timestamp is not None
            else None
        )
        self._last_timestamp, self._last_frame_id = ts, ident
        self._row["frame_gap_sec"] = gap
        if self.phase == "completed":
            return self._result(ts)
        if gap is not None and gap > self.config.max_frame_gap_sec:
            self._clear_segment()
            self._row["reset_reason"] = "frame_gap_exceeded"
        valid_metadata = (
            getattr(context, "exercise_name", None) == "plank"
            and isinstance(size, (tuple, list))
            and len(size) == 2
            and all(self._finite(v) and v > 0 for v in size)
        )
        if valid_metadata:
            if self.selected_side is None:
                choices = [
                    (side, self._observe(points, side, size)) for side in _SIDES
                ]
                valid = [item for item in choices if item[1]["posture_valid"]]
                side, detail = max(
                    valid or choices,
                    key=lambda item: item[1].get("quality", -1),
                )
                if valid:
                    self.selected_side = side
            else:
                detail = self._observe(points, self.selected_side, size)
        else:
            detail = dict(
                posture_valid=False,
                posture_invalid_reason="invalid_context_or_frame_size",
            )
        self._row.update(detail)
        previous_joint_timestamp = self._last_joint_observation
        if detail.get("joint_observation_valid"):
            self._last_joint_observation = ts
        if not detail["posture_valid"]:
            if self.phase in ("holding", "stabilizing"):
                self._resume_holding = self.phase == "holding"
                self.phase = "interrupted"
                self._interrupted_since = ts
            self._stable_since = None
            self._stable_samples = 0
            if self.phase == "interrupted":
                if detail.get("joint_observation_valid"):
                    if self._posture_invalid_since is None:
                        self._posture_invalid_since = self._interrupted_since
                elif self._occlusion_since is None:
                    self._occlusion_since = (
                        previous_joint_timestamp
                        if previous_joint_timestamp is not None
                        else ts
                    )
                expired = self._interruption_expiry(ts)
                if expired:
                    self._clear_segment()
                    self._row["reset_reason"] = expired
                elif detail.get("joint_observation_valid"):
                    self._occlusion_since = None
            return self._result(ts)
        if self.phase == "interrupted":
            expired = self._interruption_expiry(ts)
            if expired:
                self._clear_segment()
                self._row["reset_reason"] = expired
                return self._result(
                    ts
                )  # 다음 새 관측부터 안정화하며 반대 측면으로 기존 구간을 이어 세지 않습니다.
            if self._resume_holding:
                self.phase = "holding"
                self._last_valid = ts
                self._interrupted_since = None
                self._posture_invalid_since = None
                self._occlusion_since = None
                self._row["hold_resumed"] = True
                return self._result(ts)  # 무효·복구 구간은 더하지 않습니다.
            self.phase = "waiting"
            self._interrupted_since = None
            self._posture_invalid_since = None
            self._occlusion_since = None
        if self.phase == "waiting":
            self.phase = "stabilizing"
            self._stable_since = ts
            self._stable_samples = 1
        elif self.phase == "stabilizing":
            self._stable_samples += 1
            if (
                self._stable_samples >= self.config.stabilization_samples
                and ts - self._stable_since + 1e-9
                >= self.config.stabilization_sec
            ):
                self.phase = "holding"
        elif self.phase == "holding":
            self.current_hold_sec += ts - self._last_valid
        self._last_valid = ts
        event = None
        if (
            self.target_duration_sec is not None
            and self.current_hold_sec + 1e-9 >= self.target_duration_sec
        ):
            self.current_hold_sec = self.target_duration_sec
            self.phase = "completed"
            self._completion_number += 1
            event = HoldCompleted(
                f"plank:{self._instance_id}:{self._completion_number}"
            )
        return self._result(ts, event)

    def analyze(self, keypoints, context):
        previous = self.phase
        self._row = dict(
            posture_valid=False,
            posture_invalid_reason=None,
            frame_gap_sec=None,
            occlusion_elapsed_sec=0.0,
            reset_reason=None,
            posture_invalid_elapsed_sec=0.0,
            grace_remaining_sec=0.0,
            hold_resumed=False,
        )
        result = self._analyze(keypoints, context)
        self._row["hold_reset_reason"] = self._row["reset_reason"]
        if self._debug:
            try:
                now = time.monotonic()
                change = (
                    self.phase,
                    self._row["posture_invalid_reason"],
                    self._row["reset_reason"],
                )
                if (
                    previous != self.phase
                    or result.hold_completion_event
                    or change != self._debug_change
                    or now - self._debug_last >= 1.0
                ):

                    def safe(v):
                        if isinstance(v, float) and not math.isfinite(v):
                            return str(v)
                        if isinstance(v, dict):
                            return {k: safe(x) for k, x in v.items()}
                        if isinstance(v, list):
                            return [safe(x) for x in v]
                        return v

                    logging.getLogger("uvicorn.error").info(
                        "[PlankDebug] %s",
                        json.dumps(
                            safe(
                                dict(
                                    self._row,
                                    selected_side=self.selected_side,
                                    current_phase=self.phase,
                                    timestamp=result.timestamp,
                                    current_hold_sec=self.current_hold_sec,
                                    target_duration_sec=self.target_duration_sec,
                                    completed=self.phase == "completed",
                                    completion_event_id=(
                                        result.hold_completion_event.event_id
                                        if result.hold_completion_event
                                        else None
                                    ),
                                    thresholds=asdict(self.config),
                                )
                            ),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    )
                    self._debug_last, self._debug_change = now, change
            except Exception:
                pass
        return result
