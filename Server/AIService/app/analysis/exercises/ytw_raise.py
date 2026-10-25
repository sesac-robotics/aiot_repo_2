"""정면 Y→T→W 전체를 1회로 측정합니다. 점수·자세 오류는 계산하지 않습니다."""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from uuid import uuid4

from app.analysis.exercise import AnalysisResult, RepCompleted


@dataclass(frozen=True)
class YTWRaiseConfig:
    extended_elbow_min_deg: float = 140.0
    y_min_direction_deg: float = 30.0
    y_max_direction_deg: float = 85.0
    t_max_direction_deg: float = 28.0
    w_min_elbow_deg: float = 55.0
    w_max_elbow_deg: float = 135.0
    # 위치는 각 어깨 기준, 어깨 폭 단위입니다. 위쪽과 각 팔의 바깥쪽이 양수입니다.
    min_extended_outward: float = 0.45
    w_min_elbow_outward: float = 0.08
    w_min_elbow_height: float = -0.85
    w_max_elbow_height: float = 0.25
    w_min_wrist_above_elbow: float = 0.15
    w_min_wrist_height: float = -0.1
    w_max_wrist_height: float = 1.4
    min_visibility: float = 0.6
    min_presence: float = 0.5
    transition_duration_sec: float = 0.10
    transition_samples: int = 2
    occlusion_grace_sec: float = 0.6
    max_frame_gap_sec: float = 0.75
    max_cycle_duration_sec: float = 12.0

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != "transition_samples"]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values
        ):
            raise ValueError("YTW 설정은 유한한 숫자여야 합니다.")
        if not (
            0
            < self.t_max_direction_deg
            < self.y_min_direction_deg
            < self.y_max_direction_deg
            < 90
            and 0
            < self.w_min_elbow_deg
            < self.w_max_elbow_deg
            < self.extended_elbow_min_deg
            <= 180
        ):
            raise ValueError("YTW 각도 범위가 올바르지 않습니다.")
        if not (
            self.min_extended_outward > 0
            and self.w_min_elbow_outward > 0
            and self.w_min_elbow_height < self.w_max_elbow_height
            and self.w_min_wrist_height < self.w_max_wrist_height
            and self.w_min_wrist_above_elbow > 0
            and 0 <= self.min_visibility <= 1
            and 0 <= self.min_presence <= 1
        ):
            raise ValueError("YTW 위치 또는 신뢰도 설정이 올바르지 않습니다.")
        if (
            type(self.transition_samples) is not int
            or self.transition_samples < 2
            or self.transition_duration_sec <= 0
            or not 0
            < self.occlusion_grace_sec
            <= self.max_frame_gap_sec
            < self.max_cycle_duration_sec
        ):
            raise ValueError("YTW 안정화/시간 설정이 올바르지 않습니다.")


class YTWRaiseAnalyzer:
    def __init__(
        self,
        exercise_name="ytw_raise",
        exercise_type="repetition",
        target_duration_sec=None,
        *,
        config=None,
    ):
        if (
            exercise_name != "ytw_raise"
            or exercise_type != "repetition"
            or target_duration_sec is not None
        ):
            raise ValueError(
                "YTWRaiseAnalyzer는 ytw_raise/repetition 설정만 지원합니다."
            )
        self.config = config if config is not None else YTWRaiseConfig()
        self._instance_id, self._event_number = uuid4().hex, 0
        self._debug = os.getenv("YTW_DEBUG") == "1"
        self._debug_last, self._debug_change = -math.inf, None
        self._debug_joints = {}
        self.reset()

    def _clear_sequence(self):
        if self._debug:
            self._debug_reset_occurred = True
        self.phase = "wait_y"
        self._candidate = self._candidate_since = None
        self._candidate_samples = 0
        self._cycle_started = None

    def reset(self):
        # 같은 인스턴스의 이벤트 번호는 reset/관절 손실 후에도 재사용하지 않습니다.
        previous, context = getattr(self, "phase", None), getattr(
            self, "_context", None
        )
        self._clear_sequence()
        self._context = self._last_timestamp = self._last_valid_timestamp = None
        self._missing_since = self._last_fingerprint = None
        self._joint_history = {}
        self._cycle_discard_reason = None
        if self._debug and previous is not None:
            self._emit_debug(
                dict(
                    kind="reset",
                    previous_phase=previous,
                    current_phase=self.phase,
                    next_phase=self.phase,
                    reset_reason=getattr(self, "_debug_reset_reason", None)
                    or "external_reset",
                    session_id=getattr(context, "session_id", None),
                    exercise_id=getattr(context, "exercise_id", None),
                ),
                force=True,
            )

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _joint(self, points, ident, *, required):
        found = [
            p
            for p in points
            if isinstance(p, dict)
            and type(p.get("id")) is int
            and p["id"] == ident
        ]
        reason = "missing_or_duplicate" if len(found) != 1 else None
        point = found[0] if reason is None else {}
        if reason is None:
            if any(
                not self._finite(point.get(a)) or not 0 <= point[a] <= 1
                for a in ("x", "y")
            ):
                reason = "coordinates_invalid"
            elif (
                not self._finite(point.get("visibility"))
                or not self.config.min_visibility <= point["visibility"] <= 1
            ):
                reason = "visibility_invalid_or_low"
            elif point.get("presence") is not None and (
                not self._finite(point["presence"])
                or not self.config.min_presence <= point["presence"] <= 1
            ):
                reason = "presence_invalid_or_low"
        if self._debug:
            # 실제 _joint 검증 결과를 재사용하며 별도 유효성 판정식을 만들지 않습니다.
            self._debug_joints[str(ident)] = dict(
                visibility=point.get("visibility"),
                presence=point.get("presence"),
                valid=reason is None,
                invalid_reason=reason,
                optional=ident in (23, 24),
            )
        if reason:
            if required:
                raise ValueError(f"joint_{ident}_{reason}")
            return None
        return point

    def _observe(self, points, size):
        joints = {
            i: self._joint(points, i, required=True) for i in range(11, 17)
        }
        hips = [self._joint(points, i, required=False) for i in (23, 24)]
        width, height = size

        def pixel(point):
            return (point["x"] * width, point["y"] * height)

        def subtract(a, b):
            return (a[0] - b[0], a[1] - b[1])

        def unit(vector):
            length = math.hypot(*vector)
            if length <= 1e-6:
                raise ValueError("degenerate_body_axis")
            return (vector[0] / length, vector[1] / length)

        shoulders = [pixel(joints[i]) for i in (11, 12)]
        across = subtract(shoulders[1], shoulders[0])
        shoulder_width = math.hypot(*across)
        if shoulder_width <= 1e-6:
            raise ValueError("degenerate_shoulder_width")
        center = tuple((a + b) / 2 for a, b in zip(*shoulders))
        if all(hip is not None for hip in hips):
            hip_center = tuple(
                (a + b) / 2 for a, b in zip(*(pixel(p) for p in hips))
            )
            up = unit(subtract(center, hip_center))
            axis_source = "shoulders_and_hips"
        else:
            # 골반이 보이지 않으면 어깨 선의 수직축을 영상 위쪽으로 향하게 합니다.
            up = unit((across[1], -across[0]))
            if up[1] > 0:
                up = (-up[0], -up[1])
            axis_source = "shoulders_only"
        outward = (-up[1], up[0])
        if sum(a * b for a, b in zip(outward, across)) < 0:
            outward = (-outward[0], -outward[1])
        if (
            abs(sum(a * b for a, b in zip(outward, across))) / shoulder_width
            < 0.1
        ):
            raise ValueError("body_axis_parallel_to_shoulders")
        cfg = self.config
        features = {}
        for side, sign, ids in (
            ("left", -1, (11, 13, 15)),
            ("right", 1, (12, 14, 16)),
        ):
            shoulder, elbow, wrist = [pixel(joints[i]) for i in ids]
            first, second = subtract(shoulder, elbow), subtract(wrist, elbow)
            norm = math.hypot(*first) * math.hypot(*second)
            if norm <= 1e-9:
                raise ValueError(f"{side}_degenerate_elbow_vectors")
            cosine = sum(a * b for a, b in zip(first, second)) / norm
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

            def position(point):
                vector = subtract(point, shoulder)
                return (
                    sign
                    * sum(a * b for a, b in zip(vector, outward))
                    / shoulder_width,
                    sum(a * b for a, b in zip(vector, up)) / shoulder_width,
                )

            ex, ey = position(elbow)
            wx, wy = position(wrist)
            direction = math.degrees(math.atan2(wy, wx))
            extended = (
                angle >= cfg.extended_elbow_min_deg
                and wx >= cfg.min_extended_outward
            )
            y = (
                extended
                and cfg.y_min_direction_deg
                <= direction
                <= cfg.y_max_direction_deg
            )
            t = extended and abs(direction) <= cfg.t_max_direction_deg
            w = (
                cfg.w_min_elbow_deg <= angle <= cfg.w_max_elbow_deg
                and ex >= cfg.w_min_elbow_outward
                and cfg.w_min_elbow_height <= ey <= cfg.w_max_elbow_height
                and wy - ey >= cfg.w_min_wrist_above_elbow
                and cfg.w_min_wrist_height <= wy <= cfg.w_max_wrist_height
            )
            features[side] = dict(
                elbow_angle_deg=angle,
                direction_deg=direction,
                wrist_outward=wx,
                wrist_height=wy,
                elbow_outward=ex,
                elbow_height=ey,
                wrist_above_elbow=wy - ey,
                y=y,
                t=t,
                w=w,
            )
            if self._debug:
                # 측정값과 함께 어느 결합 조건이 실패했는지 표시하며 분류에는 재사용하지 않습니다.
                shared = (
                    []
                    if angle >= cfg.extended_elbow_min_deg
                    else ["elbow_not_extended"]
                ) + (
                    []
                    if wx >= cfg.min_extended_outward
                    else ["wrist_not_outward"]
                )
                features[side]["failures"] = {
                    "y": shared
                    + (
                        []
                        if cfg.y_min_direction_deg
                        <= direction
                        <= cfg.y_max_direction_deg
                        else ["direction_outside_y_range"]
                    ),
                    "t": shared
                    + (
                        []
                        if abs(direction) <= cfg.t_max_direction_deg
                        else ["direction_outside_t_range"]
                    ),
                    "w": [
                        reason
                        for met, reason in (
                            (
                                cfg.w_min_elbow_deg
                                <= angle
                                <= cfg.w_max_elbow_deg,
                                "elbow_outside_w_range",
                            ),
                            (
                                ex >= cfg.w_min_elbow_outward,
                                "elbow_not_outward",
                            ),
                            (
                                cfg.w_min_elbow_height
                                <= ey
                                <= cfg.w_max_elbow_height,
                                "elbow_height_outside_w_range",
                            ),
                            (
                                wy - ey >= cfg.w_min_wrist_above_elbow,
                                "wrist_not_above_elbow",
                            ),
                            (
                                cfg.w_min_wrist_height
                                <= wy
                                <= cfg.w_max_wrist_height,
                                "wrist_height_outside_w_range",
                            ),
                        )
                        if not met
                    ],
                }
        if self._debug:
            self._row.update(
                shoulder_width_px=shoulder_width,
                body_up_axis=up,
                body_outward_axis=outward,
            )
        detected = next(
            (
                pose
                for pose in ("y", "t", "w")
                if all(f[pose] for f in features.values())
            ),
            None,
        )
        # 동일 좌표·신뢰도의 재전달도 표본 수를 늘리지 않습니다. 관측 가능한 입력만 비교합니다.
        observed = list(joints.values()) + [p for p in hips if p is not None]
        fingerprint = tuple(
            tuple(p.get(k) for k in ("id", "x", "y", "visibility", "presence"))
            for p in observed
        ) + (tuple(size),)
        return detected, features, fingerprint, axis_source

    def _track_joint_observations(self, points, timestamp):
        # 좌표는 제한 시간 동안 진단/복구 추적용으로만 보관하며 _observe 입력에 합성하지 않습니다.
        for ident in (*range(11, 17), 23, 24):
            point = self._joint(points, ident, required=False)
            history = self._joint_history.setdefault(
                ident,
                dict(
                    last_valid_timestamp=None,
                    last_valid_coordinates=None,
                    invalid_since=None,
                    recovered=False,
                ),
            )
            history["recovered"] = (
                point is not None and history["invalid_since"] is not None
            )
            if point is not None:
                history.update(
                    last_valid_timestamp=timestamp,
                    last_valid_coordinates={k: point[k] for k in ("x", "y")},
                    invalid_since=None,
                )
            else:
                if history["invalid_since"] is None:
                    history["invalid_since"] = timestamp
                last = history["last_valid_timestamp"]
                if (
                    last is None
                    or timestamp - last + 1e-9
                    >= self.config.occlusion_grace_sec
                ):
                    history["last_valid_coordinates"] = None

    def _discard_cycle_once(self, reason):
        # 동일 가림/중복 관측 구간에서 매 프레임 재초기화·강제 출력하지 않습니다.
        if self._cycle_discard_reason is None:
            self._clear_sequence()
            self._cycle_discard_reason = reason
            self._row["reset_reason"] = reason

    def _stable(self, candidate, timestamp):
        if candidate is None:
            self._candidate = self._candidate_since = None
            self._candidate_samples = 0
            return False
        if candidate != self._candidate:
            self._candidate, self._candidate_since, self._candidate_samples = (
                candidate,
                timestamp,
                1,
            )
        else:
            self._candidate_samples += 1
        return (
            self._candidate_samples >= self.config.transition_samples
            and timestamp - self._candidate_since + 1e-9
            >= self.config.transition_duration_sec
        )

    def _analyze(self, points, context):
        if context != self._context:
            self.reset()
            self._context = context
        ts, size = getattr(points, "timestamp", None), getattr(
            points, "frame_size", None
        )
        if (
            getattr(context, "exercise_name", None) != "ytw_raise"
            or not self._finite(ts)
            or not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(not self._finite(v) or v <= 0 for v in size)
        ):
            self._clear_sequence()
            self._last_fingerprint = None
            return self._result(
                ts if self._finite(ts) else time.monotonic(),
                invalid="invalid_context_or_metadata",
            )
        if self._last_timestamp is not None and ts <= self._last_timestamp:
            if ts < self._last_timestamp:
                self._clear_sequence()
                return self._result(ts, invalid="timestamp_reversed")
            return self._result(ts, reason="duplicate_timestamp")
        self._last_timestamp = ts
        self._track_joint_observations(points, ts)
        if (
            self._last_valid_timestamp is not None
            and ts - self._last_valid_timestamp > self.config.max_frame_gap_sec
        ):
            self._discard_cycle_once("frame_gap_exceeded")
        if (
            self._cycle_started is not None
            and ts - self._cycle_started > self.config.max_cycle_duration_sec
        ):
            self._discard_cycle_once("cycle_timeout")
        try:
            detected, features, fingerprint, axis_source = self._observe(
                points, size
            )
        except ValueError as error:
            self._stable(None, ts)
            self._last_fingerprint = None
            if self._missing_since is None:
                self._missing_since = (
                    self._last_valid_timestamp
                    if self._last_valid_timestamp is not None
                    else ts
                )
            if (
                ts - self._missing_since + 1e-9
                >= self.config.occlusion_grace_sec
            ):
                self._discard_cycle_once("occlusion_timeout")
            return self._result(ts, invalid=str(error))
        self._row.update(
            detected_pose=detected, arms=features, axis_source=axis_source
        )
        if (
            self._missing_since is not None
            and ts - self._missing_since + 1e-9
            >= self.config.occlusion_grace_sec
        ):
            self._discard_cycle_once("occlusion_timeout")
        self._missing_since = None
        if fingerprint == self._last_fingerprint:
            return self._result(ts, reason="duplicate_landmarks")
        self._last_fingerprint, self._last_valid_timestamp = fingerprint, ts
        self._row["cycle_discard_reason"] = self._cycle_discard_reason
        self._cycle_discard_reason = None
        # 한쪽만 다음 자세에 도달한 시차는 보류하되 누락된 자세를 추정하지 않습니다.
        candidate = detected
        if self._debug:
            self._row["decision_phase"] = self.phase
        if (
            (self.phase == "y_confirmed" and detected == "y")
            or (self.phase == "t_confirmed" and detected == "t")
            or (self.phase in ("wait_y", "wait_next_y") and detected != "y")
        ):
            candidate = None
        stable = self._stable(candidate, ts)
        self._row.update(
            candidate=candidate,
            candidate_samples=self._candidate_samples,
            candidate_duration_sec=(
                ts - self._candidate_since
                if self._candidate_since is not None
                else 0.0
            ),
            transition_stable=stable,
        )
        event = None
        reason = (
            "stabilizing"
            if candidate and not stable
            else "waiting_for_expected_pose"
        )
        if stable:
            if detected == "y":
                self.phase, self._cycle_started = (
                    "y_confirmed",
                    self._candidate_since,
                )
            elif detected == "t" and self.phase == "y_confirmed":
                self.phase = "t_confirmed"
            elif detected == "w" and self.phase == "t_confirmed":
                self._event_number += 1
                event = RepCompleted(
                    f"ytw_raise:{self._instance_id}:{self._event_number}"
                )
                self.phase, self._cycle_started = "wait_next_y", None
            else:
                self._clear_sequence()
                reason = "wrong_pose_order"
            self._stable(None, ts)
        return self._result(ts, event=event, reason=reason)

    def _result(self, ts, *, invalid=None, event=None, reason=None):
        self._row.update(
            input_valid=invalid is None, invalid_reason=invalid, reason=reason
        )
        self._row.setdefault("candidate", self._candidate)
        self._row.setdefault("candidate_samples", self._candidate_samples)
        self._row.setdefault(
            "candidate_duration_sec",
            (
                ts - self._candidate_since
                if self._finite(ts) and self._candidate_since is not None
                else 0.0
            ),
        )
        return AnalysisResult(
            "ytw_raise",
            "repetition",
            "invalid" if invalid else self.phase,
            ts,
            repetition_event=event,
        )

    def _emit_debug(self, payload, *, force=False, change=None):
        # JSON 변환/출력 오류는 상태 머신이나 이벤트에 영향을 주지 않습니다.
        if not self._debug:
            return
        try:
            now = time.monotonic()
            if (
                not force
                and change == self._debug_change
                and now - self._debug_last < 1.0
            ):
                return

            def safe(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                if isinstance(value, dict):
                    return {k: safe(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [safe(v) for v in value]
                return value

            logging.getLogger("uvicorn.error").info(
                "[YTWDebug] %s",
                json.dumps(
                    safe(payload),
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                ),
            )
            self._debug_last, self._debug_change = now, change
        except Exception:
            pass

    def _pose_diagnostics(self):
        # 이미 분류에 사용한 수치와 실패 목록을 표시합니다. 추가 분류/보정은 수행하지 않습니다.
        arms = self._row.get("arms", {})
        checks = {}
        for pose in ("y", "t", "w"):
            sides = {
                side: arms.get(side, {}).get(pose) for side in ("left", "right")
            }
            failures = {
                side: arms.get(side, {}).get("failures", {}).get(pose)
                for side in sides
            }
            checks[pose] = dict(
                **sides,
                recognized=(
                    all(sides.values())
                    if all(v is not None for v in sides.values())
                    else None
                ),
                failures=failures,
                conditions={},
            )
            for side, reasons in failures.items():
                if reasons is None:
                    checks[pose]["conditions"][side] = None
                elif pose in ("y", "t"):
                    checks[pose]["conditions"][side] = dict(
                        elbow_extended="elbow_not_extended" not in reasons,
                        wrist_outward="wrist_not_outward" not in reasons,
                        direction="direction_outside_" + pose + "_range"
                        not in reasons,
                    )
                else:
                    checks[pose]["conditions"][side] = dict(
                        elbow_bent="elbow_outside_w_range" not in reasons,
                        elbow_outward="elbow_not_outward" not in reasons,
                        elbow_height="elbow_height_outside_w_range"
                        not in reasons,
                        wrist_above_elbow="wrist_not_above_elbow"
                        not in reasons,
                        wrist_height="wrist_height_outside_w_range"
                        not in reasons,
                    )
        matches = [p for p, check in checks.items() if check["recognized"]]
        per_arm = {
            side: [p for p in checks if checks[p][side]]
            for side in ("left", "right")
        }
        # 좌우 차이는 동일 정규화 결과의 진단값이며 현재 판정에 대칭성 임계값은 없습니다.
        symmetry = {"used_as_gate": False}
        if all(side in arms for side in ("left", "right")):
            for field in (
                "elbow_angle_deg",
                "direction_deg",
                "wrist_height",
                "elbow_height",
            ):
                symmetry[field + "_difference"] = abs(
                    arms["left"][field] - arms["right"][field]
                )
        return checks, matches, per_arm, symmetry

    def analyze(self, keypoints, context):
        previous = self.phase
        context_changed = context != self._context
        previous_timestamp, previous_valid = (
            self._last_timestamp,
            self._last_valid_timestamp,
        )
        self._row = dict(detected_pose=None)
        if self._debug:
            self._debug_joints = {}
            self._debug_reset_occurred = False
            self._debug_reset_reason = (
                "context_changed" if context_changed else None
            )
        result = self._analyze(keypoints, context)
        if self._debug:
            try:
                self._debug_reset_reason = None
                # 선행 필수 관절 실패로 검사하지 못한 나머지 관절도 같은 _joint 경로로 진단합니다.
                for ident in (*range(11, 17), 23, 24):
                    if str(ident) not in self._debug_joints:
                        self._joint(keypoints, ident, required=False)
                ts = getattr(keypoints, "timestamp", None)
                if self._finite(ts):
                    for ident, history in self._joint_history.items():
                        last = history["last_valid_timestamp"]
                        since = history["invalid_since"]
                        self._debug_joints[str(ident)].update(
                            last_valid_timestamp=last,
                            last_valid_coordinates=history[
                                "last_valid_coordinates"
                            ],
                            last_valid_age_sec=(
                                ts - last if last is not None else None
                            ),
                            continuous_invalid_duration_sec=(
                                ts - since if since is not None else 0.0
                            ),
                            recovered=history["recovered"],
                        )
                checks, matches, per_arm, symmetry = self._pose_diagnostics()
                decision_phase = self._row.get("decision_phase", self.phase)
                expected = {
                    "wait_y": "y",
                    "y_confirmed": "t",
                    "t_confirmed": "w",
                    "wait_next_y": "y",
                }[decision_phase]
                reset_reason = self._row.get("reset_reason")
                if self._debug_reset_occurred and reset_reason is None:
                    reset_reason = (
                        "context_changed"
                        if context_changed
                        else self._row.get("invalid_reason")
                        or self._row.get("reason")
                        or "sequence_reset"
                    )
                failures = []
                if self._row.get("invalid_reason"):
                    failures.append("joint_or_metadata_invalid")
                if reset_reason:
                    failures.append(reset_reason)
                reason = self._row.get("reason")
                if reason in (
                    "duplicate_timestamp",
                    "duplicate_landmarks",
                    "wrong_pose_order",
                ):
                    failures.append(reason)
                if (
                    self._row.get("candidate")
                    and not self._row.get("transition_stable")
                    and reason
                    not in ("duplicate_timestamp", "duplicate_landmarks")
                ):
                    if (
                        self._row["candidate_samples"]
                        < self.config.transition_samples
                    ):
                        failures.append("insufficient_distinct_samples")
                    if (
                        self._row["candidate_duration_sec"] + 1e-9
                        < self.config.transition_duration_sec
                    ):
                        failures.append("insufficient_stability_duration")
                if (
                    not self._row.get("invalid_reason")
                    and self._row.get("detected_pose") != expected
                ):
                    failures.append(
                        "waiting_for_rearm_y"
                        if decision_phase == "wait_next_y"
                        else "waiting_for_" + expected
                    )
                if len(matches) > 1 or any(
                    len(poses) > 1 for poses in per_arm.values()
                ):
                    failures.append("pose_condition_overlap")
                ts = getattr(keypoints, "timestamp", None)
                event = result.repetition_event
                payload = dict(
                    self._row,
                    kind="analyzer",
                    analyzer_called=True,
                    frame_id=getattr(keypoints, "frame_id", None),
                    session_id=getattr(context, "session_id", None),
                    exercise_id=getattr(context, "exercise_id", None),
                    timestamp=result.timestamp,
                    previous_phase=previous,
                    current_phase=result.phase,
                    phase=result.phase,
                    next_phase=self.phase,
                    candidate_pose=self._row.get("candidate"),
                    transition_stable=self._row.get("transition_stable", False),
                    frame_gap_sec=(
                        ts - previous_timestamp
                        if self._finite(ts) and previous_timestamp is not None
                        else None
                    ),
                    valid_observation_gap_sec=(
                        ts - previous_valid
                        if self._finite(ts) and previous_valid is not None
                        else None
                    ),
                    context_changed=context_changed,
                    reset_reason=reset_reason,
                    cycle_discard_reason=self._row.get(
                        "cycle_discard_reason", self._cycle_discard_reason
                    ),
                    expected_pose=expected,
                    joints=self._debug_joints,
                    poses=checks,
                    recognized_poses=matches,
                    per_arm_matches=per_arm,
                    condition_overlap=len(matches) > 1,
                    symmetry=symmetry,
                    rep_completed=event is not None,
                    event_id=event.event_id if event else None,
                    rearmed=decision_phase == "wait_next_y"
                    and self.phase == "y_confirmed",
                    transition_failures=failures,
                    thresholds=asdict(self.config),
                )
                trace = getattr(keypoints, "frame_trace", None)
                if trace is not None:
                    from app.analysis.frame_debug import compare, fingerprint

                    payload["frame_provenance"] = dict(
                        source_frame_id=trace["frame_id"],
                        source_timestamp=trace["timestamp"],
                        metadata_equal=getattr(keypoints, "frame_id", None)
                        == trace["frame_id"]
                        and result.timestamp == trace["timestamp"],
                        selected_pose_index=trace["selected_pose_index"],
                        raw_to_analyzer=compare(trace["raw_points"], keypoints),
                        extracted_to_analyzer=compare(
                            trace["extracted_points"], keypoints
                        ),
                        analyzer_fingerprint=fingerprint(keypoints),
                    )
                change = (
                    self.phase,
                    self._row.get("detected_pose"),
                    self._row.get("candidate"),
                    self._row.get("invalid_reason"),
                    self._cycle_discard_reason,
                    reason == "wrong_pose_order",
                    tuple(
                        (ident, joint["invalid_reason"])
                        for ident, joint in self._debug_joints.items()
                        if not joint["valid"]
                    ),
                )
                self._emit_debug(
                    payload,
                    force=previous != self.phase
                    or event is not None
                    or self._debug_reset_occurred,
                    change=change,
                )
            except Exception:
                pass  # 진단만 실패해도 원래 분석 결과를 그대로 반환합니다.
        return result


# 기존 Registry와 주입 코드에서 사용하던 이름을 호환합니다.
YtwRaiseAnalyzer = YTWRaiseAnalyzer
