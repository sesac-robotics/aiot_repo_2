"""5종 자세 관측기. 모든 임계값은 실영상 검증 전 실험값입니다."""

import math

from .base import Check, MotionRange
from .geometry import SIDES


def unavailable(codes, reason):
    return [Check(code, None, reason=reason) for code in codes]


def asymmetry(arms, code, cfg):
    height = abs(arms[0]["height"] - arms[1]["height"])
    angle = abs(arms[0]["elbow"] - arms[1]["elbow"])
    direction = abs(arms[0]["upper_elevation"] - arms[1]["upper_elevation"])
    return Check(
        code,
        height > cfg.asymmetry_height_enter
        or angle > cfg.asymmetry_angle_enter_deg
        or direction > cfg.asymmetry_angle_enter_deg,
        height < cfg.asymmetry_height_exit
        and angle < cfg.asymmetry_angle_exit_deg
        and direction < cfg.asymmetry_angle_exit_deg,
        value=height,
    )


class SquatPosture:
    codes = (
        "SQUAT_KNEE_VALGUS",
        "SQUAT_EXCESSIVE_FORWARD_LEAN",
        "SQUAT_INSUFFICIENT_DEPTH",
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.motion = MotionRange(cfg.max_cycle_sec)

    def reset(self):
        self.motion.reset()

    def applicable_codes(self, view):
        # 반대 방향 전용 검사는 정상 집계 대상에서 제외합니다.
        return (
            self.codes[:1]
            if view == "front"
            else self.codes[1:] if view == "side" else self.codes
        )

    def observe(self, pose, phase, timestamp, view):
        cfg = self.cfg
        if view not in ("front", "side"):
            return unavailable(self.codes, "camera_view_unknown")
        if phase not in ("standing", "descending", "bottom", "ascending"):
            self.reset()
            return [
                Check(code, None, reason="phase_unconfirmed", state="pending")
                for code in self.codes
            ]
        if view == "front":
            if phase == "standing":
                return [
                    Check(
                        self.codes[0],
                        None,
                        reason="active_squat_phase_required",
                        state="pending",
                    )
                ]
            pose.require(23, 24, 25, 26, 27, 28)
            hip_width, ankle_width, knee_width = (
                abs(pose.xy(a)[0] - pose.xy(b)[0])
                for a, b in ((23, 24), (27, 28), (25, 26))
            )
            if hip_width < 0.08 or ankle_width < hip_width * 0.5:
                return unavailable(
                    self.codes, "front_projection_or_stance_unreliable"
                )
            ratio = knee_width / ankle_width
            return [
                Check(
                    self.codes[0],
                    ratio < cfg.valgus_enter_ratio,
                    ratio > cfg.valgus_exit_ratio,
                    value=ratio,
                )
            ]
        # 어깨/골반이 보이면 상체를 독립 평가합니다. 발목 누락이 그 오류를 지우지 않습니다.
        try:
            side = pose.choose_side((0, 3), self.motion.side)
            s, e, w, h, k, a = SIDES[side]
            sx, sy = pose.xy(s)
            hx, hy = pose.xy(h)
            if sy >= hy or math.dist((sx, sy), (hx, hy)) < 0.05:
                raise ValueError("upright_torso_unreliable")
            lean = math.degrees(math.atan2(abs(sx - hx), hy - sy))
            lean_check = (
                Check(
                    self.codes[1],
                    lean > cfg.squat_lean_enter_deg,
                    lean < cfg.squat_lean_exit_deg,
                    value=lean,
                )
                if phase != "standing"
                else Check(self.codes[1], False, True, value=lean)
            )
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            lean_check = Check(
                self.codes[1], None, reason=str(error), state="unavailable"
            )
        try:
            side = pose.choose_side((0, 3, 4, 5), self.motion.side)
            s, e, w, h, k, a = SIDES[side]
            knee, hip = pose.angle(h, k, a), pose.angle(s, h, k)
            self.motion.update(
                phase,
                knee,
                timestamp,
                resting=("standing",),
                moving=("descending", "bottom", "ascending"),
                terminal=("standing",),
                side=side,
                auxiliary=hip,
            )
            observed = self.motion.last_result
            if phase == "bottom" and self.motion.started is not None:
                observed = dict(
                    minimum=self.motion.minimum,
                    auxiliary=self.motion.aux_minimum,
                    travel=self.motion.maximum - self.motion.minimum,
                )
            if (
                observed is None
                or observed["travel"] < cfg.squat_min_motion_deg
            ):
                depth = Check(
                    self.codes[2],
                    None,
                    reason="range_not_confirmed",
                    state="pending",
                )
            else:
                bad = (
                    observed["minimum"] > cfg.squat_depth_knee_deg
                    and observed["auxiliary"] > cfg.squat_depth_hip_deg
                )
                depth = Check(
                    self.codes[2],
                    bad,
                    not bad,
                    reason="observed_bottom_or_completed_cycle",
                    event=True,
                    value=observed["minimum"],
                )
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            self.motion.reset()
            depth = Check(
                self.codes[2], None, reason=str(error), state="unavailable"
            )
        return [lean_check, depth]


class BodyPosture:
    """한쪽 Shoulder/Hip/Ankle 기준. 반대쪽 관절은 요구하지 않습니다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.motion = MotionRange(cfg.max_cycle_sec)
        self.body_side = None

    def reset(self):
        self.motion.reset()
        self.body_side = None

    def observe(self, pose, phase, timestamp, view):
        if view != "side":
            return unavailable(self.codes, "side_view_required")
        if phase not in self.phases:
            self.reset()
            return unavailable(self.codes, "phase_unconfirmed")
        cfg = self.cfg
        side = pose.choose_side(
            (0, 1, 2, 3, 5), self.motion.side or self.body_side
        )
        self.body_side = side
        s, e, w, h, k, a = SIDES[side]
        body = pose.body(side)
        if body["tilt"] > cfg.max_body_tilt_deg:
            self.reset()
            return unavailable(self.codes, "not_horizontal_body")
        elbow = pose.angle(s, e, w)
        if self.exercise == "plank":
            if (
                not cfg.plank_min_elbow_deg <= elbow <= cfg.plank_max_elbow_deg
                or (pose.xy(e)[1] - pose.xy(s)[1]) / body["length"]
                < cfg.plank_min_elbow_drop_ratio
            ):
                self.reset()
                return unavailable(self.codes, "forearm_support_unconfirmed")
        deviation = body["deviation"]
        sag = Check(
            self.codes[0],
            deviation < -cfg.hip_deviation_enter,
            deviation > -cfg.hip_deviation_exit,
            value=deviation,
        )
        raised = Check(
            self.codes[1],
            deviation > cfg.hip_deviation_enter,
            deviation < cfg.hip_deviation_exit,
            value=deviation,
        )
        if self.exercise == "plank":
            # 골반 처짐/상승이 확정되면 공통 결과에서 일반 정렬 오류를 숨깁니다.
            alignment = Check(
                self.codes[2],
                body["alignment"] < cfg.body_alignment_enter_deg,
                body["alignment"] > cfg.body_alignment_exit_deg,
                value=body["alignment"],
            )
            return [sag, raised, alignment]
        self.motion.update(
            phase,
            elbow,
            timestamp,
            resting=("high",),
            moving=("descending", "bottom", "ascending"),
            terminal=("high",),
            side=side,
        )
        observed = self.motion.last_result
        if phase == "bottom" and self.motion.started is not None:
            observed = dict(
                minimum=self.motion.minimum,
                travel=self.motion.maximum - self.motion.minimum,
            )
        if observed is None or observed["travel"] < cfg.pushup_min_motion_deg:
            depth = Check(self.codes[2], None, reason="range_not_confirmed")
        else:
            bad = observed["minimum"] > cfg.pushup_depth_deg
            depth = Check(
                self.codes[2],
                bad,
                not bad,
                event=True,
                reason="observed_bottom_or_completed_cycle",
                value=observed["minimum"],
            )
        return [sag, raised, depth]


class PushUpPosture(BodyPosture):
    exercise = "push_up"
    codes = ("PUSHUP_HIP_SAG", "PUSHUP_HIP_RAISED", "PUSHUP_INSUFFICIENT_DEPTH")
    phases = ("high", "descending", "bottom", "ascending")


class PlankPosture(BodyPosture):
    exercise = "plank"
    codes = ("PLANK_HIP_SAG", "PLANK_HIP_RAISED", "PLANK_BODY_MISALIGNMENT")
    phases = ("stabilizing", "holding", "interrupted")


class YtwPosture:
    codes = ("YTW_ELBOW_BENT", "YTW_ARM_ASYMMETRY", "YTW_WRIST_POSITION_ERROR")

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        pass

    def observe(self, pose, phase, timestamp, view):
        if view != "front":
            return unavailable(self.codes, "front_view_required")
        # wait_next_y은 기존 W 완료 후 상태입니다. 새 Y/전환 구간에 W 기준을 적용하지 않습니다.
        stage = {
            "y_confirmed": "y",
            "t_confirmed": "t",
            "wait_next_y": "w",
        }.get(phase)
        if stage is None:
            return unavailable(self.codes, "phase_unconfirmed")
        arms = pose.arms()
        cfg = self.cfg
        elevations = [arm["upper_elevation"] for arm in arms]
        near_stage = (
            all(
                cfg.y_min_elevation_deg <= v <= cfg.y_max_elevation_deg
                for v in elevations
            )
            if stage == "y"
            else (
                all(abs(v) <= cfg.t_max_elevation_deg for v in elevations)
                if stage == "t"
                else all(
                    -cfg.w_max_down_elevation_deg
                    <= v
                    <= cfg.w_max_up_elevation_deg
                    for v in elevations
                )
                and all(
                    cfg.w_stage_min_elbow_deg
                    <= a["elbow"]
                    <= cfg.w_stage_max_elbow_deg
                    for a in arms
                )
            )
        )
        if not near_stage:
            return unavailable(self.codes, "transition_or_stage_unconfirmed")
        symmetry = asymmetry(arms, self.codes[1], cfg)
        if stage in ("y", "t"):
            elbow = min(a["elbow"] for a in arms)
            return [
                Check(
                    self.codes[0],
                    elbow < cfg.ytw_elbow_enter_deg,
                    elbow > cfg.ytw_elbow_exit_deg,
                    value=elbow,
                ),
                symmetry,
            ]
        lift = min(a["wrist_above_elbow"] for a in arms)
        angles = [a["elbow"] for a in arms]
        bad = lift < cfg.w_wrist_enter_ratio or any(
            a < cfg.w_elbow_enter_min_deg or a > cfg.w_elbow_enter_max_deg
            for a in angles
        )
        recovered = lift > cfg.w_wrist_exit_ratio and all(
            cfg.w_elbow_exit_min_deg <= a <= cfg.w_elbow_exit_max_deg
            for a in angles
        )
        return [symmetry, Check(self.codes[2], bad, recovered, value=lift)]


class PressPosture:
    codes = (
        "PRESS_TORSO_LEAN",
        "PRESS_ARM_ASYMMETRY",
        "PRESS_INCOMPLETE_EXTENSION",
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.motion = MotionRange(cfg.max_cycle_sec)

    def reset(self):
        self.motion.reset()

    def observe(self, pose, phase, timestamp, view):
        if view != "front":
            return unavailable(self.codes, "front_view_required")
        if phase not in ("ready", "pressing", "top", "lowering"):
            self.reset()
            return unavailable(self.codes, "phase_unconfirmed")
        arms = pose.arms()
        cfg = self.cfg
        # 양팔 중 덜 펴진 팔을 평가하며 진행 중 신전 부족을 출력하지 않습니다.
        elbow = min(a["elbow"] for a in arms)
        self.motion.update(
            phase,
            elbow,
            timestamp,
            resting=("ready",),
            moving=("pressing", "top", "lowering"),
            terminal=("ready",),
        )
        shoulder = tuple((a + b) / 2 for a, b in zip(pose.xy(11), pose.xy(12)))
        hip = tuple((a + b) / 2 for a, b in zip(pose.xy(23), pose.xy(24)))
        if shoulder[1] >= hip[1]:
            self.reset()
            return unavailable(self.codes, "upright_torso_unconfirmed")
        tilt = math.degrees(
            math.atan2(abs(shoulder[0] - hip[0]), hip[1] - shoulder[1])
        )
        lean = Check(
            self.codes[0],
            tilt > cfg.press_lean_enter_deg,
            tilt < cfg.press_lean_exit_deg,
            value=tilt,
        )
        symmetry = asymmetry(arms, self.codes[1], cfg)
        observed = self.motion.last_result
        if phase == "top":
            extension = Check(
                self.codes[2],
                elbow < cfg.press_extension_deg,
                elbow >= cfg.press_extension_deg,
                value=elbow,
            )
        elif (
            observed is not None
            and observed["travel"] >= cfg.press_min_motion_deg
        ):
            bad = observed["maximum"] < cfg.press_extension_deg
            extension = Check(
                self.codes[2],
                bad,
                not bad,
                event=True,
                reason="completed_cycle",
                value=observed["maximum"],
            )
        else:
            extension = Check(
                self.codes[2], None, reason="top_or_range_not_confirmed"
            )
        return [lean, symmetry, extension]
