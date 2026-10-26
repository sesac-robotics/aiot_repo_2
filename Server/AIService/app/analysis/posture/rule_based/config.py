"""검증 전 실험용 자세 기준. Reps/Hold Config와 독립적입니다."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class RuleConfig:
    min_visibility: float = .6
    min_presence: float = .5
    confirm_sec: float = .4
    clear_sec: float = .25
    max_frame_gap_sec: float = .75
    max_cycle_sec: float = 12.
    normal_confirm_sec: float = .4
    squat_pose_grace_sec: float = .35
    squat_result_hold_sec: float = 2.
    hip_deviation_enter: float = .12
    hip_deviation_exit: float = .08
    body_alignment_enter_deg: float = 145.
    body_alignment_exit_deg: float = 155.
    max_body_tilt_deg: float = 40.
    squat_depth_knee_deg: float = 115.
    squat_depth_hip_deg: float = 120.
    squat_min_motion_deg: float = 25.
    squat_lean_enter_deg: float = 50.
    squat_lean_exit_deg: float = 40.
    valgus_enter_ratio: float = .75
    valgus_exit_ratio: float = .85
    # 정면 투영의 최소 해상도와 신체 크기 대비 폭을 함께 검사합니다(실험값).
    squat_front_min_hip_width: float = .04
    squat_front_min_hip_to_leg_ratio: float = .10
    squat_front_min_leg_height: float = .15
    squat_front_min_ankle_to_hip_ratio: float = .5
    pushup_depth_deg: float = 120.
    pushup_min_motion_deg: float = 25.
    ytw_elbow_enter_deg: float = 150.
    ytw_elbow_exit_deg: float = 162.
    asymmetry_height_enter: float = .45
    asymmetry_height_exit: float = .30
    asymmetry_angle_enter_deg: float = 25.
    asymmetry_angle_exit_deg: float = 15.
    y_min_elevation_deg: float = 25.
    y_max_elevation_deg: float = 85.
    t_max_elevation_deg: float = 25.
    w_max_down_elevation_deg: float = 65.
    w_max_up_elevation_deg: float = 25.
    w_stage_min_elbow_deg: float = 20.
    w_stage_max_elbow_deg: float = 165.
    w_elbow_enter_min_deg: float = 40.
    w_elbow_enter_max_deg: float = 145.
    w_elbow_exit_min_deg: float = 55.
    w_elbow_exit_max_deg: float = 130.
    plank_min_elbow_deg: float = 45.
    plank_max_elbow_deg: float = 140.
    plank_min_elbow_drop_ratio: float = .03
    w_wrist_enter_ratio: float = .10
    w_wrist_exit_ratio: float = .20
    press_extension_deg: float = 155.
    press_min_motion_deg: float = 25.
    press_lean_enter_deg: float = 15.
    press_lean_exit_deg: float = 10.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    or field.name.endswith('_deg') and value > 180):
                raise ValueError('Rule Config 값은 양의 유한한 숫자여야 합니다: ' + field.name)
        if not (self.min_visibility <= 1 and self.min_presence <= 1
                and self.hip_deviation_exit < self.hip_deviation_enter
                and self.body_alignment_enter_deg < self.body_alignment_exit_deg <= 180
                and self.squat_lean_exit_deg < self.squat_lean_enter_deg < 90
                and self.valgus_enter_ratio < self.valgus_exit_ratio
                and self.ytw_elbow_enter_deg < self.ytw_elbow_exit_deg <= 180
                and self.asymmetry_height_exit < self.asymmetry_height_enter
                and self.asymmetry_angle_exit_deg < self.asymmetry_angle_enter_deg
                and self.y_min_elevation_deg < self.y_max_elevation_deg < 90
                and self.w_elbow_enter_min_deg < self.w_elbow_exit_min_deg
                < self.w_elbow_exit_max_deg < self.w_elbow_enter_max_deg
                and self.w_stage_min_elbow_deg < self.w_stage_max_elbow_deg
                and self.plank_min_elbow_deg < self.plank_max_elbow_deg
                and self.w_wrist_enter_ratio < self.w_wrist_exit_ratio
                and self.press_lean_exit_deg < self.press_lean_enter_deg < 90
                and self.confirm_sec <= self.max_frame_gap_sec
                and self.max_body_tilt_deg < 90):
            raise ValueError('Rule Config 히스테리시스/시간 설정이 올바르지 않습니다.')
