"""지속 오류, 해제 히스테리시스와 범위 제한이 있는 공통 Rule 결과."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    code: str
    bad: bool | None
    recovered: bool = False
    reason: str | None = None
    event: bool = False
    value: float | None = None
    # pending: 관측 시점/동작 범위 대기, unavailable: 필수 관측 불가,
    # not_applicable: 현재 촬영 방향에서 애초에 적용하지 않는 항목입니다.
    state: str | None = None


# 확률/점수는 만들지 않습니다. 앞 항목일수록 우선 피드백입니다.
ERRORS = {
    'SQUAT_KNEE_VALGUS': ('무릎 안쪽 모임', '무릎이 발 방향을 따라 움직이도록 해주세요.'),
    'SQUAT_EXCESSIVE_FORWARD_LEAN': ('과도한 상체 숙임', '상체가 과도하게 숙여지지 않도록 움직임을 조절하세요.'),
    'SQUAT_INSUFFICIENT_DEPTH': ('스쿼트 깊이 부족', '통증 없는 가동범위 안에서 조금 더 내려가 보세요.'),
    'PUSHUP_HIP_SAG': ('엉덩이 처짐', '골반을 조금 들어 몸통 정렬을 유지하세요.'),
    'PUSHUP_HIP_RAISED': ('엉덩이 과도한 상승', '골반을 조금 내려 몸통 정렬을 유지하세요.'),
    'PUSHUP_INSUFFICIENT_DEPTH': ('푸시업 깊이 부족', '통증 없는 범위에서 팔꿈치를 조금 더 굽혀 내려가세요.'),
    'YTW_ELBOW_BENT': ('Y/T 팔꿈치 굽힘', 'Y와 T 자세에서 팔꿈치를 편안하게 펴주세요.'),
    'YTW_ARM_ASYMMETRY': ('양팔 비대칭', '양팔의 높이와 움직이는 방향을 맞춰주세요.'),
    'YTW_WRIST_POSITION_ERROR': ('W 손목 위치', 'W 자세에서 손목이 팔꿈치 위에 오도록 조절하세요.'),
    'PLANK_HIP_SAG': ('엉덩이 처짐', '골반을 조금 들어 몸통을 일직선으로 유지하세요.'),
    'PLANK_HIP_RAISED': ('엉덩이 과도한 상승', '골반을 조금 내려 몸통을 일직선으로 유지하세요.'),
    'PLANK_BODY_MISALIGNMENT': ('몸통 정렬 불량', '어깨와 골반, 발목의 정렬을 맞춰주세요.'),
    'PRESS_INCOMPLETE_EXTENSION': ('상단 팔 신전 부족', '통증 없는 범위에서 팔을 조금 더 펴 올려주세요.'),
    'PRESS_ARM_ASYMMETRY': ('양팔 높이 비대칭', '양팔을 비슷한 높이로 함께 올려주세요.'),
    'PRESS_TORSO_LEAN': ('몸통 좌우 기울어짐', '몸통이 좌우로 기울지 않도록 중심을 유지하세요.'),
}


class TemporalGate:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.since = self.clear_since = None
        self.active = False
        self.samples = 0

    def update(self, check, timestamp):
        if check.bad is None:
            self.reset()
        elif check.event:
            # 가동범위는 완료된 사이클의 측정치로 판정하고 프레임 지속시간을 요구하지 않습니다.
            self.active = check.bad
            self.since = self.clear_since = None
        elif self.active:
            if check.recovered:
                if self.clear_since is None:
                    self.clear_since = timestamp
                if timestamp-self.clear_since+1e-9 >= self.config.clear_sec:
                    self.reset()
            else:
                self.clear_since = None
        elif check.bad:
            if self.since is None:
                self.since, self.samples = timestamp, 1
            else:
                self.samples += 1
            if self.samples >= 2 and timestamp-self.since+1e-9 >= self.config.confirm_sec:
                self.active = True
        else:
            self.since = None
            self.samples = 0
        return self.active


class MotionRange:
    """기존 phase를 읽어 관측된 가동범위만 기록합니다. Reps 이벤트를 만들지 않습니다."""
    def __init__(self, max_duration):
        self.max_duration = max_duration
        self.reset()

    def reset(self):
        self.baseline = None
        self.started = None
        self.minimum = self.maximum = None
        self.aux_minimum = None
        self.side = None
        self.last_result = self.last_result_at = None

    def update(self, phase, value, timestamp, *, resting, moving, terminal, side=None, auxiliary=None):
        if self.started is not None and timestamp-self.started > self.max_duration:
            self.reset()
        if self.last_result_at is not None and timestamp-self.last_result_at > self.max_duration:
            self.last_result = self.last_result_at = None
        result = None
        if self.started is not None:
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)
            if auxiliary is not None:
                self.aux_minimum = min(self.aux_minimum, auxiliary)
            if phase in terminal:
                result = dict(minimum=self.minimum, maximum=self.maximum,
                              travel=self.maximum-self.minimum, auxiliary=self.aux_minimum)
                self.reset()
                self.last_result, self.last_result_at = result, timestamp
        if phase in resting:
            self.baseline = value
            self.side = side
        elif self.started is None and self.baseline is not None and phase in moving:
            self.started = timestamp
            self.last_result = self.last_result_at = None
            self.minimum, self.maximum = min(value,self.baseline), max(value,self.baseline)
            self.aux_minimum = auxiliary
            self.side = side
        return result
