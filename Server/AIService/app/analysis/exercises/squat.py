"""양쪽 Hip/Knee/Ankle의 영상 비율 보정 2D 각도로 반복만 판정합니다.

초기 설정값은 합성 테스트용 시작점이며 실제 영상의 정확도를 보증하지 않습니다.
관절/시간 연속성이 깨지면 미완료 동작을 폐기하고 선 자세부터 다시 확인합니다.
"""
from dataclasses import dataclass
import math
import time
from uuid import uuid4

from app.analysis.exercise import AnalysisResult, RepCompleted


@dataclass(frozen=True)
class SquatConfig:
    standing_enter_deg: float = 165.0
    standing_exit_deg: float = 150.0
    bottom_enter_deg: float = 120.0
    bottom_exit_deg: float = 135.0
    min_visibility: float = .6
    min_presence: float = .5
    max_leg_difference_deg: float = 25.0
    transition_duration_sec: float = .15
    transition_samples: int = 2
    min_rep_duration_sec: float = .8
    max_rep_duration_sec: float = 10.0
    max_frame_gap_sec: float = .75

    def __post_init__(self):
        # 잘못된 임계값으로 상태 전이/시간 검증이 무력화되지 않도록 확인합니다.
        numeric = [value for name, value in vars(self).items() if name != 'transition_samples']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
            raise ValueError('스쿼트 설정은 유한한 숫자여야 합니다.')
        if not (0 < self.bottom_enter_deg < self.bottom_exit_deg < self.standing_exit_deg
                < self.standing_enter_deg <= 180):
            raise ValueError('각도 임계값의 순서가 올바르지 않습니다.')
        if not (0 <= self.min_visibility <= 1 and 0 <= self.min_presence <= 1
                and 0 < self.max_leg_difference_deg <= 180):
            raise ValueError('관절 신뢰도 또는 좌우 각도 차이 설정이 올바르지 않습니다.')
        if (self.transition_duration_sec <= 0 or self.max_frame_gap_sec <= 0
                or type(self.transition_samples) is not int or self.transition_samples < 2
                or not 0 < self.min_rep_duration_sec < self.max_rep_duration_sec):
            raise ValueError('안정화/반복 시간 설정이 올바르지 않습니다.')


def knee_angle(hip, knee, ankle, frame_size):
    """정규화 x/y를 영상 픽셀 비율로 보정한 무릎 내각(도). 퇴화한 벡터는 거부합니다."""
    width, height = frame_size
    first = ((hip['x'] - knee['x']) * width, (hip['y'] - knee['y']) * height)
    second = ((ankle['x'] - knee['x']) * width, (ankle['y'] - knee['y']) * height)
    norm = math.hypot(*first) * math.hypot(*second)
    if norm <= 1e-9:
        raise ValueError('무릎 각도를 계산할 수 없는 관절 벡터입니다.')
    cosine = (first[0] * second[0] + first[1] * second[1]) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


class SquatAnalyzer:
    def __init__(self, exercise_name='squat', exercise_type='repetition', target_duration_sec=None,
                 *, config=None):
        if exercise_name != 'squat' or exercise_type != 'repetition' or target_duration_sec is not None:
            raise ValueError('SquatAnalyzer는 squat/repetition 설정만 지원합니다.')
        self.config = config if config is not None else SquatConfig()
        # reset이 같은 세대에서 발생해도 완료 이벤트 ID를 재사용하지 않습니다.
        self._instance_id = uuid4().hex
        self._event_number = 0
        self.reset()

    def _clear_motion(self):
        self.phase = 'unknown'
        self._candidate = None
        self._candidate_since = None
        self._candidate_samples = 0
        self._cycle_started = None

    def reset(self):
        self._clear_motion()
        self._last_timestamp = None
        self._context = None

    def _result(self, timestamp, *, phase=None, event=None):
        return AnalysisResult('squat', 'repetition', phase or self.phase, timestamp,
                              repetition_event=event)

    @staticmethod
    def _finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _angles(self, keypoints, frame_size):
        required = {23, 24, 25, 26, 27, 28}
        points = {}
        for point in keypoints:
            if not isinstance(point, dict) or point.get('id') not in required:
                continue
            ident = point['id']
            if ident in points:
                raise ValueError('중복 관절 ID입니다.')
            for coordinate in ('x', 'y'):
                value = point.get(coordinate)
                if not self._finite_number(value) or not 0 <= value <= 1:
                    raise ValueError('관절 좌표가 유효하지 않습니다.')
            visibility = point.get('visibility')
            presence = point.get('presence')
            if (not self._finite_number(visibility) or not self.config.min_visibility <= visibility <= 1
                    or (presence is not None and (not self._finite_number(presence)
                                                 or not self.config.min_presence <= presence <= 1))):
                raise ValueError('관절 신뢰도가 부족합니다.')
            points[ident] = point
        if set(points) != required:
            raise ValueError('양쪽 하체 관절이 모두 필요합니다.')
        left = knee_angle(points[23], points[25], points[27], frame_size)
        right = knee_angle(points[24], points[26], points[28], frame_size)
        if abs(left - right) > self.config.max_leg_difference_deg:
            raise ValueError('좌우 무릎 각도 차이가 큽니다.')
        return (left + right) / 2

    def _stable(self, candidate, timestamp):
        if candidate is None:
            self._candidate = None
            self._candidate_since = None
            self._candidate_samples = 0
            return False
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = timestamp
            self._candidate_samples = 1
            return False
        self._candidate_samples += 1
        return (self._candidate_samples >= self.config.transition_samples
                and timestamp - self._candidate_since >= self.config.transition_duration_sec)

    def analyze(self, keypoints, context):
        if context != self._context:
            self.reset()
            self._context = context
        timestamp = getattr(keypoints, 'timestamp', None)
        frame_size = getattr(keypoints, 'frame_size', None)
        if (not self._finite_number(timestamp) or frame_size is None
                or not isinstance(frame_size, (tuple, list)) or len(frame_size) != 2
                or any(type(size) is not int or size <= 0 for size in frame_size)
                or context.exercise_name != 'squat'):
            self._clear_motion()
            return self._result(time.monotonic(), phase='invalid')
        try:
            angle = self._angles(keypoints, frame_size)
        except (ValueError, TypeError, OverflowError):
            self._clear_motion()
            return self._result(timestamp, phase='invalid')
        if self._last_timestamp is not None:
            if timestamp == self._last_timestamp:
                # 같은 표본 재전달은 안정화 시간/표본 수나 이벤트를 증가시키지 않습니다.
                return self._result(timestamp)
            if timestamp < self._last_timestamp:
                self._clear_motion()
                return self._result(timestamp, phase='invalid')
            if timestamp - self._last_timestamp > self.config.max_frame_gap_sec:
                self._clear_motion()
        self._last_timestamp = timestamp
        if (self._cycle_started is not None
                and timestamp - self._cycle_started > self.config.max_rep_duration_sec):
            self._clear_motion()

        cfg = self.config
        candidate = None
        if self.phase == 'unknown':
            if angle >= cfg.standing_enter_deg:
                candidate = 'standing'
        elif self.phase == 'standing':
            if angle <= cfg.standing_exit_deg:
                candidate = 'descending'
        elif self.phase == 'descending':
            if angle <= cfg.bottom_enter_deg:
                candidate = 'bottom'
            elif angle >= cfg.standing_enter_deg:
                candidate = 'standing'  # 깊이가 부족한 부분 동작: 이벤트 없이 복귀
        elif self.phase == 'bottom':
            if angle >= cfg.bottom_exit_deg:
                candidate = 'ascending'
        elif self.phase == 'ascending':
            if angle >= cfg.standing_enter_deg:
                candidate = 'standing'
            elif angle <= cfg.bottom_enter_deg:
                candidate = 'bottom'

        event = None
        if self._stable(candidate, timestamp):
            previous = self.phase
            if candidate == 'descending':
                self._cycle_started = self._candidate_since
            elif candidate == 'standing':
                if (previous == 'ascending' and self._cycle_started is not None
                        and cfg.min_rep_duration_sec <= timestamp - self._cycle_started
                        <= cfg.max_rep_duration_sec):
                    self._event_number += 1
                    event = RepCompleted(f'squat:{self._instance_id}:{self._event_number}')
                self._cycle_started = None
            self.phase = candidate
            self._stable(None, timestamp)
        return self._result(timestamp, event=event)
