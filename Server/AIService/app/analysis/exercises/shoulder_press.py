"""양팔 팔꿈치 각도와 상완 길이로 정규화한 손목 높이로 반복만 측정합니다.

시작 자세 → 양팔 신전 → 시작 자세 복귀가 관측돼야 완료됩니다. 점수/오류/피드백은
계산하지 않습니다. 임계값은 실영상 검증 전 초기 실험값이며 실제 정확도를 보장하지 않습니다.
"""
from dataclasses import asdict, dataclass
import math
import time
from uuid import uuid4

from app.analysis.exercise import AnalysisResult, RepCompleted
from app.analysis.shoulder_debug import ShoulderPressDebug, context_fields


@dataclass(frozen=True)
class ShoulderPressConfig:
    ready_min_angle_deg: float = 40.0
    ready_max_angle_deg: float = 120.0
    pressing_angle_deg: float = 110.0
    top_enter_angle_deg: float = 135.0
    top_exit_angle_deg: float = 125.0
    # 높이 = (어깨 y - 손목 y) / 같은 팔의 픽셀 상완 길이. 위쪽이 양수입니다.
    ready_min_wrist_height: float = -.35
    ready_max_wrist_height: float = 1.05
    pressing_wrist_height: float = 1.10
    top_enter_wrist_height: float = 1.35
    top_exit_wrist_height: float = 1.15
    min_visibility: float = .6
    min_presence: float = .5
    max_arm_difference_deg: float = 30.0
    max_wrist_height_difference: float = .6
    transition_duration_sec: float = .15
    transition_samples: int = 2
    min_rep_duration_sec: float = .8
    max_rep_duration_sec: float = 12.0
    max_frame_gap_sec: float = .75

    def __post_init__(self):
        numeric = [value for name, value in vars(self).items() if name != 'transition_samples']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
            raise ValueError('숄더 프레스 설정은 유한한 숫자여야 합니다.')
        # READY/PRESSING의 각도 범위 중첩은 허용합니다. 손목 높이 조건은 분리되어 있습니다.
        if not (0 < self.ready_min_angle_deg < self.ready_max_angle_deg
                < self.top_enter_angle_deg <= 180
                and self.ready_min_angle_deg < self.pressing_angle_deg
                <= self.top_exit_angle_deg < self.top_enter_angle_deg):
            raise ValueError('팔꿈치 각도 임계값의 순서가 올바르지 않습니다.')
        if not (self.ready_min_wrist_height < self.ready_max_wrist_height
                < self.pressing_wrist_height <= self.top_exit_wrist_height
                < self.top_enter_wrist_height):
            raise ValueError('손목 높이 임계값의 순서가 올바르지 않습니다.')
        if not (0 <= self.min_visibility <= 1 and 0 <= self.min_presence <= 1
                and 0 < self.max_arm_difference_deg <= 180 and self.max_wrist_height_difference > 0):
            raise ValueError('가시성 또는 좌우 차이 설정이 올바르지 않습니다.')
        if (self.transition_duration_sec <= 0 or self.max_frame_gap_sec <= 0
                or type(self.transition_samples) is not int or self.transition_samples < 2
                or not 0 < self.min_rep_duration_sec < self.max_rep_duration_sec):
            raise ValueError('시간 안정화 설정이 올바르지 않습니다.')


def elbow_angle(shoulder, elbow, wrist, frame_size):
    """정규화 x/y를 픽셀 비율로 보정한 팔꿈치 내각(도)."""
    width, height = frame_size
    upper = ((shoulder['x'] - elbow['x']) * width, (shoulder['y'] - elbow['y']) * height)
    lower = ((wrist['x'] - elbow['x']) * width, (wrist['y'] - elbow['y']) * height)
    norm = math.hypot(*upper) * math.hypot(*lower)
    if norm <= 1e-9:
        raise ValueError('팔꿈치 각도를 계산할 수 없는 관절 벡터입니다.')
    cosine = (upper[0] * lower[0] + upper[1] * lower[1]) / norm
    return math.degrees(math.acos(max(-1., min(1., cosine))))


class DumbbellShoulderPressAnalyzer:
    def __init__(self, exercise_name='dumbbell_shoulder_press', exercise_type='repetition',
                 target_duration_sec=None, *, config=None):
        if (exercise_name != 'dumbbell_shoulder_press' or exercise_type != 'repetition'
                or target_duration_sec is not None):
            raise ValueError('dumbbell_shoulder_press/repetition 설정만 지원합니다.')
        self.config = config if config is not None else ShoulderPressConfig()
        # reset/관절 손실 이후에도 같은 인스턴스의 이벤트 번호는 재사용하지 않습니다.
        self._debug = ShoulderPressDebug()
        self._debug_frame = {}
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
        previous = getattr(self, 'phase', None)
        context = getattr(self, '_context', None)
        last_timestamp = getattr(self, '_last_timestamp', None)
        self._clear_motion()
        self._last_timestamp = None
        self._context = None
        if previous not in (None, 'unknown'):
            self._debug.emit('reset', lambda: {
                **context_fields(context), 'previous_phase': previous, 'next_phase': 'unknown',
                'last_frame_timestamp': last_timestamp, 'reason': 'analyzer_reset',
            }, force=True)

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _result(self, timestamp, *, phase=None, event=None):
        return AnalysisResult('dumbbell_shoulder_press', 'repetition', phase or self.phase,
                              timestamp, repetition_event=event)

    def _reject(self, reason, message):
        if self._debug.enabled:
            self._debug_frame['invalid_reason'] = reason
        raise ValueError(message)

    def _features(self, keypoints, frame_size):
        required = {11, 12, 13, 14, 15, 16}
        points = {}
        for point in keypoints:
            if not isinstance(point, dict) or point.get('id') not in required:
                continue
            ident = point['id']
            if ident in points:
                self._reject(f'joint_{ident}_duplicate', '중복 상체 관절 ID입니다.')
            if any(not self._finite(point.get(axis)) or not 0 <= point[axis] <= 1 for axis in ('x', 'y')):
                self._reject(f'joint_{ident}_coordinates_invalid', '상체 관절 좌표가 유효하지 않습니다.')
            visibility, presence = point.get('visibility'), point.get('presence')
            if (not self._finite(visibility) or not self.config.min_visibility <= visibility <= 1
                    or (presence is not None and (not self._finite(presence)
                                                 or not self.config.min_presence <= presence <= 1))):
                field = 'visibility' if (not self._finite(visibility)
                    or not self.config.min_visibility <= visibility <= 1) else 'presence'
                self._reject(f'joint_{ident}_{field}_invalid_or_below_threshold', '양팔 관절의 가시성이 부족합니다.')
            points[ident] = point
        if set(points) != required:
            self._reject(f'missing_joints_{sorted(required-set(points))}', '양쪽 Shoulder/Elbow/Wrist 관절이 모두 필요합니다.')
        width, height = frame_size
        features = []
        for shoulder_id, elbow_id, wrist_id in ((11, 13, 15), (12, 14, 16)):
            shoulder, elbow, wrist = (points[i] for i in (shoulder_id, elbow_id, wrist_id))
            length = math.hypot((shoulder['x']-elbow['x'])*width,
                                (shoulder['y']-elbow['y'])*height)
            if length <= 1e-6:
                self._reject(f'joint_{shoulder_id}_{elbow_id}_upper_arm_degenerate', '상완 길이로 손목 높이를 정규화할 수 없습니다.')
            features.append((elbow_angle(shoulder, elbow, wrist, frame_size),
                             (shoulder['y']-wrist['y'])*height/length))
            if self._debug.enabled:
                side = 'left' if shoulder_id == 11 else 'right'
                self._debug_frame[side+'_elbow'] = features[-1][0]
                self._debug_frame[side+'_wrist_height'] = features[-1][1]
        if self._debug.enabled:
            self._debug_frame['arm_angle_difference_deg'] = abs(features[0][0]-features[1][0])
            self._debug_frame['wrist_height_difference'] = abs(features[0][1]-features[1][1])
        if (abs(features[0][0]-features[1][0]) > self.config.max_arm_difference_deg
                or abs(features[0][1]-features[1][1]) > self.config.max_wrist_height_difference):
            self._reject('arm_angle_difference_exceeded'
                         if abs(features[0][0]-features[1][0]) > self.config.max_arm_difference_deg
                         else 'wrist_height_difference_exceeded', '좌우 팔의 관측 차이가 큽니다.')
        return features

    def _stable(self, candidate, timestamp):
        if candidate is None:
            self._candidate, self._candidate_since, self._candidate_samples = None, None, 0
            return False
        if candidate != self._candidate:
            self._candidate, self._candidate_since, self._candidate_samples = candidate, timestamp, 1
            return False
        self._candidate_samples += 1
        return (self._candidate_samples >= self.config.transition_samples
                and timestamp-self._candidate_since >= self.config.transition_duration_sec)

    def _analyze(self, keypoints, context):
        if context != self._context:
            self.reset()
            self._context = context
        timestamp, size = getattr(keypoints, 'timestamp', None), getattr(keypoints, 'frame_size', None)
        if (not self._finite(timestamp) or not isinstance(size, (tuple, list)) or len(size) != 2
                or any(type(v) is not int or v <= 0 for v in size)
                or getattr(context, 'exercise_name', None) != 'dumbbell_shoulder_press'):
            if self._debug.enabled:
                self._debug_frame['invalid_reason'] = (
                    'frame_timestamp_invalid' if not self._finite(timestamp)
                    else 'exercise_name_mismatch'
                    if getattr(context, 'exercise_name', None) != 'dumbbell_shoulder_press'
                    else 'frame_size_invalid')
            self._clear_motion()
            return self._result(time.monotonic(), phase='invalid')
        try:
            arms = self._features(keypoints, size)
        except (ValueError, TypeError, OverflowError) as error:
            if self._debug.enabled:
                self._debug_frame.setdefault('invalid_reason', 'joint_geometry_invalid')
                self._debug_frame['invalid_detail'] = str(error)
            self._clear_motion()
            return self._result(timestamp, phase='invalid')
        if self._last_timestamp is not None:
            if timestamp == self._last_timestamp:
                if self._debug.enabled:
                    self._debug_frame['duplicate_timestamp'] = True
                return self._result(timestamp)  # 재전달은 안정화나 완료 이벤트를 진행시키지 않습니다.
            if timestamp < self._last_timestamp:
                if self._debug.enabled:
                    self._debug_frame['invalid_reason'] = 'timestamp_reversed'
                self._clear_motion()
                return self._result(timestamp, phase='invalid')
            if timestamp-self._last_timestamp > self.config.max_frame_gap_sec:
                if self._debug.enabled:
                    self._debug_frame['reset_reason'] = 'frame_gap_exceeded'
                self._clear_motion()
        self._last_timestamp = timestamp
        if (self._cycle_started is not None
                and timestamp-self._cycle_started > self.config.max_rep_duration_sec):
            if self._debug.enabled:
                self._debug_frame['reset_reason'] = 'cycle_timeout'
            self._clear_motion()

        cfg = self.config
        # 평균으로 한쪽 팔의 미완료를 덮지 않습니다. 모든 조건은 양팔 각각 만족해야 합니다.
        ready = all(cfg.ready_min_angle_deg <= a <= cfg.ready_max_angle_deg
                    and cfg.ready_min_wrist_height <= h <= cfg.ready_max_wrist_height for a, h in arms)
        pressing = all(a >= cfg.pressing_angle_deg and h >= cfg.pressing_wrist_height for a, h in arms)
        top = all(a >= cfg.top_enter_angle_deg and h >= cfg.top_enter_wrist_height for a, h in arms)
        lowering = all(a <= cfg.top_exit_angle_deg and h <= cfg.top_exit_wrist_height for a, h in arms)
        candidate = None
        if self.phase == 'unknown':
            if ready:
                candidate = 'ready'
        elif self.phase == 'ready':
            if pressing:
                candidate = 'pressing'
        elif self.phase == 'pressing':
            if top:
                candidate = 'top'
            elif ready:
                candidate = 'ready'  # TOP 미도달 후 복귀는 횟수 없이 종료합니다.
        elif self.phase == 'top':
            if lowering:
                candidate = 'lowering'
        elif self.phase == 'lowering':
            if ready:
                candidate = 'ready'
            elif top:
                candidate = 'top'  # 불완전 복귀 뒤 다시 올려도 추가 횟수는 없습니다.

        event = None
        stable = self._stable(candidate, timestamp)
        if self._debug.enabled:
            self._debug_frame.update(candidate=candidate, candidate_samples=self._candidate_samples,
                                     candidate_duration_sec=(timestamp-self._candidate_since
                                         if self._candidate_since is not None else 0.),
                                     transition_stable=stable,
                                     cycle_duration_sec=(timestamp-self._cycle_started
                                         if self._cycle_started is not None else None))
        if stable:
            previous = self.phase
            if candidate == 'pressing':
                self._cycle_started = self._candidate_since
            elif candidate == 'ready':
                if self._debug.enabled:
                    self._debug_frame['completion_rejection'] = (
                        'top_not_reached' if previous == 'pressing'
                        else 'cycle_not_started' if self._cycle_started is None
                        else 'cycle_too_short' if timestamp-self._cycle_started < cfg.min_rep_duration_sec
                        else None)
                if (previous == 'lowering' and self._cycle_started is not None
                        and cfg.min_rep_duration_sec <= timestamp-self._cycle_started <= cfg.max_rep_duration_sec):
                    self._event_number += 1
                    event = RepCompleted(f'shoulder_press:{self._instance_id}:{self._event_number}')
                self._cycle_started = None
            self.phase = candidate
            self._stable(None, timestamp)
        return self._result(timestamp, event=event)


    def _condition_debug(self):
        # 실제 측정값과 같은 Config로 양팔 각각의 실패 항목을 설명합니다. 판정에는 쓰지 않습니다.
        cfg = self.config
        result = {}
        for side in ('left', 'right'):
            angle, height = self._debug_frame.get(side+'_elbow'), self._debug_frame.get(side+'_wrist_height')
            for name, angle_low, angle_high, height_low, height_high in (
                    ('ready', cfg.ready_min_angle_deg, cfg.ready_max_angle_deg,
                     cfg.ready_min_wrist_height, cfg.ready_max_wrist_height),
                    ('pressing', cfg.pressing_angle_deg, 180., cfg.pressing_wrist_height, math.inf),
                    ('top', cfg.top_enter_angle_deg, 180., cfg.top_enter_wrist_height, math.inf),
                    ('lowering', 0., cfg.top_exit_angle_deg, -math.inf, cfg.top_exit_wrist_height)):
                if angle is None or height is None:
                    result[f'{name}_{side}'] = None
                    result[f'{name}_{side}_reasons'] = ['measurement_unavailable']
                    continue
                reasons = []
                if angle < angle_low: reasons.append(f'{side}_elbow_below_{name}_threshold')
                if angle > angle_high: reasons.append(f'{side}_elbow_above_{name}_threshold')
                if height < height_low: reasons.append(f'{side}_wrist_below_{name}_threshold')
                if height > height_high: reasons.append(f'{side}_wrist_above_{name}_threshold')
                result[f'{name}_{side}'] = not reasons
                result[f'{name}_{side}_reasons'] = reasons
        for name in ('ready', 'pressing', 'top', 'lowering'):
            result[name] = (None if any(result[f'{name}_{s}'] is None for s in ('left','right'))
                            else all(result[f'{name}_{s}'] for s in ('left','right')))
        return result

    def analyze(self, keypoints, context):
        if not self._debug.enabled:
            return self._analyze(keypoints, context)
        before = self.phase
        timestamp = getattr(keypoints, 'timestamp', None)
        names = dict(zip(range(11,17), ('left_shoulder', 'right_shoulder', 'left_elbow',
                                        'right_elbow', 'left_wrist', 'right_wrist')))
        joints = {str(i): {'name': names[i], 'visibility': None, 'presence': None, 'observed': False}
                  for i in range(11,17)}
        for point in keypoints:
            if isinstance(point, dict) and type(point.get('id')) is int and 11 <= point['id'] <= 16:
                joints[str(point['id'])] = {'name': names[point['id']], 'visibility': point.get('visibility'),
                                          'presence': point.get('presence'), 'observed': True}
        self._debug_frame = {
            **context_fields(context), 'analyzer_called': True, 'frame_timestamp': timestamp,
            'previous_phase': before, 'joints': joints, 'left_elbow': None, 'right_elbow': None,
            'left_wrist_height': None, 'right_wrist_height': None,
            'frame_gap_sec': (timestamp-self._last_timestamp
                if self._finite(timestamp) and self._last_timestamp is not None else None),
            'context_changed': context != self._context,
        }
        result = self._analyze(keypoints, context)
        self._debug_frame.update(self._condition_debug())
        event = result.repetition_event
        self._debug_frame.update(phase=result.phase, next_phase=self.phase,
                                 input_valid=result.phase != 'invalid',
                                 rep_completed=event is not None, event_id=event.event_id if event else None)
        # 주기 로그에도 임계값과 안정화 조건을 같은 레코드로 출력합니다.
        self._debug.emit('analyzer', lambda: {**self._debug_frame, 'thresholds': asdict(self.config)},
                         force=event is not None or before != self.phase or self._debug_frame['context_changed'],
                         change=(result.phase, self._debug_frame.get('invalid_reason'),
                                 self._debug_frame.get('reset_reason')))
        return result


# 기존 Registry/외부 주입 코드의 클래스 이름과 호환합니다.
ShoulderPressAnalyzer = DumbbellShoulderPressAnalyzer
