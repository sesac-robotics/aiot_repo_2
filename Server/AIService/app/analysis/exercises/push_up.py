"""측면 푸시업의 반복만 측정합니다. 점수/자세 오류는 별도 모델의 책임입니다.

선택 팔의 연속 관측, 수평에 가까운 몸통, 어깨 하강·복귀를 함께 요구합니다.
임계값은 실영상 검증이 필요한 설정이며, 선택 팔의 짧은 관측 누락은 상태 전진 없이 기다립니다.
"""
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import time
from uuid import uuid4

from app.analysis.exercise import AnalysisResult, RepCompleted


@dataclass(frozen=True)
class PushUpConfig:
    high_enter_deg: float = 145.
    descending_enter_deg: float = 140.
    bottom_enter_deg: float = 135.
    ascending_enter_deg: float = 140.

    min_visibility: float = .5
    min_presence: float = .4

    transition_duration_sec: float = .15
    transition_samples: int = 2

    min_rep_duration_sec: float = .8
    max_rep_duration_sec: float = 10.
    max_frame_gap_sec: float = .75

    min_shoulder_drop_ratio: float = .22
    max_shoulder_return_ratio: float = .06

    max_torso_inclination_deg: float = 55.
    max_leg_inclination_deg: float = 65.
    occlusion_grace_sec: float = 0.3

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != 'transition_samples']
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('푸시업 설정은 유한한 숫자여야 합니다.')
        # 하강/상승 방향의 hysteresis를 각각 검사합니다. 상승 임계값은 하강보다 높을 수 있습니다.
        if not (0 < self.bottom_enter_deg < self.descending_enter_deg < self.high_enter_deg <= 180
                and self.bottom_enter_deg < self.ascending_enter_deg < self.high_enter_deg):
            raise ValueError('푸시업 각도 임계값의 순서가 올바르지 않습니다.')
        if not (0 <= self.min_visibility <= 1 and 0 <= self.min_presence <= 1
                and 0 < self.max_torso_inclination_deg < 90
                and 0 < self.max_leg_inclination_deg < 90):
            raise ValueError('관절 신뢰도 또는 신체 방향 설정이 올바르지 않습니다.')
        if (type(self.transition_samples) is not int or self.transition_samples < 2
                or self.transition_duration_sec <= 0 or self.max_frame_gap_sec <= 0
                or not 0 < self.occlusion_grace_sec <= self.max_frame_gap_sec
                or not 0 < self.min_rep_duration_sec < self.max_rep_duration_sec
                or self.min_shoulder_drop_ratio <= 0 or self.max_shoulder_return_ratio <= 0):
            raise ValueError('안정화 시간 또는 어깨 이동량 설정이 올바르지 않습니다.')


# 팔 선택의 필수 관절은 Shoulder/Elbow/Wrist입니다. Hip은 몸통 관측에 하나가 필요합니다.
_SIDES = {'left': (11, 13, 15, 23, 27), 'right': (12, 14, 16, 24, 28)}


class PushUpAnalyzer:
    def __init__(self, exercise_name='push_up', exercise_type='repetition',
                 target_duration_sec=None, *, config=None):
        if exercise_name != 'push_up' or exercise_type != 'repetition' or target_duration_sec is not None:
            raise ValueError('PushUpAnalyzer는 push_up/repetition 설정만 지원합니다.')
        self.config = config if config is not None else PushUpConfig()
        # 객체 생성 시에만 읽습니다. 진단은 판정에 관여하지 않으며 기본적으로 꺼져 있습니다.
        # 새 표기 PUSHUP_DEBUG를 우선하며 기존 표기도 호환합니다.
        self._debug_enabled = os.getenv('PUSHUP_DEBUG', os.getenv('PUSH_UP_DEBUG', '0')) == '1'
        self._debug_last = -math.inf
        self._debug_change = None
        self._debug_frame = {}
        self._instance_id = uuid4().hex
        self._event_number = 0
        self.reset()
        if self._debug_enabled:
            # 실제 로드 경로와 생성된 Config를 남겨 이전 프로세스/설정 혼동을 구분합니다.
            try:
                logging.getLogger('uvicorn.error').info('[PushUpDebug] %s', json.dumps({
                    'kind': 'initialization', 'analyzer_called': False, 'config_created': True,
                    'analyzer_created': True, 'source_file': __file__, 'thresholds': asdict(self.config),
                }, ensure_ascii=False, allow_nan=False))
            except Exception:
                pass

    def _clear_motion(self):
        self.phase = 'unknown'
        self.selected_side = None
        self._candidate = None
        self._candidate_since = None
        self._candidate_samples = 0
        self._cycle_started = None
        self._high_shoulder_y = None
        self._torso_length = None
        self._last_observed_timestamp = None
        self._missing_since = None

    def reset(self):
        # 완료 이벤트 번호는 초기화하지 않아 같은 세대의 ID 재사용도 막습니다.
        self._clear_motion()
        self._context = None
        self._last_timestamp = None

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _result(self, timestamp, *, invalid=False, event=None):
        return AnalysisResult('push_up', 'repetition', 'invalid' if invalid else self.phase,
                              timestamp, repetition_event=event)

    def _joint(self, points, ident):
        # 중복 ID도 유효 관측으로 간주하지 않습니다. 반대 팔에는 영향을 주지 않습니다.
        found = [p for p in points if isinstance(p, dict) and type(p.get('id')) is int
                 and p['id'] == ident]
        if len(found) != 1:
            return None
        point = found[0]
        if any(not self._finite(point.get(axis)) or not 0 <= point[axis] <= 1 for axis in ('x', 'y')):
            return None
        visibility, presence = point.get('visibility'), point.get('presence')
        if (not self._finite(visibility) or not self.config.min_visibility <= visibility <= 1
                or (presence is not None and (not self._finite(presence)
                                             or not self.config.min_presence <= presence <= 1))):
            return None
        return point

    def _features(self, points, side, size):
        joints = [self._joint(points, ident) for ident in _SIDES[side]]
        shoulder, elbow, wrist, hip, ankle = joints
        detail = {}
        if self._debug_enabled:
            self._debug_frame[side] = detail
        if any(p is None for p in joints[:3]):
            if self._debug_enabled:
                detail['invalid_reason'] = 'required_joint_invalid'
                detail['invalid_joint_ids'] = [i for i, p in zip(_SIDES[side][:3], joints[:3]) if p is None]
            return None
        width, height = size
        def vector(first, second):
            return ((second['x']-first['x'])*width, (second['y']-first['y'])*height)
        def horizontal(first, second, limit):
            dx, dy = vector(first, second)
            length = math.hypot(dx, dy)
            return length > 1e-6 and abs(dy)/length <= math.sin(math.radians(limit))
        upper, lower = vector(elbow, shoulder), vector(elbow, wrist)
        norm = math.hypot(*upper)*math.hypot(*lower)
        if norm <= 1e-9:
            if self._debug_enabled:
                detail['invalid_reason'] = 'degenerate_elbow_vectors'
            return None
        cosine = sum(a*b for a, b in zip(upper, lower))/norm
        angle = math.degrees(math.acos(max(-1., min(1., cosine))))
        # 선택 팔 쪽 골반을 우선합니다. 보이지 않으면 유효한 반대 골반으로 몸통만 확인합니다.
        hip_side = side
        if hip is None:
            hip_side = 'right' if side == 'left' else 'left'
            hip = self._joint(points, _SIDES[hip_side][3])
            ankle = self._joint(points, _SIDES[hip_side][4])
        if hip is None:
            if self._debug_enabled:
                detail.update(elbow_angle_deg=angle, invalid_reason='no_valid_hip')
            return None
        torso_ok = horizontal(shoulder, hip, self.config.max_torso_inclination_deg)
        leg_ok = ankle is None or horizontal(hip, ankle, self.config.max_leg_inclination_deg)
        if self._debug_enabled:
            dx, dy = vector(shoulder, hip)
            leg_vector = vector(hip, ankle) if ankle is not None else None
            detail.update(elbow_angle_deg=angle, shoulder_y_px=shoulder['y']*height,
                          torso_length_px=math.hypot(dx, dy),
                          torso_inclination_deg=math.degrees(math.atan2(abs(dy), abs(dx))),
                          torso_valid=torso_ok, ankle_observed=ankle is not None, leg_valid=leg_ok)
            detail['leg_inclination_deg'] = (math.degrees(math.atan2(abs(leg_vector[1]), abs(leg_vector[0])))
                                            if leg_vector is not None else None)
            detail['body_hip_side'] = hip_side
        # 측면의 다리 추정은 불안정할 수 있습니다. 기울기는 진단에만 남기고 팔을 배제하지 않습니다.
        if self._debug_enabled:
            detail['leg_check_blocking'] = False
        if not torso_ok:
            if self._debug_enabled:
                detail['invalid_reason'] = 'torso_direction'
            return None
        quality = min(min(p['visibility'], p.get('presence') if p.get('presence') is not None else 1.)
                      for p in joints[:3])
        return angle, shoulder['y']*height, math.hypot(*vector(shoulder, hip)), quality

    def _joint_diagnostics(self, points):
        """판정용 입력을 수정하지 않고 양쪽 필수/보조 관절의 거부 사유를 설명합니다."""
        result = {}
        for side, ids in _SIDES.items():
            for name, ident in zip(('shoulder', 'elbow', 'wrist', 'hip', 'ankle'), ids):
                found = [p for p in points if isinstance(p, dict) and type(p.get('id')) is int
                         and p['id'] == ident]
                point = found[0] if len(found) == 1 else {}
                reason = ('missing' if not found else 'duplicate' if len(found) > 1
                          else 'coordinates_invalid' if any(not self._finite(point.get(a))
                              or not 0 <= point[a] <= 1 for a in ('x', 'y'))
                          else 'visibility_below_threshold_or_invalid'
                              if not self._finite(point.get('visibility'))
                              or not self.config.min_visibility <= point['visibility'] <= 1
                          else 'presence_below_threshold_or_invalid'
                              if point.get('presence') is not None and (not self._finite(point['presence'])
                                  or not self.config.min_presence <= point['presence'] <= 1)
                          else None)
                # NaN 등 비정상 입력도 로그 전체를 유실하지 않도록 문자열로 표현합니다.
                values = {k: (point.get(k) if point.get(k) is None or self._finite(point.get(k))
                              else str(point.get(k))) for k in ('x', 'y', 'visibility', 'presence')}
                result[f'{side}_{name}'] = dict(id=ident, **values, valid=reason is None,
                                              reason=reason, optional=name in ('hip', 'ankle'))
        return result

    def _stable(self, candidate, timestamp):
        if candidate is None:
            self._candidate = self._candidate_since = None
            self._candidate_samples = 0
            return False
        if candidate != self._candidate:
            self._candidate, self._candidate_since, self._candidate_samples = candidate, timestamp, 1
        else:
            self._candidate_samples += 1
        return (self._candidate_samples >= self.config.transition_samples
                and timestamp-self._candidate_since + 1e-9 >= self.config.transition_duration_sec)

    def _analyze(self, keypoints, context):
        if context != self._context:
            self.reset()
            self._context = context
        timestamp = getattr(keypoints, 'timestamp', None)
        size = getattr(keypoints, 'frame_size', None)
        if (getattr(context, 'exercise_name', None) != 'push_up'
                or not self._finite(timestamp)
                or not isinstance(size, (tuple, list)) or len(size) != 2
                or any(not self._finite(v) or v <= 0 for v in size)):
            if self._debug_enabled:
                self._debug_frame['reason'] = 'invalid_context_or_frame_metadata'
            self._clear_motion()
            return self._result(timestamp if self._finite(timestamp) else time.monotonic(), invalid=True)
        if self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                if self._debug_enabled:
                    self._debug_frame['reason'] = 'timestamp_reversed'
                self._clear_motion()
                return self._result(timestamp, invalid=True)
            if timestamp == self._last_timestamp:
                if self._debug_enabled:
                    self._debug_frame['reason'] = 'duplicate_timestamp'
                return self._result(timestamp)  # 중복 표본은 안정화 시간/표본 수를 진행시키지 않습니다.
            if timestamp-self._last_timestamp > self.config.max_frame_gap_sec:
                if self._debug_enabled:
                    self._debug_frame['reset_reason'] = 'frame_gap_exceeded'
                self._clear_motion()
        self._last_timestamp = timestamp
        if self._cycle_started is not None and timestamp-self._cycle_started > self.config.max_rep_duration_sec:
            if self._debug_enabled:
                self._debug_frame['reset_reason'] = 'cycle_timeout'
            self._clear_motion()

        # HIGH 후보 안정화부터 선택을 고정해 품질 순위 변화로 표본이 계속 초기화되지 않게 합니다.
        features = None
        if self.selected_side is not None:
            features = self._features(keypoints, self.selected_side, size)
            if features is None:
                if self._missing_since is None:
                    self._missing_since = self._last_observed_timestamp
                elapsed = timestamp-self._missing_since
                self._stable(None, timestamp)  # 누락 이전 안정화 시간을 재사용하지 않습니다.
                if self._debug_enabled:
                    self._debug_frame.update(reason='selected_side_invalid', selected_side=self.selected_side,
                                             occlusion_elapsed_sec=elapsed,
                                             selection_reason='occlusion_grace' if elapsed <= self.config.occlusion_grace_sec
                                                              else 'occlusion_timeout')
                if elapsed > self.config.occlusion_grace_sec:
                    self._clear_motion()
                return self._result(timestamp, invalid=True)
            if self._missing_since is not None:
                if timestamp-self._missing_since > self.config.occlusion_grace_sec:
                    self._clear_motion()
                    features = None
                    if self._debug_enabled:
                        self._debug_frame['selection_reason'] = 'occlusion_timeout'
                else:
                    self._missing_since = None
                    if self._debug_enabled:
                        self._debug_frame['selection_reason'] = 'selected_arm_recovered'

        if self.selected_side is None:
            valid = [(side, observed) for side in _SIDES
                     if (observed := self._features(keypoints, side, size)) is not None]
            high_arms = [(side, observed) for side, observed in valid
                         if observed[0] >= self.config.high_enter_deg]
            if not high_arms:
                self._clear_motion()
                if self._debug_enabled:
                    self._debug_frame.update(reason='waiting_for_high' if valid else 'no_valid_side',
                                             selection_reason='no_stable_high_candidate', selected_side=None)
                return self._result(timestamp, invalid=not valid)
            selected, features = max(high_arms, key=lambda item: item[1][3])
            if selected != self.selected_side:
                self._stable(None, timestamp)  # 다른 팔의 HIGH 표본을 합쳐 안정화하지 않습니다.
                if self._debug_enabled:
                    self._debug_frame['selection_changed'] = True
                    self._debug_frame['selection_reason'] = 'highest_quality_high_arm'
            self.selected_side = selected
        self._last_observed_timestamp = timestamp
        angle, shoulder_y, torso_length, _ = features
        cfg = self.config
        high = angle >= cfg.high_enter_deg
        drop = ((shoulder_y-self._high_shoulder_y)/self._torso_length
                if self._high_shoulder_y is not None else 0.)
        candidate = None
        if self.phase == 'unknown':
            if high:
                candidate = 'high'
        elif self.phase == 'high':
            if angle <= cfg.descending_enter_deg:
                candidate = 'descending'
        elif self.phase == 'descending':
            if angle <= cfg.bottom_enter_deg and drop >= cfg.min_shoulder_drop_ratio:
                candidate = 'bottom'
            elif high:
                candidate = 'high'  # 부분 동작 복귀에는 이벤트를 생성하지 않습니다.
        elif self.phase == 'bottom':
            if angle >= cfg.ascending_enter_deg:
                candidate = 'ascending'
        elif self.phase == 'ascending':
            if high and abs(drop) <= cfg.max_shoulder_return_ratio:
                candidate = 'high'
            elif angle <= cfg.bottom_enter_deg:
                candidate = 'bottom'

        event = None
        stable = self._stable(candidate, timestamp)
        if self._debug_enabled:
            reason = ('waiting_for_high' if self.phase == 'unknown'
                      else 'bottom_angle_not_met' if self.phase == 'descending' and angle > cfg.bottom_enter_deg
                      else 'shoulder_drop_below_minimum' if self.phase == 'descending' and drop < cfg.min_shoulder_drop_ratio
                      else 'shoulder_not_returned' if self.phase == 'ascending' and high and abs(drop) > cfg.max_shoulder_return_ratio
                      else 'stabilizing' if candidate and not stable else 'waiting_for_phase_condition')
            self._debug_frame.update(
                selected_side=self.selected_side, elbow_angle_deg=angle,
                shoulder_drop_px=(shoulder_y-self._high_shoulder_y if self._high_shoulder_y is not None else None),
                shoulder_drop_ratio=drop, baseline_torso_length_px=self._torso_length,
                shoulder_return_ratio=abs(drop),
                bottom_angle_met=angle <= cfg.bottom_enter_deg,
                bottom_drop_met=self._high_shoulder_y is not None and drop >= cfg.min_shoulder_drop_ratio,
                bottom_phase_met=self.phase == 'descending', candidate=candidate,
                candidate_samples=self._candidate_samples,
                candidate_duration_sec=(timestamp-self._candidate_since if self._candidate_since is not None else 0.),
                transition_stable=stable, reason='transition' if stable else reason)
            # 조건 충족과 현재 phase에서 허용되는 전환을 구분해 중간 상태 누락도 드러냅니다.
            self._debug_frame['conditions'] = {
                'high': {'angle_met': high, 'shoulder_return_met': abs(drop) <= cfg.max_shoulder_return_ratio,
                         'met': high and (self.phase != 'ascending' or abs(drop) <= cfg.max_shoulder_return_ratio),
                         'phase_allowed': self.phase in ('unknown', 'descending', 'ascending')},
                'descending': {'met': angle <= cfg.descending_enter_deg, 'phase_allowed': self.phase == 'high'},
                'bottom': {'angle_met': angle <= cfg.bottom_enter_deg,
                           'drop_met': self._high_shoulder_y is not None and drop >= cfg.min_shoulder_drop_ratio,
                           'met': angle <= cfg.bottom_enter_deg and (self.phase == 'ascending'
                               or self._high_shoulder_y is not None and drop >= cfg.min_shoulder_drop_ratio),
                           'phase_allowed': self.phase in ('descending', 'ascending')},
                'ascending': {'met': angle >= cfg.ascending_enter_deg, 'phase_allowed': self.phase == 'bottom'},
            }
            failures = []
            if candidate and not stable:
                if self._candidate_samples < cfg.transition_samples:
                    failures.append('insufficient_distinct_samples')
                if timestamp-self._candidate_since + 1e-9 < cfg.transition_duration_sec:
                    failures.append('insufficient_stability_duration')
            elif candidate is None:
                failures = (['high_angle_not_met'] if self.phase == 'unknown'
                            else ['descending_angle_not_met'] if self.phase == 'high'
                            else (['bottom_angle_not_met'] if angle > cfg.bottom_enter_deg else [])
                                 + (['shoulder_drop_below_minimum'] if drop < cfg.min_shoulder_drop_ratio else [])
                                 if self.phase == 'descending'
                            else ['ascending_angle_not_met'] if self.phase == 'bottom'
                            else (['high_angle_not_met'] if not high else ['shoulder_not_returned']))
            self._debug_frame['transition_failures'] = failures
        if stable:
            previous = self.phase
            if candidate == 'descending':
                self._cycle_started = self._candidate_since
            elif candidate == 'high':
                if self._debug_enabled and previous == 'ascending':
                    self._debug_frame['completion_rejection'] = (
                        'cycle_not_started' if self._cycle_started is None else
                        'cycle_too_short' if timestamp-self._cycle_started < cfg.min_rep_duration_sec else None)
                if (previous == 'ascending' and self._cycle_started is not None
                        and cfg.min_rep_duration_sec <= timestamp-self._cycle_started <= cfg.max_rep_duration_sec):
                    self._event_number += 1
                    event = RepCompleted(f'push_up:{self._instance_id}:{self._event_number}')
                self._cycle_started = None
                self._high_shoulder_y, self._torso_length = shoulder_y, torso_length
            self.phase = candidate
            self._stable(None, timestamp)
        return self._result(timestamp, event=event)


    def analyze(self, keypoints, context):
        if not self._debug_enabled:
            return self._analyze(keypoints, context)
        before = self.phase
        timestamp = getattr(keypoints, 'timestamp', None)
        self._debug_frame = {
            'kind': 'analyzer', 'analyzer_called': True,
            'session_id': getattr(context, 'session_id', None),
            'exercise_id': getattr(context, 'exercise_id', None),
            'timestamp': timestamp if self._finite(timestamp) else str(timestamp),
            'previous_phase': before,
            'selected_side': self.selected_side,
            'elbow_angle_deg': None,
            'shoulder_drop_ratio': None, 'shoulder_return_ratio': None,
            'candidate': None, 'candidate_samples': 0, 'candidate_duration_sec': 0.,
            'conditions': {name: {'met': None, 'phase_allowed': None}
                           for name in ('high', 'descending', 'bottom', 'ascending')},
            'joints': self._joint_diagnostics(keypoints),
            'frame_gap_sec': (timestamp-self._last_timestamp
                              if self._finite(timestamp) and self._last_timestamp is not None else None),
        }
        result = self._analyze(keypoints, context)
        size = getattr(keypoints, 'frame_size', None)
        if (isinstance(size, (tuple, list)) and len(size) == 2
                and all(self._finite(v) and v > 0 for v in size)):
            for arm_side in _SIDES:
                if arm_side not in self._debug_frame:
                    self._features(keypoints, arm_side, size)
        event = result.repetition_event
        self._debug_frame.update(phase=result.phase, rep_completed=event is not None,
                                 next_phase=self.phase, input_valid=result.phase != 'invalid',
                                 event_id=event.event_id if event else None)
        side = self._debug_frame.get('selected_side')
        detail = self._debug_frame.get(side, {})
        if self._debug_frame['elbow_angle_deg'] is None:
            self._debug_frame['elbow_angle_deg'] = detail.get('elbow_angle_deg')
        # 골반/다리 때문에 무효여도 유효한 팔의 각도는 진단에 남깁니다. 판정에는 재사용하지 않습니다.
        size = getattr(keypoints, 'frame_size', None)
        if (side in _SIDES and self._debug_frame['elbow_angle_deg'] is None
                and isinstance(size, (tuple, list)) and len(size) == 2
                and all(self._finite(v) and v > 0 for v in size)):
            joints = self._debug_frame['joints']
            arm = [joints[f'{side}_{name}'] for name in ('shoulder', 'elbow', 'wrist')]
            if all(p['valid'] for p in arm):
                shoulder, elbow, wrist = arm
                upper = ((shoulder['x']-elbow['x'])*size[0], (shoulder['y']-elbow['y'])*size[1])
                lower = ((wrist['x']-elbow['x'])*size[0], (wrist['y']-elbow['y'])*size[1])
                norm = math.hypot(*upper)*math.hypot(*lower)
                if norm > 1e-9:
                    cosine = sum(a*b for a,b in zip(upper, lower))/norm
                    self._debug_frame['elbow_angle_deg'] = math.degrees(math.acos(max(-1., min(1., cosine))))
        self._debug_frame['invalid_reason'] = (
            detail.get('invalid_reason', self._debug_frame.get('reason')) if result.phase == 'invalid' else None)
        # 품질은 몸통과 분리해 세 팔 관절만으로 계산하고, 기존 로그 필드도 호환합니다.
        for arm_side in _SIDES:
            arm = [self._debug_frame['joints'][f'{arm_side}_{name}']
                   for name in ('shoulder', 'elbow', 'wrist')]
            valid = all(p['valid'] for p in arm) and self._debug_frame.get(arm_side, {}).get(
                'invalid_reason') != 'degenerate_elbow_vectors'
            self._debug_frame[arm_side+'_arm_quality'] = (min(
                min(p['visibility'], p['presence'] if p['presence'] is not None else 1.) for p in arm)
                if valid else None)
        self._debug_frame.update(selected_arm=self._debug_frame.get('selected_side'),
                                 selected_elbow_angle=self._debug_frame['elbow_angle_deg'],
                                 torso_inclination_deg=detail.get('torso_inclination_deg'),
                                 leg_inclination_deg=detail.get('leg_inclination_deg'),
                                 candidate_phase=self._debug_frame.get('candidate'))
        self._debug_frame.setdefault('transition_failures', [self._debug_frame['invalid_reason']]
            if result.phase == 'invalid' else ['high_angle_not_met']
            if self._debug_frame.get('reason') == 'waiting_for_high' else [])
        self._debug_frame.setdefault('selection_reason', 'selected_arm_locked' if self.selected_side else 'waiting_for_high')
        # 일반 거부 사유 변경은 주기 제한을 지킵니다. 상태 전환/입력 무효 사유 변경은 즉시 출력합니다.
        try:
            reason = ((self._debug_frame['invalid_reason'],
                       tuple((name, data['reason']) for name, data in self._debug_frame['joints'].items()
                             if data['reason'] is not None),
                       tuple(self._debug_frame.get(side, {}).get('invalid_reason') for side in _SIDES))
                      if result.phase == 'invalid' else None)
            changed = (result.phase, reason) != self._debug_change
            now = time.monotonic()
            if (changed or before != self.phase or event is not None
                    or self._debug_frame.get('selection_changed') or now-self._debug_last >= 1.):
                logging.getLogger('uvicorn.error').info('[PushUpDebug] %s', json.dumps(
                    {**self._debug_frame, 'thresholds': asdict(self.config)}, ensure_ascii=False, allow_nan=False))
                self._debug_change, self._debug_last = (result.phase, reason), now
        except Exception:
            pass
        return result
