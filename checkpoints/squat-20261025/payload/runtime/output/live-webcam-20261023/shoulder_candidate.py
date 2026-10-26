"""독립 숄더 반복 후보. 모델/점수는 변경하지 않으며 완료된 실제 표본만 반환한다."""
from collections import deque
import numpy as np
from movement import joint_angle
from live_observation import mask_frame_border, movement_observation, segment_coverage

class ShoulderTracker:
    """양팔 하단→상단→하단. 문턱은 기능 검증용 공학 초안이다."""
    def __init__(self, exercise='shoulder', view='front'):
        if exercise != 'shoulder':
            raise ValueError('shoulder only')
        self.view = view
        self.count = 0
        self.last_tick = None
        self.movement_seconds = 0.
        self.standing_reference = None
        self.last_completion_coverage = None
        self.observation = None
        self.uncertain_attempts = []
        self._reset('준비 자세 확인 중')

    def _reset(self, phase):
        self.phase = phase
        self.active = False
        self.buffer = []
        self.pre = deque(maxlen=6)
        self.ready = self.top = self.returned = 0
        self.has_top = False
        self.previous_valid = None

    def push(self, tick, xy, width=640, height=480):
        if self.last_tick is not None and tick <= self.last_tick:
            raise ValueError('strictly increasing 10Hz slots required')
        if self.last_tick is not None and tick-self.last_tick > 4:
            self._reset('관측 끊김 · 준비 자세 재확인')
        self.last_tick = tick
        self.observation = movement_observation(xy,width,height,'shoulder',self.view)
        points = mask_frame_border(xy,width,height)
        if not self.observation['ready']:
            self._reset(self.observation['message'])
            return None
        # Confirmation holds require consecutive actual slots, never sample count alone.
        if self.previous_valid is not None and tick-self.previous_valid > 1:
            self.ready = self.top = self.returned = 0
            if not self.active:
                self.pre.clear()
        self.previous_valid = tick
        angles = [joint_angle(points,*j) for j in [(11,13,15),(12,14,16)]]
        if any(a is None for a in angles):
            self._reset('팔 투영 확인 필요 · 준비 자세 재확인')
            return None
        torso = max(30.,float(np.linalg.norm(points[[11,12]].mean(0)-points[[23,24]].mean(0))))
        wrist_lift = (points[[11,12],1]-points[[15,16],1])/torso
        elbow_lift = (points[[11,12],1]-points[[13,14],1])/torso
        low = max(angles) <= 110 and max(elbow_lift) < .2
        high = min(angles) >= 150 and min(wrist_lift) >= .35 and min(elbow_lift) >= .05
        sample = (int(tick),points.copy())
        if not self.active:
            if low:
                self.pre.append(sample)
                self.ready += 1
                self.phase = '준비 완료 · 양팔을 올리세요' if self.ready>=6 else '하단 준비 자세 확인 중'
                return None
            if self.ready < 6:
                self.ready = 0
                self.pre.clear()
                self.phase = '하단 준비 자세 확인 중'
                return None
            self.active = True
            self.buffer = list(self.pre)
        self.buffer.append(sample)
        self.phase = '하단 복귀 확인 중' if self.has_top else '상승 · 양팔 상단 확인 중'
        if tick-self.buffer[0][0] > 150:
            self._reset('구간 초과 · 미완료 동작 보류')
            return None
        self.top = self.top+1 if high else 0
        if self.top >= 2:
            self.has_top = True
        self.returned = self.returned+1 if self.has_top and low else 0
        if not self.has_top and low:
            self._reset('상단 미확인 · 하단에서 다시 준비하세요')
            return None
        if self.returned < 3:
            return None
        segment = list(self.buffer)
        self.last_completion_coverage = segment_coverage(segment)
        if self.last_completion_coverage < .8 or len(segment)<10:
            self._reset('관측 부족 · 완료 평가 보류')
            return None
        self.count += 1
        self.movement_seconds += (segment[-1][0]-segment[0][0])/10
        self._reset('반복 완료 · 하단 준비 자세 재확인')
        return segment

    def status(self):
        return dict(detection_version='shoulder-bilateral-overhead-v1-candidate',
                    phase=self.phase,count=self.count,active=self.active,
                    observation=self.observation,completion_observation_coverage=self.last_completion_coverage,
                    movement_seconds=round(self.movement_seconds,1),hold_seconds=0.,
                    uncertain_attempts=self.uncertain_attempts,
                    scope='engineering_candidate_not_accuracy_validation')
