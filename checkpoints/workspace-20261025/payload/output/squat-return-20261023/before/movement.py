"""버튼 시각과 실제 동작 구간을 분리하는 실험용 반복 상태 머신.

스쿼트는 서기·하강·서기 복귀, 프레스는 하단·상단·하단,
푸시업은 팔 펴기·굽히기·펴기를 관측한다. 품질 판정과 횟수는 별개다.
"""
from collections import deque
import numpy as np


def joint_angle(points, a, b, c):
    """세 픽셀 관절의 투영 각도(도)를 반환하며 누락은 None이다."""
    if not np.isfinite(points[[a, b, c]]).all():
        return None
    u, v = points[a]-points[b], points[c]-points[b]
    norm = np.linalg.norm(u)*np.linalg.norm(v)
    if norm < 1:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/norm, -1, 1))))


class MovementTracker:
    """10Hz 실제 관측에서 준비·동작·복귀 구간을 분리한다.

    누락 0.4초·표본 간격 0.4초 초과·구간 15초 초과 시 미완료 반복을
    폐기한다. 시작부터 굽힌 자세이거나 서서 기다리는 시간은 세지 않는다.
    문턱은 실시간 검증 중인 공학 초안이며 정확도 보장 기준이 아니다.
    """
    def __init__(self, exercise, view="front"):
        self.exercise = exercise
        self.view = view
        self.base_center = None
        self.base_scale = None
        self.hip_drop = None
        self.top_confirm = 0
        self.phase = '준비 자세 확인 중'
        self.count = 0
        self.uncertain_attempts = []
        self.ready = 0
        self.bottom = 0
        self.returned = 0
        self.missing = 0
        self.active = False
        self.has_bottom = False
        self.buffer = []
        self.pre = deque(maxlen=8)
        self.last_tick = None
        self.base_hip = None
        self.base_feet = None
        self.movement_seconds = 0.0
        self.hold_seconds = 0.0
        self.angle = None
        self.top_confirm = 0

    def reset_attempt(self, message):
        """완료 횟수를 유지하며 미완료 동작과 준비 기준을 초기화한다."""
        if self.exercise=='squats' and self.view=='front' and self.active and self.buffer and self.last_tick-self.buffer[0][0]>=8 and any(word in message for word in ('관측','위치','초과','투영')):
            self.uncertain_attempts.append(dict(start_tick=int(self.buffer[0][0]),end_tick=int(self.last_tick),reason=message,counted=False))
        self.active = False
        self.ready = self.bottom = self.returned = 0
        self.has_bottom = False
        self.buffer = []
        self.pre.clear()
        self.base_hip = self.base_feet = None
        self.base_center = self.base_scale = None
        self.hip_drop = None
        self.top_confirm = 0
        self.phase = message

    def push(self, tick, xy):
        """표본을 반영해 완료 구간만 반환한다. 미완료·대기는 None이다.

        Args:
            tick: 실제 관측 10Hz 슬롯 정수.
            xy: 33×2 픽셀 좌표. 누락은 NaN.

        Returns:
            list | None: 자동 경계 검출한 한 반복의 관측 표본.
        """
        if self.last_tick is not None and tick-self.last_tick > 4:
            self.reset_attempt('관측 끊김 · 준비 자세를 다시 확인하세요')
        previous_tick = self.last_tick
        self.last_tick = tick
        if self.exercise == 'ytw':
            self.phase = 'YTW 자동 반복 구분은 아직 미지원'
            return None
        required = [11, 12, 23, 24]
        if self.exercise == 'squats':
            if self.view != 'front': required += [25,26,27,28]
            elif not np.isfinite(xy[[25,26]]).all(1).any(): required += [25,26]
        elif self.exercise == 'plank':
            required += [25, 26, 27, 28]
        else:
            required += [13, 14, 15, 16]
        if not np.isfinite(xy[required]).all():
            self.missing += 1
            self.phase = '필요 관절 미인식 · 카메라 구도를 맞추세요'
            if self.missing >= 4:
                self.reset_attempt('관측 중단 · 준비 자세 재확인')
            return None
        self.missing = 0
        shoulder = (xy[11]+xy[12])/2
        hip = (xy[23]+xy[24])/2
        scale = max(30., float(np.linalg.norm(shoulder-hip)))
        if self.exercise == 'plank':
            angles = [joint_angle(xy,11,23,27), joint_angle(xy,12,24,28)]
            aligned = all(a is not None and a >= 150 for a in angles)
            if aligned and previous_tick is not None and tick-previous_tick <= 2:
                self.hold_seconds += (tick-previous_tick)/10
            self.phase = '정렬 관측 · 유지 시간 측정 중' if aligned else '유지 자세 대기 · 몸통 정렬 확인'
            return None
        joints = [(23,25,27),(24,26,28)] if self.exercise == 'squats' else [(11,13,15),(12,14,16)]
        values = [joint_angle(xy,*j) for j in joints]
        valid_angles = [v for v in values if v is not None]
        angle = float(np.mean(valid_angles)) if valid_angles else None
        self.angle = angle
        if self.exercise == 'squats' and self.view == 'front':
            # 정면 무릎 투영 각도 대신 고정한 서기 기준의 골반 이동을 사용한다.
            upright = abs(shoulder[0]-hip[0]) < scale*.35
            knee_sides=[j for j in [0,1] if np.isfinite(xy[25+j]).all()]
            knees_under_hips = bool(knee_sides) and max(abs(xy[25+j,0]-xy[23+j,0]) for j in knee_sides) < scale*.35
            knees_below = bool(knee_sides) and float(np.mean([xy[25+j,1] for j in knee_sides])-hip[1]) > scale*.5
            at_start = upright and knees_under_hips and knees_below
            at_bottom = False
            if self.base_hip is not None:
                drop = (hip[1]-self.base_hip)/self.base_scale
                self.hip_drop = float(drop)
                if abs(hip[0]-self.base_center) > self.base_scale*.45 or not .4 < scale/self.base_scale < 2.5:
                    self.reset_attempt('위치 이동 감지 · 준비 자세 재확인' if abs(hip[0]-self.base_center)>self.base_scale*.45 else '몸통 투영 변화 · 구간 확인 필요')
                    return None
                # 시작 시 살짝 굽혔다가 더 높게 서는 경우 새 서기 기준을 확인한다.
                # 하강 방향으로는 기준을 따라 바꾸지 않는다.
                self.top_confirm = self.top_confirm+1 if drop < -.10 and upright and not self.has_bottom else 0
                if self.top_confirm >= 3:
                    self.base_hip = float(hip[1])
                    self.base_center = float(hip[0])
                    self.base_scale = scale
                    self.active = False
                    self.buffer = []
                    self.pre.clear()
                    self.ready = 6
                    self.top_confirm = 0
                    self.hip_drop = 0.0
                    self.phase = '준비 완료 · 서기 기준 보정'
                    return None
                at_start = -.35 < drop < (.07 if self.has_bottom else .09) and upright
                at_bottom = drop > .08
        else:
            if angle is None:
                return None
            if self.exercise == 'shoulder':
                at_start, at_bottom = angle <= 110, angle >= 150
            else:
                at_start, at_bottom = angle >= 157, angle <= 140
            if self.exercise == 'squats' and self.base_hip is not None:
                feet = (xy[27]+xy[28])/2
                if np.linalg.norm(feet-self.base_feet) > scale*.45:
                    self.reset_attempt('위치 이동 감지 · 준비 자세 재확인')
                    return None
                at_bottom = at_bottom and hip[1]-self.base_hip > scale*.12
                at_start = at_start and abs(hip[1]-self.base_hip) < scale*.18
        sample = (tick, xy.copy())
        if not self.active:
            self.pre.append(sample)
            self.ready = self.ready+1 if at_start else 0
            if self.ready >= 6:
                self.phase = '준비 완료 · 동작을 시작하세요'
                if self.exercise == 'squats' and self.base_hip is None:
                    self.base_hip = float(hip[1])
                    self.base_center = float(hip[0])
                    self.base_scale = scale
                    self.base_feet = (xy[27]+xy[28])/2 if np.isfinite(xy[[27,28]]).all() else None
            if self.ready == 0 and self.base_hip is not None and self.exercise == 'squats':
                self.active = True
            elif self.exercise != 'squats' and self.ready == 0 and self.phase == '준비 완료 · 동작을 시작하세요':
                self.active = True
            if not self.active:
                return None
            self.buffer = list(self.pre)
            self.phase = '동작 진행 중'
        else:
            self.buffer.append(sample)
        if tick-self.buffer[0][0] > 150:
            self.reset_attempt('동작 구간 초과 · 준비 자세 재확인')
            return None
        self.bottom = self.bottom+1 if at_bottom else 0
        if self.bottom >= 2:
            self.has_bottom = True
            self.phase = '하강 확인 · 복귀 대기' if self.exercise != 'shoulder' else '상승 확인 · 하단 복귀 대기'
        # 미완료 작은 시도를 서기 복귀에서 닫아 대기 시간을 다음 회차와 합치지 않는다.
        if self.exercise=='squats' and self.view=='front' and not self.has_bottom:
            self.returned=self.returned+1 if at_start else 0
            if self.returned>=3:
                baseline=(self.base_hip,self.base_center,self.base_scale,self.base_feet)
                self.reset_attempt('작은 움직임 종료 · 다음 동작 대기')
                self.base_hip,self.base_center,self.base_scale,self.base_feet=baseline
                self.ready=6
                return None
        else:
            self.returned = self.returned+1 if self.has_bottom and at_start else 0
        if self.returned >= 3 and tick-self.buffer[0][0] >= 8:
            segment = self.buffer[:]
            self.count += 1
            self.movement_seconds += (segment[-1][0]-segment[0][0])/10
            baseline = (self.base_hip,self.base_center,self.base_scale,self.base_feet)
            self.reset_attempt('반복 완료 · 다음 준비 자세 확인 중')
            if self.exercise == 'squats' and self.view == 'front':
                self.base_hip,self.base_center,self.base_scale,self.base_feet = baseline
                self.ready = 6
                self.phase = '준비 완료 · 다음 동작 대기'
            return segment
        return None

    def status(self):
        """표시용 상태를 반환한다. 반복 미지원 종목은 횟수를 None으로 둔다."""
        return dict(detection_version='front-hysteresis-v2',uncertain_attempts=list(self.uncertain_attempts),phase=self.phase, count=None if self.exercise in ('ytw','plank') else self.count,
                    angle=self.angle, hip_drop=self.hip_drop, active=self.active,
                    movement_seconds=round(self.movement_seconds,1),
                    hold_seconds=round(self.hold_seconds,1),
                    scope='experimental_2d_motion_boundaries')
