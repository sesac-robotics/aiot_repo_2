"""독립 YTW 시간 분절 후보. 공통 앱에 자동 설치하거나 모델을 학습하지 않는다.

Y 지정 시작 → T 유지 → W 종료 유지가 한 순서 시도다. 팔 내림/기립
복귀를 발명하지 않는다. 순서 완료는 정상 모양이나 0/1/2 등급이 아니다.
"""
import numpy as np
from live_observation import mask_frame_border, segment_coverage

VERSION = 'ytw-ready-run-origin-fix-v2-candidate'
REQUIRED = [11, 12, 13, 14, 15, 16, 23, 24]


class LearnedPhaseSource:
    """호출자가 신뢰해 로드한 개발 bundle 사용. 온라인 fit/파일쓰기 없음."""
    def __init__(self, bundle):
        import hashlib
        from pathlib import Path
        from app.scoring import initial_pose_return
        from app.scoring.initial_pose_return import pose
        if hashlib.sha256(Path(initial_pose_return.__file__).read_bytes()).hexdigest() != bundle.get("feature_implementation_sha256"):
            raise ValueError("State feature implementation hash mismatch")
        self.pose = pose
        if bundle.get('schema') != 'reviewed_initial_ready_and_return_state_v3':
            raise ValueError('Unsupported trusted bundle')
        self.phase = bundle['heads'].get(('ytw', 'x', 'phase'))
        self.ready = bundle['heads'].get(('ytw', 'x', 'target'))
        if not all(h and h.get('state_gate_passed') is True for h in (self.phase, self.ready)):
            raise ValueError('YTW state development gates required')
        if set(self.phase['model'].classes_) != {'Y', 'T', 'W'}:
            raise ValueError('Y/T/W classes required')

    def __call__(self, xy):
        vector = self.pose('ytw', xy)
        if vector is None:
            return None, False
        # Class decision follows the existing analyzer; raw probabilities are not
        # treated as calibrated universal confidence scores.
        return str(self.phase['model'].predict([vector])[0]), bool(self.ready['model'].predict([vector])[0] == 1)


class YTWTracker:
    """push(tick,xy,width,height) -> event|None; tick is actual integer 10Hz slot.

    Five consecutive observed slots confirm each phase (0.4s span). Gaps do not
    accumulate dwell. >=4 missing slots or >3 slot jump invalidate an attempt.
    Missing slots stay NaN in score input; stage windows are AUTO candidates,
    never direct-review phase_windows. Only front/upright_Y_T_W_v1 is supported.
    """
    def __init__(self, phase_source, view='front', protocol='upright_Y_T_W_v1', dwell=5):
        if dwell < 3:
            raise ValueError('At least three consecutive slots required')
        self.source, self.view, self.protocol, self.dwell = phase_source, view, protocol, dwell
        self.count = 0
        self.last_tick = None
        self.events = []
        self.rejected = []
        self._reset('waiting_Y')

    def _reset(self, reason):
        self.phase = reason
        self.expected = 'Y'
        self.current = None
        self.streak = 0
        self.run_start = None
        self.samples = []
        self.windows = {}
        self.missing = 0
        self.active = False

    def _reject(self, reason):
        if self.active:
            self.rejected.append(dict(reason=reason, start_tick=self.samples[0][0], end_tick=self.last_tick, counted=False, grade=None))
        self._reset(reason)

    def push(self, tick, xy, width=640, height=480):
        if not isinstance(tick, (int, np.integer)) or tick < 0 or (self.last_tick is not None and tick <= self.last_tick):
            raise ValueError('Strictly increasing nonnegative integer slots required')
        previous = self.last_tick
        self.last_tick = int(tick)
        if self.view != 'front' or self.protocol != 'upright_Y_T_W_v1':
            self._reject('unsupported_view_or_protocol')
            return None
        if previous is not None and tick - previous > 3:
            self._reject('observation_gap')
        elif previous is not None and tick - previous != 1:
            self.streak = 0
            self.current = None
        points = mask_frame_border(xy, width, height)
        valid = np.isfinite(points[REQUIRED]).all()
        if not valid:
            self.streak = 0
            self.current = None
            self.missing += 1
            if self.active:
                self.samples.append((int(tick), points))
            if self.missing >= 4:
                self._reject('required_joints_unobserved')
            return None
        self.missing = 0
        label, ready = self.source(points)
        if label not in ('Y', 'T', 'W'):
            self.current = None
            self.streak = 0
            if self.active:
                self.samples.append((int(tick), points))
            return None
        if label == self.current:
            self.streak += 1
        else:
            self.current, self.streak, self.run_start = label, 1, int(tick)
        if self.active:
            self.samples.append((int(tick), points))
            if tick-self.samples[0][0] > 300:
                self._reject('attempt_timeout')
                return None
        # Readiness must persist through the whole Y dwell, not just final frame.
        if not self.active and (label != 'Y' or not ready):
            self.streak = 0
            self.current = None
            self.run_start = None
            self.samples = []
            return None
        if not self.active:
            self.samples.append((int(tick), points))
        if self.streak < self.dwell:
            return None
        if not self.active:
            self.active = True
            self.expected = 'T'
            self.windows['Y'] = [self.run_start, int(tick)]
            self.phase = 'Y_ready_observed'
            return None
        if label == self.expected:
            self.windows[label] = [self.run_start, int(tick)]
            if label == 'T':
                self.expected = 'W'
                self.phase = 'Y_then_T_observed'
                return None
            # W is the defined terminal phase, not an arms-down return.
            return self._complete(width, height)
        previous_phase = 'Y' if self.expected == 'T' else 'T'
        if label == previous_phase:
            self.windows[label][1] = int(tick)
            return None
        self._reject('out_of_order_phase')
        return None

    def _complete(self, width, height):
        start, end = self.samples[0][0], self.samples[-1][0]
        dense = np.full((end-start+1, 33, 2), np.nan)
        for tick, points in self.samples:
            dense[tick-start] = points
        valid = np.isfinite(dense[:, REQUIRED]).all((1, 2))
        coverage = float(valid.mean())
        if coverage < .8:
            self._reject('observation_below_80_percent')
            return None
        windows = {name: [a-start, b-start] for name, (a,b) in self.windows.items()}
        event = dict(version=VERSION, samples=[(start+i, p.copy()) for i,p in enumerate(dense)],
                     auto_phase_windows=windows, observation_coverage=coverage,
                     temporal_slot_coverage=segment_coverage(self.samples), start_tick=start,end_tick=end,
                     width=width,height=height, ordered_attempt_complete=True,
                     complete_attempt_reviewed=False, projection_reviewed=False,
                     phase_windows_reviewed=None, grade=None, quality_verified=False)
        self.count += 1
        # Retain only compact metadata in tracker memory.
        self.events.append({k:v for k,v in event.items() if k != 'samples'})
        self._reset('Y_T_W_order_completed_waiting_next_Y')
        return event

    def finish(self):
        self._reject('stream_ended_incomplete')
        return self.status()

    def status(self):
        return dict(version=VERSION, phase=self.phase, count=self.count, active=self.active,
                    expected=self.expected, rejected=list(self.rejected), grade=None,
                    scope='development_ordered_attempts_not_quality_or_severity')


def score_event(engine, event):
    """동일 dense segment로 기존 AI/Rule 호출. 수동 승인/총점은 생성하지 않음."""
    from app.scoring.sequence import PoseSequence
    xy = np.asarray([p for _,p in event['samples']])
    seq = PoseSequence(xy, np.arange(len(xy)), 10., event['width'], event['height'])
    return engine.score('ytw', seq, view='front', complete_attempt=False,
                        phase_windows=None, projection_reviewed=False)
