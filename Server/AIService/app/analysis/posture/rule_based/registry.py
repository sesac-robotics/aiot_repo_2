"""운동별 자세 평가기 선택과 공통 Live 결과. 기존 Analyzer를 실행하지 않습니다."""
import math
from app.analysis.registry import get_exercise_definition
from app.analysis.posture_state import posture_state
from .base import ERRORS, TemporalGate, Check
from .config import RuleConfig
from .geometry import PoseGeometry
from .exercises import SquatPosture, PushUpPosture, YtwPosture, PlankPosture, PressPosture, unavailable

EVALUATORS = {'squat':SquatPosture, 'push_up':PushUpPosture, 'ytw_raise':YtwPosture,
              'plank':PlankPosture, 'dumbbell_shoulder_press':PressPosture}


class RuleBasedPredictor:
    """PostureService의 기존 prepare/push/reset 계약을 사용하는 별도 구현입니다."""
    def __init__(self, *, config=None, camera_view='unknown', camera_views=None):
        if camera_view not in ('unknown','front','side'):
            raise ValueError('POSTURE_CAMERA_VIEW는 unknown/front/side여야 합니다.')
        if camera_views is not None and not isinstance(camera_views,dict):
            raise ValueError('POSTURE_CAMERA_VIEWS는 운동 이름별 방향 객체여야 합니다.')
        for name,view in (camera_views or {}).items():
            get_exercise_definition(name)
            if view not in ('unknown','front','side'):
                raise ValueError('POSTURE_CAMERA_VIEWS 방향은 unknown/front/side여야 합니다.')
        self.config = config or RuleConfig()
        self.default_view = self.view = camera_view
        self.views = dict(camera_views or {})
        self.sequence = None  # 모델 전용 시퀀스를 생성하지 않습니다.
        self.reset()

    def reset(self):
        self.context = self.evaluator = None
        self.gates = {}
        self.last_timestamp = None
        self.normal_since = None
        self.normal_samples = 0

    def prepare(self, context):
        if context == self.context:
            return
        self.reset()
        if context is not None:
            get_exercise_definition(context.exercise_name)
            self.evaluator = EVALUATORS[context.exercise_name](self.config)
            self.view = self.views.get(context.exercise_name,self.default_view)
            self.context = context

    def push(self,context,keypoints,timestamp,frame_size,*,analysis_result=None):
        self.prepare(context)
        if context is None:
            return posture_state('inactive','no_active_exercise')
        if type(timestamp) not in (int,float) or not math.isfinite(timestamp):
            self.reset()
            return posture_state('insufficient_pose','invalid_timestamp')
        if self.last_timestamp is not None:
            if timestamp == self.last_timestamp:
                return None
            if timestamp < self.last_timestamp or timestamp-self.last_timestamp > self.config.max_frame_gap_sec:
                self.evaluator.reset()
                self.gates.clear()
                self.normal_since = None
                self.normal_samples = 0
                if timestamp < self.last_timestamp:
                    self.last_timestamp = timestamp
                    return posture_state('insufficient_pose','timestamp_reversed')
        self.last_timestamp = timestamp
        # 같은 프레임의 기존 판정만 읽고 오래된 phase/다른 운동 결과는 사용하지 않습니다.
        phase = getattr(analysis_result,'phase',None)
        if (getattr(analysis_result,'timestamp',None) != timestamp
                or getattr(analysis_result,'exercise_name',None) != context.exercise_name):
            phase = None
        try:
            pose = PoseGeometry(keypoints,frame_size,self.config)
            if not pose.points:
                raise ValueError('no_valid_body_side')
            if phase is None:
                self.evaluator.reset()
                checks = unavailable(self.evaluator.codes,'phase_unconfirmed')
            else:
                checks = self.evaluator.observe(pose,phase,timestamp,self.view)
        except (ValueError,TypeError,OverflowError,KeyError) as error:
            self.evaluator.reset()
            checks = unavailable(self.evaluator.codes,str(error))
        scope = getattr(self.evaluator,'applicable_codes',None)
        if scope is not None:
            applicable = scope(self.view)
            indexed = {c.code:c for c in checks}
            # 결측/단계 보류에도 방향 제외 항목이 unavailable로 되살아나지 않게 합니다.
            checks = [indexed.get(code,Check(code,None,reason='not_evaluated')) if code in applicable
                      else Check(code,None,reason='other_camera_view',state='not_applicable')
                      for code in self.evaluator.codes]
        excluded = {c.code:c.reason for c in checks if c.state == 'not_applicable'}
        applicable_checks = [c for c in checks if c.state != 'not_applicable']
        observed = {c.code:c for c in applicable_checks}
        # 단계가 바뀌어 적용 대상에서 빠진 오류는 다른 단계로 이월하지 않습니다.
        for code in set(self.gates)-set(observed):
            del self.gates[code]
        active = []
        for check in applicable_checks:
            gate = self.gates.setdefault(check.code,TemporalGate(self.config))
            if gate.update(check,timestamp):
                active.append(check.code)
        if 'PLANK_HIP_SAG' in active or 'PLANK_HIP_RAISED' in active:
            active = [c for c in active if c != 'PLANK_BODY_MISALIGNMENT']
        evaluated = [c.code for c in applicable_checks if c.bad is not None]
        missing = {c.code:c.reason for c in applicable_checks if c.bad is None}
        waiting = {c.code:c.reason for c in applicable_checks if c.bad is None and c.state == 'pending'}
        unavailable_checks = {c.code:c.reason for c in applicable_checks if c.bad is None and c.state != 'pending'}
        pending = list(waiting) + [c.code for c in applicable_checks if c.bad and c.code not in active]
        complete = bool(evaluated) and not missing
        normal = complete and not active and not pending and all(c.recovered for c in applicable_checks)
        if normal:
            if self.normal_since is None:
                self.normal_since,self.normal_samples = timestamp,1
            else:
                self.normal_samples += 1
        else:
            self.normal_since = None
            self.normal_samples = 0
        confirmed_normal = (normal and self.normal_samples >= 2
                            and timestamp-self.normal_since+1e-9 >= self.config.normal_confirm_sec)
        pose_missing = any('joint_' in str(reason) or reason in ('no_valid_body_side','invalid_frame_size')
                           for reason in unavailable_checks.values())
        # 확정 오류는 다른 항목의 결측/보류보다 우선하며 정상만 전체 적용 항목을 요구합니다.
        status = ('ready' if active else 'insufficient_pose' if pose_missing else
                  'warming_up' if pending or normal and not confirmed_normal else 'ready')
        reason = ('partial_error_assessment' if active and missing else None if active else
                  'insufficient_pose' if pose_missing else 'pending_assessment' if waiting else
                  'stabilizing_error' if pending else 'unavailable_assessment' if unavailable_checks else
                  'stabilizing_normal' if normal and not confirmed_normal else None)
        result = posture_state(status,reason)
        result.update(is_correct=False if active else True if confirmed_normal else None,
                      predicted_at=timestamp if evaluated else None,
                      classification=dict(method='rule_based',exercise=context.exercise_name,
                                          model_name='rule_based:'+context.exercise_name,status='ERROR' if active else 'NORMAL' if confirmed_normal else 'UNKNOWN',
                                          phase=phase,camera_view=self.view,thresholds_validated=False),
                      evaluated_checks=evaluated,unevaluated_checks=missing,pending_checks=pending,
                      not_applicable_checks=excluded,unavailable_checks=unavailable_checks,
                      check_statuses={c.code:dict(
                          status='not_applicable' if c.state=='not_applicable' else 'pending' if c.code in pending
                          else 'unavailable' if c.bad is None else 'error' if c.code in active else 'evaluated',
                          reason=c.reason) for c in checks},
                      measured_values={c.code:c.value for c in checks},
                      assessment_scope='complete_for_phase' if complete else 'partial',
                      last_cycle_evaluated_at=getattr(getattr(self.evaluator,'motion',None),'last_result_at',None))
        result['posture_errors'] = [dict(code=code,label=ERRORS[code][0],confidence=None,feedback=ERRORS[code][1])
                                     for code in active]
        feedback = ERRORS[active[0]][1] if active else '현재 촬영 방향에서 자세를 확인하고 있습니다.' if missing else '동작을 이어가세요.'
        result['display_feedback'] = dict(text=feedback,corrective=bool(active),
                                          error_type=active[0] if active else None,provisional=True,
                                          reason='rule_based_experimental')
        return result
