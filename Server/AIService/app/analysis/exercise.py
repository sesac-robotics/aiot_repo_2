"""추론 워커용 공통 결과 및 분석 연결부. 점수/피드백은 별도 모델의 책임입니다.

ExerciseAnalyzer는 Registry에서 선택하거나 PoseStreamService(exercise_analyzer=...)로 주입합니다.
analyze/reset은 단일 추론 워커에서만 호출하며 분석기에서 DB나 SessionManager를
직접 변경하지 않습니다. 반복 이벤트 ID는 현재 분석 세대 안에서 고유해야 합니다.
"""
from copy import deepcopy
from dataclasses import dataclass
import logging
import json
import math
import os
import time
from typing import Literal, Protocol

from app.session.manager import SessionManager
from app.session.models import AnalysisContext
from app.analysis.shoulder_debug import ShoulderPressDebug, context_fields, session_fields
from app.analysis.frame_debug import FrameDebug


@dataclass(frozen=True)
class RepCompleted:
    event_id: str


@dataclass(frozen=True)
class HoldCompleted:
    """반복 이벤트와 분리한 목표 유지 시간 완료 통지."""
    event_id: str


class KeypointFrame(list):
    """analyze(list, context) 호환 내부 입력. 정규화 좌표의 영상 비율/표본 시각을 전달합니다.

    timestamp는 추론 워커가 프레임을 가져온 직후 기록한 monotonic 초입니다. REST keypoints의
    필드를 변경하지 않으며 촬영 시각 또는 카메라/디코더 내부 지연을 보장하지 않습니다.
    """
    def __init__(self, keypoints, *, frame_size, timestamp, frame_id=None, frame_trace=None):
        super().__init__(keypoints)
        self.frame_size = frame_size  # (width, height)
        self.timestamp = timestamp
        self.frame_id = frame_id  # 내부 출처 정보이며 기존 API Keypoint 스키마에는 추가하지 않습니다.
        self.frame_trace = frame_trace


ExerciseType = Literal['repetition', 'hold']


@dataclass(frozen=True)
class AnalysisResult:
    """Rule-based 결과만 표현합니다. timestamp는 monotonic 초이며 DB 시각이 아닙니다.

    None인 유지 시간은 아직 계산하지 않았다는 뜻입니다. target_duration_sec는 내부
    분석기 설정이며 이번 작업에서는 REST/DB 필드로 추가하지 않습니다.
    Score/Posture Error는 별도 학습 모델의 책임입니다.
    """
    exercise_name: str
    exercise_type: ExerciseType
    phase: str
    timestamp: float
    repetition_event: RepCompleted | None = None
    target_duration_sec: float | None = None
    hold_duration_sec: float | None = None
    hold_completion_event: HoldCompleted | None = None


class ExerciseAnalyzer(Protocol):
    def reset(self) -> None:
        """진행 중이던 동작 단계를 폐기합니다."""
        ...

    def analyze(self, keypoints: list[dict], context: AnalysisContext) -> AnalysisResult | RepCompleted | None:
        """한 프레임의 공통 결과를 반환합니다. 기존 RepCompleted/None 반환도 지원합니다."""
        ...


class NoOpExerciseAnalyzer:
    """실제 분석기를 연결하기 전의 기본 구현: 반복 이벤트를 생성하지 않습니다."""
    def reset(self):
        pass

    def analyze(self, keypoints, context):
        return None


class PlaceholderExerciseAnalyzer(NoOpExerciseAnalyzer):
    """운동별 뼈대의 공통 No-op. 시간/반복 수를 추정하거나 누적하지 않습니다."""
    def __init__(self, exercise_name: str, exercise_type: ExerciseType,
                 target_duration_sec: float | None = None):
        self.exercise_name = exercise_name
        self.exercise_type = exercise_type
        self.target_duration_sec = target_duration_sec

    def analyze(self, keypoints, context):
        # 향후 holding/invalid 및 운동별 phase를 각 운동 구현에서 반환합니다.
        return AnalysisResult(self.exercise_name, self.exercise_type, 'not_implemented',
                              time.monotonic(), target_duration_sec=self.target_duration_sec)


class ExerciseAnalysisBridge:
    def __init__(self, manager: SessionManager, analyzer: ExerciseAnalyzer):
        self.manager = manager
        self.analyzer = analyzer
        self._context = None
        self._shoulder_debug = ShoulderPressDebug()
        # 판정/수락 경로를 바꾸지 않고 푸시업의 호출·이벤트 반영 경계만 관측합니다.
        self._pushup_debug = os.getenv('PUSHUP_DEBUG', os.getenv('PUSH_UP_DEBUG', '0')) == '1'
        self._pushup_debug_last = -math.inf
        self._pushup_debug_change = None
        self._ytw_debug = os.getenv('YTW_DEBUG') == '1'
        self._ytw_debug_last = -math.inf
        self._ytw_debug_change = None
        self._frame_debug = FrameDebug()
        self._frame_id = None

    def reset(self):
        # 실패한 분석기 상태로 다음 프레임을 처리하지 않도록 문맥부터 무효화합니다.
        self._context = None
        self.analyzer.reset()

    def _debug_report(self, context, reason, *, frame_timestamp, analyzer_called=False,
                      event=None, manager_called=False, accepted=None, detail=None):
        if self._ytw_debug:
            self._report_ytw(context,reason,frame_timestamp=frame_timestamp,analyzer_called=analyzer_called,
                             event=event,manager_called=manager_called,accepted=accepted,detail=detail)
        self._shoulder_debug.emit('bridge', lambda: {
            **context_fields(context), 'frame_timestamp': frame_timestamp,
            'analyzer_called': analyzer_called, 'selected_wrapper': type(self.analyzer).__name__,
            'selected_analyzer': type(getattr(self.analyzer, '_analyzer', self.analyzer)).__name__,
            'reason': reason, 'detail': detail,
            'rep_completed': event is not None, 'event_id': event.event_id if event else None,
            'manager_called': manager_called, 'event_accepted': accepted,
            'manager': session_fields(self.manager),
            'current_context': context_fields(self.manager.get_analysis_context()),
        }, force=event is not None,
           change=(reason, getattr(context, 'session_id', None), getattr(context, 'exercise_id', None)))
        if self._pushup_debug:
            try:
                snapshot = self.manager.get_snapshot()
                exercise = snapshot['exercise']
                if (getattr(context, 'exercise_name', None) != 'push_up'
                        and (exercise is None or exercise['exercise_name'] != 'push_up')):
                    return
                now = time.monotonic()
                # 대기 사유 변화만으로 매 프레임 출력하지 않습니다. 이벤트/예외는 즉시 출력합니다.
                signature = (reason, getattr(context, 'session_id', None), getattr(context, 'exercise_id', None))
                urgent = event is not None or (reason == 'analysis_exception'
                                               and signature != self._pushup_debug_change)
                if urgent or now-self._pushup_debug_last >= 1.:
                    logging.getLogger('uvicorn.error').info('[PushUpDebug] %s', json.dumps({
                        'kind': 'bridge', **context_fields(context), 'frame_timestamp': frame_timestamp,
                        'analyzer_called': analyzer_called,
                        'selected_analyzer': type(getattr(self.analyzer, '_analyzer', self.analyzer)).__name__,
                        'reason': reason, 'detail': detail, 'rep_completed': event is not None,
                        'event_id': event.event_id if event else None,
                        'manager_called': manager_called, 'event_accepted': accepted,
                        'session_status': snapshot['session']['session_status'], 'exercise': exercise,
                        'current_context': context_fields(self.manager.get_analysis_context()),
                    }, ensure_ascii=False, default=str))
                    self._pushup_debug_last, self._pushup_debug_change = now, signature
            except Exception:
                pass  # 진단 오류가 기존 결과 반영을 방해하지 않도록 합니다.

    def _report_ytw(self, context, reason, *, frame_timestamp, analyzer_called,
                    event, manager_called, accepted, detail):
        # 기존 반영 결과만 관측합니다. 관리자 잠금/중복 검사/상태 전이는 변경하지 않습니다.
        try:
            snapshot = self.manager.get_snapshot()
            session,exercise = snapshot['session'],snapshot['exercise']
            if (getattr(context,'exercise_name',None) != 'ytw_raise'
                    and (exercise is None or exercise['exercise_name'] != 'ytw_raise')):
                return
            change = (reason,getattr(context,'session_id',None),getattr(context,'exercise_id',None))
            now = time.monotonic()
            if event is None and change==self._ytw_debug_change and now-self._ytw_debug_last<1.:
                return
            rejected = ('manager_returned_false' if manager_called and accepted is False else reason)
            logging.getLogger('uvicorn.error').info('[YTWDebug] %s',json.dumps({
                'kind':'bridge','analyzer_called':analyzer_called,
                'session_id':getattr(context,'session_id',session['session_id']),
                'exercise_id':getattr(context,'exercise_id',exercise['exercise_id'] if exercise else None),
                'session_status':session['session_status'],
                'exercise_name':getattr(context,'exercise_name',exercise['exercise_name'] if exercise else None),
                'frame_timestamp':frame_timestamp,'reason':reason,'detail':detail,
                'frame_id':self._frame_id,
                'event_id':event.event_id if event else None,'rep_completed':event is not None,
                'manager_called':manager_called,'event_accepted':accepted,
                'event_rejected_reason':rejected if event is not None and accepted is not True else None,
                'analysis_context':context_fields(context),
                'current_context':context_fields(self.manager.get_analysis_context()),
            },ensure_ascii=False,default=str))
            self._ytw_debug_last,self._ytw_debug_change=now,change
        except Exception:
            pass  # 로그 실패로 이벤트 수락 결과를 바꾸지 않습니다.

    def process(self, context: AnalysisContext | None, keypoints: list[dict], *,
                frame_size=None, timestamp=None, frame_id=None, frame_trace=None) -> bool:
        analyzer_called = False
        event_for_debug = None
        frame_timestamp = timestamp if timestamp is not None else getattr(keypoints, 'timestamp', None)
        self._frame_id = frame_id if frame_id is not None else getattr(keypoints,'frame_id',None)
        try:
            current = self.manager.get_analysis_context()
            if context is None or context != current:
                self._debug_report(context, 'no_running_exercise' if context is None else 'stale_context_before_analysis',
                                   frame_timestamp=frame_timestamp)
                if self._context is not None:
                    self.reset()
                return False
            if self._context != context:
                self.reset()
                self._context = context
            # 사람을 놓친 동안 부분 동작이 이어지지 않도록 초기화합니다.
            if not keypoints:
                self._debug_report(context, 'no_keypoints', frame_timestamp=frame_timestamp)
                self.reset()
                return False
            if self.manager.get_analysis_context() != context:
                self._debug_report(context, 'stale_context_before_analysis', frame_timestamp=frame_timestamp)
                self.reset()
                return False
            # 분석기에 API 반환용 keypoints의 수정 권한을 주지 않습니다.
            analysis_points = deepcopy(keypoints)
            if frame_size is not None and timestamp is not None:
                analysis_points = KeypointFrame(analysis_points, frame_size=frame_size,
                                               timestamp=timestamp,frame_id=self._frame_id,frame_trace=frame_trace)
            self._frame_debug.analyzer_input(analysis_points,getattr(analysis_points,'frame_trace',None),
                                             frame_id=getattr(analysis_points,'frame_id',None),timestamp=frame_timestamp)
            analyzer_called = True
            result = self.analyzer.analyze(analysis_points, context)
            event_for_debug = (result.repetition_event if isinstance(result, AnalysisResult)
                               else result if isinstance(result, RepCompleted) else None)
            if self.manager.get_analysis_context() != context:
                self._debug_report(context, 'stale_context_after_analysis', frame_timestamp=frame_timestamp,
                                   analyzer_called=True, event=event_for_debug)
                self.reset()
                return False
            if result is None:
                self._debug_report(context, 'no_analysis_result', frame_timestamp=frame_timestamp, analyzer_called=True)
                return False
            # 등록표의 유형을 기준으로 Hold 결과가 반복 횟수로 반영되는 것을 막습니다.
            from app.analysis.registry import get_exercise_definition
            definition = get_exercise_definition(context.exercise_name)
            if isinstance(result, AnalysisResult):
                if (result.exercise_name != context.exercise_name
                        or result.exercise_type != definition.exercise_type):
                    raise ValueError('현재 운동과 분석 결과의 메타데이터가 다릅니다.')
                event = result.repetition_event
            elif isinstance(result, RepCompleted):
                event = result  # 기존 분석기/테스트 대역과 호환
            else:
                raise TypeError('지원하지 않는 분석 결과 타입입니다.')
            if (definition.exercise_type=='hold' and isinstance(result,AnalysisResult)
                    and event is None and result.hold_duration_sec is not None):
                # 세대 재검증/저장은 관리자 안에서 원자적으로 수행하며 rep_count는 변경하지 않습니다.
                return self.manager.record_analysis_hold(context,current_hold_sec=result.hold_duration_sec,
                    target_duration_sec=result.target_duration_sec,hold_status=result.phase,timestamp=result.timestamp,
                    completion_event_id=result.hold_completion_event.event_id if result.hold_completion_event else None)
            if event is None:
                self._debug_report(context, 'no_rep_event', frame_timestamp=frame_timestamp, analyzer_called=True)
                return False
            if definition.exercise_type != 'repetition':
                raise ValueError('Hold 운동은 반복 완료 이벤트를 생성할 수 없습니다.')
            # 이 검사 이후의 상태 전환도 관리자의 원자적 반영 단계에서 다시 거부합니다.
            accepted = self.manager.record_analysis_rep(context, event.event_id)
            self._debug_report(context, 'manager_accepted' if accepted else 'manager_rejected',
                               frame_timestamp=frame_timestamp, analyzer_called=True, event=event,
                               manager_called=True, accepted=accepted)
            return accepted
        except Exception as error:
            self._debug_report(context, 'analysis_exception', frame_timestamp=frame_timestamp,
                               analyzer_called=analyzer_called, event=event_for_debug, detail=str(error))
            logging.getLogger('uvicorn.error').exception('운동 분석 오류: 현재 프레임 결과 폐기')
            try:
                self.reset()
            except Exception:
                logging.getLogger('uvicorn.error').exception('운동 분석기 초기화 실패')
            return False
