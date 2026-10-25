"""단일 프로세스에서 공유할 인메모리 운동 세션 관리자.

API/추론 워커가 같은 인스턴스를 주입받아야 합니다. 프로세스 간 공유나 영속성은
제공하지 않으며 서버 재시작 시 초기화됩니다. 시간 단위는 소수 초입니다.
active_duration_sec는 운동 사이 대기를 포함한 세션의 running 시간입니다.
Snapshot은 {'session': {...}, 'exercise': {...} 또는 None} 구조의 독립 복사본입니다.
"""
from copy import deepcopy
import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import threading
import time
import math
from typing import Callable

from app.session.models import (
    AnalysisContext, ExerciseNotFound, ExerciseState, InvalidSessionInput,
    SessionConflict, SessionNotFound, SessionState, HoldState,
)


class SessionManager:
    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic,
                 utc_now: Callable[[], datetime] | None = None):
        # 두 시계는 테스트에서 주입하며 경과 시간 계산에는 UTC 시계를 쓰지 않습니다.
        self._clock = monotonic
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._session = SessionState()
        self._exercise: ExerciseState | None = None
        self._started_mono: float | None = None
        self._ended_mono: float | None = None
        self._paused_since: float | None = None
        self._pause_total = 0.0
        self._rep_event_ids: set[str] = set()
        self._exercise_ids: set[int] = set()
        # 실제 상태 전이마다 증가시켜 Pause→Resume 사이의 오래된 결과도 거부합니다.
        self._analysis_revision = 0
        self._hold_state = None
        self._hold_timestamp = None
        # 자세 평가는 opt-in이며 OFF에서는 기존 Snapshot 형식을 유지합니다.
        self._posture_enabled = False
        self._posture_unavailable = None
        self._posture_result = None
        self._posture_context = None
        self._posture_updated_at = None

    @staticmethod
    def _positive_int(name, value):
        # bool은 int의 하위 타입이지만 DB ID나 목표 횟수로 허용하지 않습니다.
        if type(value) is not int or value <= 0:
            raise InvalidSessionInput(f'{name}은 양의 정수여야 합니다.')

    def _timestamp(self):
        value = self._utc_now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise InvalidSessionInput('utc_now는 시간대 정보가 있는 datetime을 반환해야 합니다.')
        return value.astimezone(timezone.utc).isoformat()

    def _require_active(self):
        if self._session.session_status == 'idle':
            raise SessionNotFound('시작한 세션이 없습니다.')
        if self._session.session_status == 'finished':
            raise SessionConflict('이미 종료한 세션입니다.')

    def _snapshot(self, now):
        # 잠금을 획득한 호출자만 사용합니다. 외부에는 내부 모델을 노출하지 않습니다.
        session = asdict(self._session)
        if self._started_mono is not None:
            end = self._ended_mono if self._ended_mono is not None else now
            paused = self._pause_total
            if self._paused_since is not None:
                paused += end - self._paused_since
            session['paused_duration_sec'] = paused
            session['active_duration_sec'] = end - self._started_mono - paused
        exercise=asdict(self._exercise) if self._exercise is not None else None
        if exercise is not None and self._hold_state is not None:
            exercise.update(asdict(self._hold_state))
        snapshot = {'session': session, 'exercise': exercise}
        if self._posture_enabled:
            # 전이 직후에도 과거 예측을 노출하지 않습니다. 모델/워커 종료를 기다리지 않습니다.
            from app.analysis.posture_state import posture_state
            context = self.get_analysis_context()
            if session['session_status'] != 'running' or context is None:
                status = 'paused' if session['session_status'] == 'paused' else 'inactive'
                snapshot['posture_analysis'] = posture_state(status, 'session_or_exercise_not_running')
            elif self._posture_unavailable is not None:
                snapshot['posture_analysis'] = posture_state('unavailable', self._posture_unavailable)
            elif self._posture_context != context or self._posture_result is None:
                snapshot['posture_analysis'] = posture_state('warming_up', 'new_analysis_context')
            elif now - self._posture_updated_at > 1.0:
                snapshot['posture_analysis'] = posture_state('insufficient_pose', 'stale_prediction')
            else:
                snapshot['posture_analysis'] = deepcopy(self._posture_result)
        return snapshot

    def start_session(self, session_id):
        self._positive_int('session_id', session_id)
        with self._lock:
            if self._session.session_status in ('running', 'paused'):
                raise SessionConflict('이미 활성 세션이 있습니다.')
            if self._session.session_id == session_id:
                raise SessionConflict('종료한 세션 ID는 재사용할 수 없습니다.')
            now, timestamp = self._clock(), self._timestamp()
            self._session = SessionState(session_id=session_id, session_status='running',
                                         started_at=timestamp)
            self._exercise = None
            self._hold_state,self._hold_timestamp = None,None
            self._started_mono, self._ended_mono = now, None
            self._paused_since, self._pause_total = None, 0.0
            self._rep_event_ids.clear()
            self._exercise_ids.clear()
            self._analysis_revision += 1
            return self._snapshot(now)

    def start_exercise(self, exercise_id, exercise_name, order_no, target_reps=None, *, target_duration_sec=None):
        self._positive_int('exercise_id', exercise_id)
        self._positive_int('order_no', order_no)
        if target_reps is not None:
            self._positive_int('target_reps', target_reps)
        if not isinstance(exercise_name, str) or not exercise_name.strip():
            raise InvalidSessionInput('exercise_name은 비어 있지 않은 문자열이어야 합니다.')
        # 분석 등록표와 같은 이름 검증을 사용합니다. 실패 시 상태는 변경하지 않습니다.
        from app.analysis.registry import get_exercise_definition
        exercise_name = get_exercise_definition(exercise_name.strip()).exercise_name
        if target_duration_sec is not None and (exercise_name!='plank' or type(target_duration_sec) not in (int,float)
                or not math.isfinite(target_duration_sec) or target_duration_sec<=0):
            raise InvalidSessionInput('target_duration_sec는 Hold 운동의 양의 유한한 초 값이어야 합니다.')
        with self._lock:
            self._require_active()
            if self._session.session_status != 'running':
                raise SessionConflict('일시정지 중에는 새 운동을 시작할 수 없습니다.')
            if self._exercise is not None and self._exercise.exercise_status == 'running':
                raise SessionConflict('현재 운동을 먼저 종료해야 합니다.')
            # 같은 세션에서 ID를 재사용하면 이전 운동의 지연 이벤트와 구별할 수 없습니다.
            if exercise_id in self._exercise_ids:
                raise SessionConflict('이 세션에서 이미 사용한 운동 ID입니다.')
            self._exercise = ExerciseState(exercise_id, exercise_name.strip(), order_no, target_reps)
            self._hold_state=HoldState(target_duration_sec=target_duration_sec) if exercise_name=='plank' else None
            self._hold_timestamp=None
            self._exercise_ids.add(exercise_id)
            self._rep_event_ids.clear()
            self._analysis_revision += 1
            return self._snapshot(self._clock())

    def pause_session(self):
        with self._lock:
            self._require_active()
            now = self._clock()
            if self._session.session_status == 'running':
                timestamp = self._timestamp()
                self._session.session_status = 'paused'
                self._session.paused_at = timestamp
                self._paused_since = now
                self._analysis_revision += 1
                if self._hold_state is not None and not self._hold_state.hold_completed:
                    # Pause는 연속 유지 구간을 끊습니다. Resume 후 새 안정화를 요구합니다.
                    self._hold_state.current_hold_sec=0.
                    self._hold_state.hold_status='waiting'
                    self._hold_timestamp=None
            # 중복 Pause에서는 최초 paused_at 및 시작 시각을 보존합니다.
            return self._snapshot(now)

    def resume_session(self):
        with self._lock:
            self._require_active()
            now = self._clock()
            if self._session.session_status == 'paused':
                self._pause_total += now - self._paused_since
                self._paused_since = None
                self._session.paused_at = None
                self._session.session_status = 'running'
                self._analysis_revision += 1
            # 이미 running인 Resume는 누적 시간을 변경하지 않습니다.
            return self._snapshot(now)

    def finish_exercise(self):
        with self._lock:
            self._require_active()
            if self._exercise is None:
                raise ExerciseNotFound('현재 운동이 없습니다.')
            if self._exercise.exercise_status == 'finished':
                raise SessionConflict('이미 종료한 운동입니다.')
            # Pause 중에도 명시적 종료는 허용합니다. 세션 상태는 그대로 유지합니다.
            self._exercise.exercise_status = 'finished'
            if self._hold_state is not None and not self._hold_state.hold_completed:
                self._hold_state.hold_status='interrupted'
            self._analysis_revision += 1
            return self._snapshot(self._clock())

    def finish_session(self):
        with self._lock:
            self._require_active()
            if self._exercise is not None and self._exercise.exercise_status == 'running':
                raise SessionConflict('세션 종료 전에 현재 운동을 종료해야 합니다.')
            now, timestamp = self._clock(), self._timestamp()
            if self._paused_since is not None:
                self._pause_total += now - self._paused_since
                self._paused_since = None
            self._ended_mono = now
            self._session.ended_at = timestamp
            self._session.paused_at = None
            self._session.session_status = 'finished'
            self._analysis_revision += 1
            return self._snapshot(now)

    def record_rep(self, session_id, exercise_id, event_id):
        """반복 완료 이벤트를 반영하면 True, 비활성/지연/중복이면 False.

        event_id는 분석기가 반복마다 부여하고 재전송 시 재사용하는 문자열입니다.
        DB record_id를 생성하지 않습니다. 중복 집합은 현재 운동에 한해 보관합니다.
        목표 도달에 따른 자동 종료는 하지 않습니다. Pause 중 거절한 이벤트는 소비하지
        않으므로 분석기는 Pause 시 미완료 반복을 폐기하고 오래된 이벤트를 재전송하지 않아야 합니다.
        """
        self._positive_int('session_id', session_id)
        self._positive_int('exercise_id', exercise_id)
        if not isinstance(event_id, str) or not event_id.strip():
            raise InvalidSessionInput('event_id는 비어 있지 않은 문자열이어야 합니다.')
        with self._lock:
            if (self._session.session_status != 'running'
                    or self._session.session_id != session_id
                    or self._exercise is None
                    or self._exercise.exercise_status != 'running'
                    or self._exercise.exercise_id != exercise_id
                    or event_id in self._rep_event_ids):
                return False
            self._rep_event_ids.add(event_id)
            self._exercise.rep_count += 1
            return True

    def enable_posture_analysis(self, *, unavailable_reason=None):
        """프로세스 초기화 시에만 켭니다. 모델을 SessionManager 안에서 로드하지 않습니다."""
        with self._lock:
            self._posture_enabled = True
            self._posture_unavailable = unavailable_reason

    def record_posture_analysis(self, context, result):
        """같은 세대의 JSON 결과만 반영하며 Reps/Hold에는 접근하지 않습니다."""
        # JSON 검증과 복사는 잠금 밖에서 수행합니다. NaN/Tensor는 저장하지 않습니다.
        json.dumps(result, allow_nan=False)
        result = deepcopy(result)
        with self._lock:
            if (not self._posture_enabled or context is None
                    or context != self.get_analysis_context()):
                return False
            self._posture_context = context
            self._posture_result = result
            self._posture_updated_at = self._clock()
            return True

    def get_snapshot(self):
        with self._lock:
            return self._snapshot(self._clock())

    def get_analysis_context(self) -> AnalysisContext | None:
        """짧은 잠금으로 분석 가능 문맥만 반환하며 추론/분석 중에는 잠금을 유지하지 않습니다."""
        with self._lock:
            if (self._session.session_status != 'running' or self._exercise is None
                    or self._exercise.exercise_status != 'running'
                    or self._hold_state is not None and self._hold_state.hold_completed):
                return None
            return AnalysisContext(self._session.session_id, self._exercise.exercise_id,
                                   self._exercise.exercise_name, self._analysis_revision,
                                   self._hold_state.target_duration_sec if self._hold_state else None)

    def record_analysis_hold(self,context,*,current_hold_sec,target_duration_sec,hold_status,timestamp,
                             completion_event_id=None):
        """유효 분석 시간만 저장합니다. 조회 시 시계로 외삽하거나 반복 횟수로 환산하지 않습니다."""
        if (type(current_hold_sec) not in (int,float) or not math.isfinite(current_hold_sec) or current_hold_sec<0
                or type(timestamp) not in (int,float) or not math.isfinite(timestamp)
                or target_duration_sec is not None and (type(target_duration_sec) not in (int,float)
                    or not math.isfinite(target_duration_sec) or target_duration_sec<=0)
                or hold_status not in ('waiting','stabilizing','holding','interrupted','completed')):
            raise InvalidSessionInput('Hold 분석 결과가 올바르지 않습니다.')
        with self._lock:
            if context is None or context!=self.get_analysis_context() or self._hold_state is None:
                return False
            if target_duration_sec!=self._hold_state.target_duration_sec:
                raise InvalidSessionInput('Hold 목표가 현재 운동과 다릅니다.')
            if self._hold_timestamp is not None and timestamp<=self._hold_timestamp:return False
            if hold_status=='completed' and (target_duration_sec is None or current_hold_sec<target_duration_sec
                    or not isinstance(completion_event_id,str) or not completion_event_id.strip()):
                raise InvalidSessionInput('Hold 완료 결과가 올바르지 않습니다.')
            if completion_event_id is not None and hold_status!='completed':
                raise InvalidSessionInput('완료 이벤트와 Hold 상태가 다릅니다.')
            self._hold_timestamp=timestamp
            self._hold_state.current_hold_sec=current_hold_sec
            self._hold_state.hold_status=hold_status
            self._hold_state.hold_completed=hold_status=='completed'
            return True

    def record_analysis_rep(self, context: AnalysisContext, event_id: str) -> bool:
        """세대·세션·운동·상태 재검증과 반복 반영을 하나의 잠금 안에서 수행합니다."""
        with self._lock:
            if context is None or context != self.get_analysis_context():
                return False
            if not isinstance(event_id, str) or not event_id.strip():
                raise InvalidSessionInput('event_id는 비어 있지 않은 문자열이어야 합니다.')
            # reset 후 이벤트 번호 재사용은 허용하되 같은 세대의 재전송은 중복 제거합니다.
            return self.record_rep(context.session_id, context.exercise_id,
                                   f'analysis:{context.revision}:{event_id}')


    @contextmanager
    def target_session(self, session_id, exercise_id=None):
        """ID 검증과 후속 메서드를 원자적으로 수행하는 동기식 호출 범위.

        API에서 이 범위 안에 I/O나 await를 넣지 않습니다. RLock으로 기존 메서드의
        재진입을 허용하여 검증 뒤 다른 요청이 세션/운동을 교체하는 경합을 방지합니다.
        종료된 현재 세션의 조회도 허용하며 상태 변경 허용 여부는 기존 메서드가 판단합니다.
        """
        self._positive_int('session_id', session_id)
        if exercise_id is not None:
            self._positive_int('exercise_id', exercise_id)
        with self._lock:
            if self._session.session_id != session_id:
                raise SessionNotFound('요청한 세션이 현재 관리자에 없습니다.')
            if exercise_id is not None and (
                    self._exercise is None or self._exercise.exercise_id != exercise_id):
                raise ExerciseNotFound('요청한 운동이 현재 세션의 운동이 아닙니다.')
            yield self
