"""HTTP/DB 의존성이 없는 세션·운동 상태 모델과 도메인 예외."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AnalysisContext:
    """추론 시작 시 발급하는 불변 문맥. REST Snapshot에는 노출하지 않습니다."""
    session_id: int
    exercise_id: int
    exercise_name: str
    revision: int
    target_duration_sec: float | None = None


class SessionError(Exception):
    """API 계층에서 변환할 도메인 오류의 공통 타입."""


class InvalidSessionInput(SessionError, ValueError):
    """잘못된 인자: HTTP 400으로 매핑 가능."""


class SessionNotFound(SessionError):
    """시작한 세션 없음: HTTP 404로 매핑 가능."""


class ExerciseNotFound(SessionError):
    """현재 운동 없음: HTTP 404로 매핑 가능."""


class SessionConflict(SessionError):
    """중복 시작 또는 허용되지 않은 상태 전이: HTTP 409로 매핑 가능."""


@dataclass
class SessionState:
    session_id: int | None = None
    session_status: Literal['idle', 'running', 'paused', 'finished'] = 'idle'
    started_at: str | None = None
    ended_at: str | None = None
    paused_at: str | None = None
    paused_duration_sec: float = 0.0
    active_duration_sec: float = 0.0


@dataclass
class ExerciseState:
    exercise_id: int
    exercise_name: str
    order_no: int
    target_reps: int | None = None
    rep_count: int = 0
    # Pause는 세션 상태로 표현하며 현재 운동은 종료 전까지 running입니다.
    exercise_status: Literal['running', 'finished'] = 'running'


@dataclass
class HoldState:
    """플랭크에만 붙이는 추가 Snapshot. 세션 활동 시간과 분리합니다."""
    current_hold_sec: float = 0.
    target_duration_sec: float | None = None
    hold_status: Literal['waiting','stabilizing','holding','interrupted','completed'] = 'waiting'
    hold_completed: bool = False
