"""모델 구현에 의존하지 않는 운동 식별자와 반복·유지 유형입니다."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from app.session.models import InvalidSessionInput

ExerciseType = Literal["repetition", "hold"]


@dataclass(frozen=True)
class ExerciseSpec:
    """업무 입력 검증에 필요한 최소 운동 명세입니다. 분석기 객체는 보관하지 않습니다."""

    exercise_name: str
    exercise_type: ExerciseType


EXERCISE_SPECS = (
    ExerciseSpec("squat", "repetition"),
    ExerciseSpec("dumbbell_shoulder_press", "repetition"),
    ExerciseSpec("push_up", "repetition"),
    ExerciseSpec("ytw_raise", "repetition"),
    ExerciseSpec("plank", "hold"),
)
_BY_NAME = MappingProxyType(
    {spec.exercise_name: spec for spec in EXERCISE_SPECS}
)


def get_exercise_spec(name: str) -> ExerciseSpec:
    """등록된 명세를 반환합니다. 지원되지 않은 이름은 상태 변경 전에 거부합니다."""
    if not isinstance(name, str) or name not in _BY_NAME:
        raise InvalidSessionInput(f"지원하지 않는 운동 이름: {name!r}")
    return _BY_NAME[name]
