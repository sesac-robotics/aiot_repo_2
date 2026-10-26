"""지원 이름·유형·분석기 생성자의 단일 등록표. 인스턴스는 캐시하지 않습니다."""
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable

from app.analysis.exercise import ExerciseAnalyzer, ExerciseType
from app.analysis.exercises.squat import SquatAnalyzer
from app.analysis.exercises.shoulder_press import ShoulderPressAnalyzer
from app.analysis.exercises.push_up import PushUpAnalyzer
from app.analysis.exercises.ytw_raise import YtwRaiseAnalyzer
from app.analysis.exercises.plank import PlankAnalyzer
from app.session.models import AnalysisContext, InvalidSessionInput

@dataclass(frozen=True)
class ExerciseDefinition:
    exercise_name: str
    exercise_type: ExerciseType
    analyzer_factory: Callable[..., ExerciseAnalyzer]


# 이름은 여기에만 선언합니다. 목록 조회·입력 검증·분석기 선택이 같은 정보를 사용합니다.
_DEFINITIONS = tuple(ExerciseDefinition(*entry) for entry in (
    ('squat', 'repetition', SquatAnalyzer),
    ('dumbbell_shoulder_press', 'repetition', ShoulderPressAnalyzer),
    ('push_up', 'repetition', PushUpAnalyzer),
    ('ytw_raise', 'repetition', YtwRaiseAnalyzer),
    ('plank', 'hold', PlankAnalyzer),
))
_REGISTRY = MappingProxyType({entry.exercise_name: entry for entry in _DEFINITIONS})


def list_exercises() -> tuple[ExerciseDefinition, ...]:
    return _DEFINITIONS


def get_exercise_definition(exercise_name: str) -> ExerciseDefinition:
    if not isinstance(exercise_name, str) or exercise_name not in _REGISTRY:
        raise InvalidSessionInput(f'지원하지 않는 운동 이름: {exercise_name!r}')
    return _REGISTRY[exercise_name]


def get_analyzer(exercise_name: str, *, target_duration_sec: float | None = None) -> ExerciseAnalyzer:
    definition = get_exercise_definition(exercise_name)
    # 내부 Hold 설정만 준비합니다. 기존 REST/DB 계약에는 필드를 추가하지 않습니다.
    if target_duration_sec is not None:
        if (definition.exercise_type != 'hold' or type(target_duration_sec) not in (int, float)
                or not math.isfinite(target_duration_sec) or target_duration_sec <= 0):
            raise InvalidSessionInput('target_duration_sec는 Hold 운동의 양의 유한한 초 값이어야 합니다.')
    return definition.analyzer_factory(definition.exercise_name, definition.exercise_type,
                                      target_duration_sec=target_duration_sec)


class RegisteredExerciseAnalyzer:
    """기존 analyze/reset 인터페이스를 지키며 워커에서 현재 운동 분석기를 선택합니다.

    첫 유효 프레임에서 인스턴스를 생성합니다. REST 스레드에서 분석기를 실행하지 않습니다.
    Bridge의 기존 generation 검증과 별개로 직접 사용 시에도 문맥 전환을 초기화합니다.
    """
    def __init__(self, factory: Callable[[str], ExerciseAnalyzer] = get_analyzer):
        self._factory = factory
        self._context: AnalysisContext | None = None
        self._analyzer: ExerciseAnalyzer | None = None

    def reset(self):
        previous = self._analyzer
        self._analyzer = None
        self._context = None
        if previous is not None:
            previous.reset()

    def analyze(self, keypoints, context):
        if self._context != context:
            self.reset()
            analyzer = (self._factory(context.exercise_name,target_duration_sec=context.target_duration_sec)
                        if context.target_duration_sec is not None else self._factory(context.exercise_name))
            analyzer.reset()
            self._analyzer = analyzer
            self._context = context
        return self._analyzer.analyze(keypoints, context)
