"""추론 워커용 공통 결과 및 분석 연결부. 점수/피드백은 별도 모델의 책임입니다.

ExerciseAnalyzer는 Registry에서 선택하거나 PoseStreamService(exercise_analyzer=...)로 주입합니다.
analyze/reset은 단일 추론 워커에서만 호출하며 분석기에서 DB나 SessionManager를
직접 변경하지 않습니다. 반복 이벤트 ID는 현재 분석 세대 안에서 고유해야 합니다.
"""
from copy import deepcopy
from dataclasses import dataclass
import logging
import time
from typing import Literal, Protocol

from app.session.manager import SessionManager
from app.session.models import AnalysisContext


@dataclass(frozen=True)
class RepCompleted:
    event_id: str


class KeypointFrame(list):
    """analyze(list, context) 호환 내부 입력. 정규화 좌표의 영상 비율/표본 시각을 전달합니다.

    timestamp는 추론 워커가 프레임을 가져온 직후 기록한 monotonic 초입니다. REST keypoints의
    필드를 변경하지 않으며 촬영 시각 또는 카메라/디코더 내부 지연을 보장하지 않습니다.
    """
    def __init__(self, keypoints, *, frame_size, timestamp):
        super().__init__(keypoints)
        self.frame_size = frame_size  # (width, height)
        self.timestamp = timestamp


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

    def reset(self):
        # 실패한 분석기 상태로 다음 프레임을 처리하지 않도록 문맥부터 무효화합니다.
        self._context = None
        self.analyzer.reset()

    def process(self, context: AnalysisContext | None, keypoints: list[dict], *,
                frame_size=None, timestamp=None) -> bool:
        try:
            current = self.manager.get_analysis_context()
            if context is None or context != current:
                if self._context is not None:
                    self.reset()
                return False
            if self._context != context:
                self.reset()
                self._context = context
            # 사람을 놓친 동안 부분 동작이 이어지지 않도록 초기화합니다.
            if not keypoints:
                self.reset()
                return False
            if self.manager.get_analysis_context() != context:
                self.reset()
                return False
            # 분석기에 API 반환용 keypoints의 수정 권한을 주지 않습니다.
            analysis_points = deepcopy(keypoints)
            if frame_size is not None and timestamp is not None:
                analysis_points = KeypointFrame(analysis_points, frame_size=frame_size,
                                               timestamp=timestamp)
            result = self.analyzer.analyze(analysis_points, context)
            if self.manager.get_analysis_context() != context:
                self.reset()
                return False
            if result is None:
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
            if event is None:
                return False
            if definition.exercise_type != 'repetition':
                raise ValueError('Hold 운동은 반복 완료 이벤트를 생성할 수 없습니다.')
            # 이 검사 이후의 상태 전환도 관리자의 원자적 반영 단계에서 다시 거부합니다.
            return self.manager.record_analysis_rep(context, event.event_id)
        except Exception:
            logging.getLogger('uvicorn.error').exception('운동 분석 오류: 현재 프레임 결과 폐기')
            try:
                self.reset()
            except Exception:
                logging.getLogger('uvicorn.error').exception('운동 분석기 초기화 실패')
            return False
