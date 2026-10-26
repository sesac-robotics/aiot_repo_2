"""기존 GUI import를 유지하는 통합 실행 호환 모듈입니다.

실행 유스케이스는 Server/ExerciseEvaluation/application, 원본 교정·카메라는
adapters에 있습니다. 이 파일은 추론·저장·세션 카운터를 구현하지 않습니다.
"""

from pathlib import Path

from Server.ExerciseEvaluation.adapters.integrated_camera import (
    PausingReader,
    SharedScoreCamera,
    WebcamVgaCapture,
    make_score_coordinator,
    original_recording_engine,
)
from Server.ExerciseEvaluation.adapters.session_repository import (
    JsonSessionRepository,
)
from Server.ExerciseEvaluation.application.integrated_session import (
    IntegratedSession as IntegratedApplication,
)


class IntegratedSession(IntegratedApplication):
    """이전 생성자에 JSON 저장소 기본값을 제공하는 호환 facade입니다."""

    def __init__(
        self,
        engine,
        score_python,
        sessions_root,
        replay_source=None,
        repository=None,
    ):
        super().__init__(
            engine=engine,
            sessions_root=Path(sessions_root),
            repository=(
                repository
                if repository is not None
                else JsonSessionRepository(Path(sessions_root))
            ),
            score_factory=lambda bridge: make_score_coordinator(
                bridge, score_python
            ),
            replay_source=replay_source,
        )


__all__ = [
    "IntegratedSession",
    "PausingReader",
    "SharedScoreCamera",
    "WebcamVgaCapture",
    "original_recording_engine",
]
