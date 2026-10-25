"""기존 실행 경로를 유지하는 런타임 설정 호환 모듈입니다."""

from Server.ExerciseEvaluation.runtime_environment import (
    correction_environment,
    ensure_correction_imports,
)

__all__ = ["correction_environment", "ensure_correction_imports"]
