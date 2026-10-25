"""기존 서비스 import 경로를 유지하는 운동 식별자 호환 어댑터입니다.

규칙의 유일한 구현은 domain.exercise_names에 있습니다. 기존 호출자가 경로를
한 번에 바꾸지 않아도 되도록 명시적인 재내보내기만 제공합니다.
"""

from Server.AIFitService.domain.exercise_names import (
    canonical_exercise_name as canonical_exercise_name,
)
from Server.AIFitService.domain.exercise_names import (
    canonicalize_routine_exercises as canonicalize_routine_exercises,
)
