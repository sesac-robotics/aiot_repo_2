"""기존 교정 서버 실행 명령을 유지하는 호환 진입점입니다.

실제 HTTP·관절 기록 어댑터는 서버 컴포넌트에서 구현합니다.
"""

from Server.ExerciseEvaluation.adapters.correction_server import (
    RecordingMixin,
    parser,
    serve,
)
from Server.ExerciseEvaluation.runtime_environment import (
    ensure_correction_imports,
)

__all__ = ["RecordingMixin", "parser", "serve"]

if __name__ == "__main__":
    args = parser.parse_args()
    ensure_correction_imports(
        args.correction_root,
        "Server.ExerciseEvaluation.adapters.correction_server",
    )
    serve(args)
