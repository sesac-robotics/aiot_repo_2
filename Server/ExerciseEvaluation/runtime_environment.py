"""새 실행기의 저장 경로와 레거시 교정 프로세스 환경을 구성합니다.

팀원 원본 모듈의 최상위 import는 해당 교정 프로세스의 PYTHONPATH로만
공급합니다. GUI 프로세스의 import 경로와 원본 파일은 변경하지 않습니다.
"""

import os
import sys
from pathlib import Path

from Server.ExerciseEvaluation.paths import CORRECTION, REPO


def correction_environment(source_root: Path = CORRECTION) -> dict[str, str]:
    """저장소와 선택한 원본 교정 경로만 자식 프로세스에 전달합니다."""
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(REPO), str(Path(source_root).absolute())]
        ),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    return env


def ensure_correction_imports(source_root: Path, module: str) -> None:
    """직접 실행 시에도 원본 교정 import 환경을 갖춘 동일 모듈로 재실행합니다."""
    root = Path(source_root).absolute()
    if str(root) not in sys.path:
        # Do not resolve the venv interpreter symlink: it would bypass the venv.
        os.chdir(REPO)
        os.execve(
            sys.executable,
            [sys.executable, "-m", module, *sys.argv[1:]],
            correction_environment(root),
        )
