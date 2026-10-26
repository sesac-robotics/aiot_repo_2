"""저장된 다종목 보고서를 구성하고 읽기 전용 결과 화면을 여는 실행 도구입니다. 전체 영상을 새로 학습하는 진입점이 아닙니다.

VS Code ▶: show saved five-exercise model results without retraining."""

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    runtime = Path(os.environ.get("POSTURE_PYTHON", sys.executable))
    env = dict(
        os.environ,
        MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", "/tmp/posture-mpl"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if runtime.exists() and runtime.resolve() != Path(sys.executable).resolve():
        os.execve(
            str(runtime),
            [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    # Load a sibling entry point explicitly; do not change global import resolution.
    import importlib.util

    from suite_report import main as build_report

    spec = importlib.util.spec_from_file_location(
        "posture_saved_results_demo",
        BASE.parent / "shoulder-qevd" / "run_demo.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("저장 결과 데모 진입점을 읽을 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = module.main
    build_report()
    run(default_page="/all_training_results.html", model_results=True)


if __name__ == "__main__":
    main()
