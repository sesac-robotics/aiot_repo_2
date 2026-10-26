"""나머지 네 종목의 추출·분할 감사·학습·보고 순서를 조정합니다. 명시적으로 실행하면 학습 산출물이 변경되므로 동결 보존본과 분리해야 합니다.

VS Code ▶: Train the remaining four exercises; resume existing pose caches."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exercise",
        choices=["pushups", "elbow_plank", "ytw", "shoulder_press"],
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--skip-prepare", action="store_true")
    a = p.parse_args()
    executable = Path(os.environ.get("POSTURE_PYTHON", sys.executable))
    python = str(executable) if executable.exists() else sys.executable
    env = dict(
        os.environ,
        MPLCONFIGDIR="/tmp/posture-mpl",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="3",
    )
    commands = []
    if not a.skip_prepare:
        commands.append(
            ["suite_prepare.py", "--workers", str(a.workers)]
            + (["--only", a.exercise] if a.exercise else [])
        )
    commands.append(["suite_data_audit.py"])
    commands.append(
        ["suite_train.py"] + (["--exercise", a.exercise] if a.exercise else [])
    )
    if not a.exercise:
        commands.append(["test_suite.py"])
    commands += [["suite_report.py"]]
    for command in commands:
        print("Running " + " ".join(command), flush=True)
        subprocess.run(
            [python, str(BASE / command[0]), *command[1:]],
            cwd=BASE,
            env=env,
            check=True,
        )
    print("Completed. Results: " + str(BASE / "multi/REPORT.md"), flush=True)


if __name__ == "__main__":
    main()
