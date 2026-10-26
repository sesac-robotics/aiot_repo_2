"""별도 런타임에서 팀 교정 서버를 준비하고 직접 교정 GUI를 실행합니다."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from GUI.FitHelper.runtime_config import correction_environment
from Server.ExerciseEvaluation.paths import DATA_ROOT, REPO


def main():
    repo = REPO
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correction-python",
        type=Path,
        default=repo / ".venv-correction/bin/python",
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=repo / "Server/PostureAnalysis/posture-error-training",
    )
    parser.add_argument("--port", type=int, default=8782)
    parser.add_argument(
        "--sessions",
        type=Path,
        default=DATA_ROOT / "correction-sessions",
    )
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))

    def ready():
        try:
            with opener.open(url + "/api/config", timeout=1) as response:
                config = json.load(response)
                if not config.get("full_pose_recording"):
                    raise RuntimeError(
                        "이 포트는 기존 서버가 사용 중입니다. 다른 포트를 지정하십시오."
                    )
                return True
        except URLError:
            return False

    engine = None
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    try:
        if not ready():
            if (
                not args.correction_python.is_file()
                or not args.correction_root.is_dir()
            ):
                parser.error("교정 런타임과 교정 소스 경로를 확인하십시오.")
            engine_env = correction_environment(args.correction_root)
            engine_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            engine = subprocess.Popen(
                [
                    str(args.correction_python.absolute()),
                    "-u",
                    "-m",
                    "GUI.FitHelper.direct_correction_server",
                    "--port",
                    str(args.port),
                    "--correction-root",
                    str(args.correction_root.resolve()),
                    "--sessions",
                    str(args.sessions.resolve()),
                ],
                cwd=repo,
                env=engine_env,
            )
            for _ in range(80):
                if engine.poll() is not None:
                    raise RuntimeError("교정 서버가 종료되었습니다.")
                if ready():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("교정 서버 시작 시간이 초과되었습니다.")
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "GUI.FitHelper.direct_correction_window",
                "--server",
                url,
            ],
            env=env,
        )
    finally:
        if engine is not None:
            engine.terminate()
            engine.wait(timeout=15)


if __name__ == "__main__":
    sys.exit(main())
