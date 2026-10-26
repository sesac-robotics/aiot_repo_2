"""원본 로그인 GUI와 서로 독립된 교정·점수 서버를 준비합니다."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from GUI.FitHelper.runtime_config import correction_environment
from Server.ExerciseEvaluation.paths import CORRECTION, DATA_ROOT, REPO


def main():
    repo = REPO
    parser = argparse.ArgumentParser(
        description="기본 GUI · 운동 교정 / 자세 점수 선택"
    )
    parser.add_argument(
        "--original-gui", type=Path, default=repo / "GUI/FitHelper"
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=CORRECTION,
    )
    parser.add_argument(
        "--correction-python",
        type=Path,
        default=repo / ".venv-correction/bin/python",
    )
    parser.add_argument("--auth-port", type=int, default=8000)
    parser.add_argument("--score-port", type=int, default=8790)
    parser.add_argument("--correction-port", type=int, default=8782)
    parser.add_argument("--integrated-port", type=int, default=8793)
    args = parser.parse_args()
    auth_url = f"http://127.0.0.1:{args.auth_port}"
    score_url = f"http://127.0.0.1:{args.score_port}"
    correction_url = f"http://127.0.0.1:{args.correction_port}"
    integrated_url = f"http://127.0.0.1:{args.integrated_port}"
    os.environ["AIFIT_SERVER_URL"] = auth_url
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    opener = build_opener(ProxyHandler({}))
    children = []
    owned_urls = []

    def fetch(url):
        try:
            with opener.open(url, timeout=1) as response:
                return json.load(response)
        except URLError:
            return None

    def ensure(url, command, predicate, child_env=None):
        data = fetch(url)
        if data is not None:
            if not predicate(data):
                raise RuntimeError("기존 서버의 모드가 다릅니다: " + url)
            return
        child = subprocess.Popen(command, cwd=repo, env=child_env or env)
        children.append(child)
        if not url.endswith("/health"):
            owned_urls.append(url.removesuffix("/api/config"))
        for _ in range(80):
            if child.poll() is not None:
                raise RuntimeError("필요한 서버가 종료되었습니다: " + url)
            data = fetch(url)
            if data is not None and predicate(data):
                return
            time.sleep(0.1)
        raise RuntimeError("서버 시작 시간이 초과되었습니다: " + url)

    try:
        if not (args.original_gui / "mainLoginWindow.ui").is_file():
            parser.error("팀 원본 GUI 경로를 확인하십시오.")
        ensure(
            auth_url + "/health",
            [
                sys.executable,
                "-m",
                "GUI.FitHelper.run_local_auth",
                "--port",
                str(args.auth_port),
            ],
            lambda d: d.get("success") is True,
        )
        ensure(
            correction_url + "/api/config",
            [
                str(args.correction_python.absolute()),
                "-m",
                "GUI.FitHelper.direct_correction_server",
                "--correction-root",
                str(args.correction_root.resolve()),
                "--sessions",
                str(DATA_ROOT / "correction-sessions"),
                "--port",
                str(args.correction_port),
            ],
            lambda d: d.get("full_pose_recording") is True,
            correction_environment(args.correction_root),
        )
        ensure(
            score_url + "/api/config",
            [
                sys.executable,
                "-m",
                "Server.ExerciseEvaluation.run_score",
                "--port",
                str(args.score_port),
                "--score-python",
                sys.executable,
            ],
            lambda d: d.get("fixed_mode") == "score",
        )
        ensure(
            integrated_url + "/api/config",
            [
                str(args.correction_python.absolute()),
                "-m",
                "GUI.FitHelper.run_integrated",
                "--correction-root",
                str(args.correction_root),
                "--score-python",
                sys.executable,
                "--port",
                str(args.integrated_port),
            ],
            lambda d: d.get("fixed_mode") == "combined",
            correction_environment(args.correction_root),
        )
        from PyQt6.QtWidgets import QApplication

        from GUI.FitHelper.mode_gui import create_login

        app = QApplication(sys.argv[:1])
        window = create_login(
            args.original_gui, score_url, correction_url, integrated_url
        )
        window.show()
        return app.exec()
    finally:
        for base in reversed(owned_urls):
            try:
                with opener.open(
                    Request(
                        base + "/api/stop",
                        b"{}",
                        {"Content-Type": "application/json"},
                    ),
                    timeout=20,
                ):
                    pass
            except (OSError, URLError):
                pass
        for child in reversed(children):
            child.terminate()
            child.wait(timeout=20)


if __name__ == "__main__":
    sys.exit(main())
