"""원본 운동 교정과 자세 점수의 단일 카메라 통합 서버입니다."""

import argparse
import json
import signal
import threading
from pathlib import Path

from Server.ExerciseEvaluation.adapters.integrated_camera import (
    make_score_coordinator,
    original_recording_engine,
)
from Server.ExerciseEvaluation.adapters.session_repository import (
    JsonSessionRepository,
)
from Server.ExerciseEvaluation.application.integrated_session import (
    IntegratedSession,
)
from Server.ExerciseEvaluation.paths import CORRECTION, DATA_ROOT, REPO
from Server.ExerciseEvaluation.runtime_environment import (
    ensure_correction_imports,
)
from Server.ExerciseEvaluation.server import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument(
        "--correction-root",
        type=Path,
        default=CORRECTION,
    )
    parser.add_argument(
        "--score-python", default=str(REPO / ".venv-score/bin/python")
    )
    parser.add_argument("--replay-file", type=Path)
    args = parser.parse_args()
    ensure_correction_imports(
        args.correction_root, "GUI.FitHelper.run_integrated"
    )
    if args.replay_file and not args.replay_file.is_file():
        parser.error("재생 파일이 없습니다.")
    engine = original_recording_engine(
        args.correction_root, DATA_ROOT / "correction-sessions"
    )
    bridge = IntegratedSession(
        engine=engine,
        sessions_root=DATA_ROOT / "integrated-sessions",
        repository=JsonSessionRepository(DATA_ROOT / "integrated-sessions"),
        score_factory=lambda bridge: make_score_coordinator(
            bridge, args.score_python
        ),
        replay_source=str(args.replay_file) if args.replay_file else None,
    )
    server = make_server(args.port, bridge)
    base = server.RequestHandlerClass

    class Handler(base):
        def do_POST(self):
            if self.path not in ("/api/pause", "/api/resume"):
                return super().do_POST()
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if (
                host
                not in {
                    f"localhost:{server.server_port}",
                    f"127.0.0.1:{server.server_port}",
                }
                or origin
                and origin != f"http://{host}"
            ):
                self.json(403, {"error": "외부 요청을 거절합니다."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 8192 or length < 0:
                    raise ValueError("잘못된 요청 크기")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON 객체가 필요합니다.")
                (bridge.pause if self.path == "/api/pause" else bridge.resume)()
                self.json(200, bridge.snapshot())
            except (ValueError, RuntimeError) as exc:
                self.json(400, {"error": str(exc)})

        def do_GET(self):
            if self.path == "/api/integrated-sessions":
                try:
                    self.json(200, {"sessions": bridge.list_sessions()})
                except (OSError, ValueError):
                    self.json(500, {"error": "통합 기록을 읽지 못했습니다."})
            else:
                super().do_GET()

    server.RequestHandlerClass = Handler
    signal.signal(
        signal.SIGTERM,
        lambda *_: threading.Thread(
            target=server.shutdown, daemon=True
        ).start(),
    )
    print(
        f"http://127.0.0.1:{server.server_port} · 교정 원본 유지 · 단일 카메라 통합",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        bridge.stop()
        server.server_close()


if __name__ == "__main__":
    main()
