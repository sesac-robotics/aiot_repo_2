"""측정 시작을 명시적으로 요청하는 로컬 통합 화면과 HTTP API를 제공합니다."""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .coordinator import Coordinator

WEB = Path(__file__).with_name("web")


def make_server(port, coordinator):
    """localhost에만 바인딩하며 다른 웹사이트의 측정 시작 요청을 거부합니다."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, mime="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def json(self, status, data):
            self.send(
                status,
                json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
            )

        def do_GET(self):
            if self.path == "/api/state":
                self.json(200, coordinator.snapshot())
            elif self.path == "/api/config":
                self.json(200, {"fixed_mode": coordinator.fixed_mode})
            elif self.path.startswith("/api/frame.jpg"):
                with coordinator.lock:
                    data = coordinator.jpeg
                (
                    self.send(200, data, "image/jpeg")
                    if data
                    else self.json(404, {"error": "카메라 대기 중입니다."})
                )
            elif self.path in ("/", "/app.js", "/style.css"):
                name = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/style.css": "style.css",
                }[self.path]
                mime = {
                    "/": "text/html; charset=utf-8",
                    "/app.js": "text/javascript; charset=utf-8",
                    "/style.css": "text/css; charset=utf-8",
                }[self.path]
                self.send(200, (WEB / name).read_bytes(), mime)
            else:
                self.json(404, {"error": "없는 경로입니다."})

        def do_POST(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if host not in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            } or (origin and origin not in {f"http://{host}"}):
                self.json(
                    403, {"error": "외부 웹사이트의 요청은 허용하지 않습니다."}
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    length < 0
                    or length > 8192
                    or self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    raise ValueError("작은 JSON 요청만 허용합니다.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON 객체가 필요합니다.")
                if self.path == "/api/start":
                    coordinator.start(payload)
                elif self.path == "/api/stop":
                    coordinator.stop()
                else:
                    self.json(404, {"error": "없는 경로입니다."})
                    return
                self.json(200, coordinator.snapshot())
            except (ValueError, TypeError) as error:
                self.json(400, {"error": str(error)})
            except Exception as error:
                self.json(500, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(fixed_mode=None):
    """화면만 먼저 열고 사용자가 시작하기 전에는 카메라·모델을 열지 않습니다."""
    parser = argparse.ArgumentParser(
        description="ESP32·웹캠 운동 교정·점수 실행"
    )
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument(
        "--correction-python",
        default=os.environ.get("CORRECTION_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--score-python", default=os.environ.get("SCORE_PYTHON", sys.executable)
    )
    args = parser.parse_args()
    coordinator = Coordinator(
        {"correction": args.correction_python, "score": args.score_python},
        fixed_mode=fixed_mode,
    )
    server = make_server(args.port, coordinator)
    print(
        f"http://127.0.0.1:{server.server_port} · 시작 버튼을 누르기 전에는 촬영하지 않습니다.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.stop()
        server.server_close()
