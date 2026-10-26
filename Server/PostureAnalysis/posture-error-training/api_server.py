"""저장된 관절 시퀀스를 받는 연구용 HTTP 추론 진입점입니다. 실제 카메라 연속성이나 운영 세션 정확도를 대신 검증하지 않습니다.

Optional localhost AI API: POST /predict with landmarks[T][33][4], aspect.
Use an explicit source of MediaPipe Pose image landmarks, not world landmarks.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from inference import PostureErrorClassifier
from suite_config import SUITE, TASKS
from suite_inference import FiveExerciseClassifier


def make_server(port=8770):
    classifier = PostureErrorClassifier()
    dispatcher = FiveExerciseClassifier()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.reply(
                    200,
                    dict(
                        status="ok",
                        exercise="squats",
                        targets=classifier.targets,
                        scope="experimental; specify exercise in POST /predict",
                        models_ready={
                            "squats": True,
                            **{
                                k: (SUITE / k / "outputs/best.pt").exists()
                                for k in TASKS
                            },
                        },
                    ),
                )
            else:
                self.reply(404, dict(error="Not found"))

        def do_POST(self):
            if self.path != "/predict":
                self.reply(404, dict(error="Not found"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    self.reply(
                        413, dict(error="Body size must be 1..2000000 bytes")
                    )
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                points = body["landmarks"]
                if not 8 <= len(points) <= 300:
                    raise ValueError("Use 8..300 frames")
                self.reply(
                    200,
                    dispatcher.predict_landmarks(
                        body.get("exercise", "squats"),
                        points,
                        float(body.get("aspect", 1.0)),
                    ),
                )
            except FileNotFoundError:
                self.reply(
                    503,
                    dict(
                        error="Requested exercise model has not finished training"
                    ),
                )
            except (ValueError, TypeError, KeyError, IndexError) as e:
                self.reply(400, dict(error=str(e)))

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()
    with make_server(a.port) as server:
        print(
            f"AI API: http://127.0.0.1:{server.server_port}/health", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
