"""Ubuntu 웹캠의 최신 프레임을 직접 연결된 유선망에 MJPEG로 제공한다."""

import argparse
import http.server
import threading
import time

import cv2


class Camera:
    """카메라를 하나만 열고 최신 JPEG 한 장만 공유한다."""

    def __init__(self, device, width, height, fps):
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"카메라를 열 수 없습니다: {device}")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        # 일부 드라이버는 설정을 지원하지 않는다. 최신 프레임 슬롯도 사용한다.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.condition = threading.Condition()
        self.running = True
        self.sequence = 0
        self.jpeg = None
        self.fps = fps
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        """캡처를 계속 읽되 JPEG 인코딩은 지정 FPS 이하로 수행한다."""
        next_encode = 0.0
        try:
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    print("카메라 읽기 실패: 연결 또는 다른 앱의 사용 여부를 확인하세요.")
                    break
                now = time.monotonic()
                if now < next_encode:
                    continue
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ok:
                    continue
                next_encode = now + 1.0 / self.fps
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            self.capture.release()
            with self.condition:
                self.running = False
                self.condition.notify_all()

    def close(self):
        """캡처 종료를 요청하고 작업 종료를 잠시 기다린다."""
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.worker.join(timeout=3)


def make_handler(camera):
    """같은 카메라의 최신 프레임을 클라이언트마다 읽는 핸들러를 만든다."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = (
                    '<!doctype html><meta charset="utf-8">'
                    '<title>유선 웹캠</title><h1>유선 웹캠 원본</h1>'
                    '<p>AI 분석 전 영상입니다.</p>'
                    '<img src="/video" style="max-width:100%">'
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/video":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.connection.settimeout(2)
            previous = 0
            try:
                while camera.running:
                    with camera.condition:
                        camera.condition.wait_for(
                            lambda: camera.sequence != previous or not camera.running,
                            timeout=1,
                        )
                        if not camera.running:
                            break
                        if camera.sequence == previous:
                            continue
                        previous, jpeg = camera.sequence, camera.jpeg
                    # 전체 이력을 쌓지 않고 다음 전송 시 최신 JPEG를 사용한다.
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (OSError, TimeoutError):
                pass  # 브라우저가 끊겨도 카메라 전체를 중지하지 않는다.

        def log_message(self, format, *args):
            return

    return Handler


def main():
    """지정한 유선 IP와 카메라로 시작하고 Ctrl+C로 종료한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, choices=range(1, 61), default=30)
    args = parser.parse_args()
    camera = Camera(args.camera, args.width, args.height, args.fps)
    try:
        with http.server.ThreadingHTTPServer(
            (args.host, args.port), make_handler(camera)
        ) as server:
            print(f"B의 브라우저에서 http://{args.host}:{args.port}/ 접속")
            print("해상도와 FPS는 카메라 지원 범위에 따라 달라집니다. 종료: Ctrl+C")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        camera.close()


if __name__ == "__main__":
    main()
