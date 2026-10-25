"""기존 실험 화면과 QEVD 학습 진행·판정 결과를 로컬에서 제공한다.

127.0.0.1 전용이며 원본 동영상은 조사된 QEVD 식별자로만 제공한다.
기존 HTML은 디스크에서 수정하지 않고 응답에 새 화면 링크만 추가한다.
"""

import functools
import http.server
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent


class CoachHandler(http.server.SimpleHTTPRequestHandler):
    """정적 실험 화면과 허용된 시험 영상만 제공하는 HTTP 어댑터."""

    def do_GET(self) -> None:
        """GET 요청에 결과·진행·허용 영상 또는 기존 정적 파일을 반환한다.

        Raises:
            OSError: 파일 읽기 또는 클라이언트 전송 실패.
        """
        if self.path.split("?")[0] == "/selected-exercise.html":
            page = (PILOT / "selected-exercise.html").read_text()
            banner = (
                '<p style="padding:18px;background:#17524d">'
                '<a href="/qevd_coach/view.html">'
                "QEVD 4종 · Heavy 교정 모델 보기 →</a></p>"
            )
            page = page.replace("<h1>", banner + "<h1>", 1)
            self._send(page.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/qevd-api/"):
            name = self.path.split("/")[-1].split("?")[0]
            if name not in ("progress.json", "results.json", "overlays.json"):
                self.send_error(404)
                return
            path = (
                ROOT / "overlays/status.json"
                if name == "overlays.json"
                else ROOT / name
            )
            self._send(
                path.read_bytes() if path.exists() else b"{}",
                "application/json",
            )
        elif self.path.startswith("/qevd-video/"):
            video_id = self.path.split("/")[-1]
            if not re.fullmatch(r"\d{8}", video_id):
                self.send_error(404)
                return
            catalog = json.loads((ROOT / "catalog.json").read_text())
            item = next(
                (
                    r
                    for r in catalog["items"]
                    if r["id"] == video_id and r["split"] == "test"
                ),
                None,
            )
            if item is None:
                self.send_error(404)
                return
            overlay = ROOT / "overlays" / (video_id + ".mp4")
            if not overlay.exists():
                self.send_error(503, "Overlay is being prepared")
                return
            self._video(overlay)
        elif "/artifacts/" in self.path or self.path.endswith("catalog.json"):
            self.send_error(403)
        else:
            super().do_GET()

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
            if match is None:
                self.send_error(416)
                return
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
        if not 0 <= start <= end < size:
            self.send_error(416)
            return
        self.send_response(206 if value else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if value:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                block = source.read(min(remaining, 1024 * 1024))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)


def main() -> None:
    """포트 8765에서 로컬 서버를 시작하고 종료 시 소켓을 해제한다.

    Raises:
        OSError: 해당 포트가 이미 사용 중이거나 파일 접근이 실패한 경우.
    """
    handler = functools.partial(CoachHandler, directory=str(PILOT))
    with http.server.ThreadingHTTPServer(
        ("127.0.0.1", 8765), handler
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
