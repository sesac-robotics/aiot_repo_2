"""127.0.0.1 전용 업로드·유튜브 구간 분석·결과 영상 서버.

최대 300MiB 원본을 로컬 jobs 폴더에 저장한다. 분석은 로컬에서 수행한다.
유튜브 입력은 영상 다운로드를 위해 외부 서비스에 접속한다.
한 번에 한 영상만 분석하며 기존 프로젝트 서버와 별도 포트를 사용한다.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import re
import threading
import uuid
from urllib.parse import urlparse, parse_qs
from analyze import analyze
from review import load_review, save_review, export_csv
from youtube_input import validate_request, download_clip
from render_score_video import render
from saved_sources import SourceLibrary

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
STATES = {}


class Handler(BaseHTTPRequestHandler):
    """허용된 화면ㆍ분석 결과만 제공하는 로컬 HTTP 처리기."""

    def send(self, body, kind="application/json", status=200):
        """바이트 본문과 MIME 형식ㆍHTTP 상태를 전송한다. 반환값은 없다."""
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """화면ㆍ규칙ㆍ진행 상태 또는 Range 지원 MP4를 반환한다."""
        path = urlparse(self.path).path
        if path in ["/", "/index.html"]:
            return self.send(
                (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        if path == "/app.js":
            return self.send(
                (ROOT / "app.js").read_bytes(), "text/javascript; charset=utf-8"
            )
        if path == "/jobs":
            jobs = []
            current = json.loads((ROOT / "rules.json").read_text())["version"]
            for file in sorted(
                (ROOT / "jobs").glob("*/result.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                result = json.loads(file.read_text())
                if result["rules"]["version"] == current:
                    jobs.append(
                        {
                            "id": file.parent.name,
                            "exercise": result["exercise"],
                            "repetitions": len(result["repetitions"]),
                        }
                    )
            return self.send(json.dumps(jobs).encode())
        if path == "/sources":
            return self.send(
                json.dumps(
                    self.server.source_library.list(), ensure_ascii=False
                ).encode()
            )
        if path == "/rules.json":
            return self.send((ROOT / "rules.json").read_bytes())
        m = re.fullmatch(
            r"/jobs/([a-f0-9]{32})/(status|score-video\.mp4|overlay\.mp4|result\.json|review\.json|labels\.csv|reviewed\.csv)",
            path,
        )
        if not m:
            return self.send(b"{}", status=404)
        ident, name = m.groups()
        folder = ROOT / "jobs" / ident
        if name == "status":
            state = STATES.get(ident)
            if state is None and (folder / "result.json").is_file():
                result = json.loads((folder / "result.json").read_text())
                current = json.loads((ROOT / "rules.json").read_text())
                state = (
                    {"state": "done", "percent": 100, "result": result}
                    if result["rules"]["version"] == current["version"]
                    else {
                        "state": "failed",
                        "error": "이전 규칙 결과입니다. 새로 분석해 주세요.",
                    }
                )
            return self.send(
                json.dumps(
                    state or {"state": "unknown"}, ensure_ascii=False
                ).encode()
            )
        if name == "review.json":
            if not (folder / "result.json").is_file():
                return self.send(b"{}", status=404)
            return self.send(
                json.dumps(load_review(folder), ensure_ascii=False).encode()
            )
        if name in ["labels.csv", "reviewed.csv"]:
            if not (folder / "result.json").is_file():
                return self.send(b"{}", status=404)
            return self.send(
                export_csv(folder, reviewed_only=name == "reviewed.csv"),
                "text/csv; charset=utf-8",
            )
        file = folder / name
        if not file.is_file():
            return self.send(b"{}", status=404)
        if name == "result.json":
            return self.send(file.read_bytes())
        size = file.stat().st_size
        start, end = 0, size - 1
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
            if not match:
                return self.send(b"", status=416)
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
        if not 0 <= start <= end < size:
            return self.send(b"", status=416)
        self.send_response(206 if value else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if value:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining:
                block = f.read(min(remaining, 65536))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self):
        """업로드 또는 유튜브 구간 요청으로 백그라운드 분석을 시작한다.

        Returns:
            None. 접수 시 작업 ID와 HTTP 202, 잘못된 입력은 400,
            동시 분석은 409를 반환한다. 다운로드는 작업 스레드가 수행한다.
        """
        parsed = urlparse(self.path)
        origin = self.headers.get("Origin")
        if origin and origin not in [
            "http://127.0.0.1:8766",
            "http://localhost:8766",
        ]:
            return self.send(b"{}", status=403)
        review_match = re.fullmatch(r"/jobs/([a-f0-9]{32})/review", parsed.path)
        if review_match:
            folder = ROOT / "jobs" / review_match[1]
            if not (folder / "result.json").is_file():
                return self.send(b"{}", status=404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("검토 요청 크기 오류")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("검토 형식 오류")
                review = save_review(folder, payload)
            except (ValueError, TypeError) as e:
                return self.send(
                    json.dumps({"error": str(e)}, ensure_ascii=False).encode(),
                    status=400,
                )
            except RuntimeError as e:
                return self.send(
                    json.dumps({"error": str(e)}, ensure_ascii=False).encode(),
                    status=409,
                )
            return self.send(json.dumps(review, ensure_ascii=False).encode())
        if parsed.path == "/analyze-youtube":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("유튜브 요청 크기를 확인해 주세요.")
                self.connection.settimeout(15)
                request = validate_request(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError) as error:
                return self.send(
                    json.dumps(
                        {"error": str(error)}, ensure_ascii=False
                    ).encode(),
                    status=400,
                )
            if not LOCK.acquire(blocking=False):
                return self.send(b"{}", status=409)
            ident = uuid.uuid4().hex
            try:
                library = self.server.source_library
                source_id = library.remember(request)
                (ROOT / "jobs" / ident).mkdir(parents=True)
                STATES[ident] = {
                    "state": "downloading",
                    "percent": 0,
                    "message": "유튜브에서 선택 구간을 가져오고 있습니다.",
                }
                threading.Thread(
                    target=run_job,
                    args=(ident, request["exercise"], request),
                    kwargs={"library": library, "source_id": source_id},
                    daemon=True,
                ).start()
            except (OSError, ValueError, RuntimeError):
                LOCK.release()
                return self.send(b"{}", status=500)
            return self.send(json.dumps({"id": ident}).encode(), status=202)
        ex = parse_qs(parsed.query).get("exercise", [""])[0]
        if parsed.path != "/analyze" or ex not in ["squat", "pushup"]:
            return self.send(b"{}", status=400)
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send(b"{}", status=400)
        if not 0 < size <= 300 * 1024 * 1024:
            return self.send(b"{}", status=413)
        if not LOCK.acquire(blocking=False):
            return self.send(b"{}", status=409)
        ident = uuid.uuid4().hex
        folder = ROOT / "jobs" / ident
        folder.mkdir(parents=True)
        self.connection.settimeout(120)
        try:
            with (folder / "input.video").open("wb") as f:
                remaining = size
                while remaining:
                    block = self.rfile.read(min(remaining, 65536))
                    if not block:
                        raise ValueError("업로드 중단")
                    f.write(block)
                    remaining -= len(block)
        except Exception:
            LOCK.release()
            return self.send(b"{}", status=400)
        STATES[ident] = {"state": "processing", "percent": 0}
        threading.Thread(target=run_job, args=(ident, ex), daemon=True).start()
        self.send(json.dumps({"id": ident}).encode(), status=202)

    def do_DELETE(self):
        """유튜브 목록의 UUID 한 건만 제거한다. 분석 파일은 유지한다.

        Returns:
            None. 삭제는 HTTP 200, 없는 항목은 404, 외부 요청은 403.
        """
        origin = self.headers.get("Origin")
        if origin and origin not in (
            "http://127.0.0.1:8766",
            "http://localhost:8766",
        ):
            return self.send(b"{}", status=403)
        match = re.fullmatch(
            r"/sources/([a-f0-9]{32})", urlparse(self.path).path
        )
        if not match:
            return self.send(b"{}", status=404)
        removed = self.server.source_library.delete(match[1])
        return self.send(b"{}", status=200 if removed else 404)


def run_job(
    ident: str,
    exercise: str,
    youtube_request: dict | None = None,
    library: SourceLibrary | None = None,
    source_id: str | None = None,
) -> None:
    """저장한 영상 하나를 분석하고 성공ㆍ실패 상태를 기록한다.

    Args:
        ident: 서버가 생성한 UUID 문자열.
        exercise: squat 또는 pushup.
        youtube_request: 검증된 유튜브 구간 요청. None이면 업로드를 분석한다.
        library: 입력 기록 저장소. 업로드 분석에서는 사용하지 않는다.
        source_id: 분석 완료 후 연결할 입력 기록 UUID.
    Returns:
        None. STATES와 작업 폴더에 결과를 남긴다. 실패는 화면에 전달한다.
    """
    folder = ROOT / "jobs" / ident
    try:
        source = (
            download_clip(youtube_request, folder)
            if youtube_request is not None
            else None
        )

        def update_progress(state):
            """최종 영상 합성 전에는 완료 상태를 외부에 공개하지 않는다."""
            if state["state"] != "done":
                STATES[ident] = state

        result = analyze(
            folder / "input.video",
            folder,
            exercise,
            update_progress,
        )
        if source is not None:
            STATES[ident] = {
                "state": "rendering",
                "percent": 97,
                "message": "관절·점수 표시 영상을 만들고 있습니다.",
            }
            result["source"] = source
            (folder / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            render(folder)
            if library is not None and source_id is not None:
                library.complete(source_id, source["title"], ident)
        STATES[ident] = {"state": "done", "percent": 100, "result": result}
    except Exception as e:
        STATES[ident] = {"state": "failed", "error": str(e)}
    finally:
        LOCK.release()


if __name__ == "__main__":
    print("http://127.0.0.1:8766", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    server.source_library = SourceLibrary(ROOT)
    server.serve_forever()
