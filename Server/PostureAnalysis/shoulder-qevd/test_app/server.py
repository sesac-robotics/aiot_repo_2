"""Local, read-only dataset test UI. Run with Python 3; no pip install needed."""
from pathlib import Path
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import argparse
import csv
import json
import mimetypes
import re
import subprocess
import sys

APP = Path(__file__).resolve().parent
SHOULDER = APP.parent
QEVD = SHOULDER.parent / "qevd-selected"
ROOTS = {"shoulder": SHOULDER, "qevd": QEVD}
GROUPS = {"squats": "squats", "pushups": "pushups", "elbow plank": "elbow_plank", "shoulder press": "shoulder_press"}
DISPLAY = {"squats": "스쿼트", "pushups": "푸시업", "elbow_plank": "엘보 플랭크", "ytw": "YTW", "shoulder_press": "숄더 프레스"}


def read(path):
    return json.loads(path.read_text())


def safe_path(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def media_url(dataset, relative):
    from urllib.parse import quote
    return "/media/" + dataset + "/" + quote(relative, safe="/")


class Catalog:
    def __init__(self):
        self.records = []
        self.index = {}
        for dataset, root in ROOTS.items():
            with (root / "manifests/short_clips.csv").open() as stream:
                manifest = {r["video_id"]: r for r in csv.DictReader(stream)}
            reviews_file = root / "manifests/annotation_review.json"
            reviews = {r["video_id"]: r for r in read(reviews_file)} if reviews_file.exists() else {}
            original_feedbacks = {Path(r["video_path"]).stem: r["feedbacks"] for r in read(root / "raw/coach/feedbacks_short_clips.json")}
            derived = root / "derived/coach/feedbacks_short_clips.json"
            feedbacks = {Path(r["video_path"]).stem: r["feedbacks"] for r in read(derived)} if derived.exists() else original_feedbacks
            mapping_file = root / "derived/manifests/feedback_label_mapping.json"
            label_mapping = {r["label"]: r for r in read(mapping_file)} if mapping_file.exists() else {}
            for annotation in read(root / "raw/coach/fine_grained_labels.json"):
                video_id = Path(annotation["video_path"]).stem
                entry, review = manifest[video_id], reviews.get(video_id)
                prefix = annotation["labels"][0].split(" - ", 1)[0]
                group = "ytw" if prefix.startswith("standing ") else GROUPS.get(prefix, prefix)
                item = {
                    "key": dataset + ":" + video_id, "dataset": dataset, "video_id": video_id,
                    "exercise": group, "exercise_name": DISPLAY.get(group, group),
                    "split": annotation["split"], "worker_id": entry["worker_id"],
                    "subject_id": review["subject_id"] if review else "worker_" + entry["worker_id"],
                    "view": review["view"] if review else None,
                    "labels": annotation["labels"], "labels_descriptive": annotation["labels_descriptive"],
                    "labels_ko": review["labels_ko"] if review else [],
                    "feedbacks": feedbacks.get(video_id, []), "feedbacks_ko": review["feedbacks_ko"] if review else [],
                    "feedback_status": "draft" if review or derived.exists() else "source_dataset",
                    "feedback_origin": "assistant_authored_label_mapping" if derived.exists() else "assistant_authored_template" if review else "source_dataset",
                    "original_feedbacks": original_feedbacks.get(video_id, []) if derived.exists() else [],
                    "label_feedback_mapping": [label_mapping[l] for l in annotation["labels"] if l in label_mapping],
                    "status": review["annotation_status"] if review else "QEVD 원본 라벨",
                    "label_origin": review["label_origin"] if review else "QEVD original annotation",
                    "source_quality": review["source_quality"] if review else None,
                    "notes": review["notes_ko"] if review else "QEVD 원본 라벨에 영어 피드백 템플릿 초안을 연결했습니다. 정상 특성·운동 변형·판독 불가·모호한 라벨을 구분했습니다. 원본 자유형 피드백은 별도로 보존됩니다." if derived.exists() else "QEVD 원본 라벨과 피드백을 그대로 표시합니다.",
                    "media_url": media_url(dataset, entry["video_path"]),
                    "duration_sec": review["duration_sec"] if review else None,
                    "evidence": None,
                }
                if review and review["example_evidence"]:
                    ex = review["example_evidence"]
                    item["evidence"] = {
                        "image_url": media_url(dataset, ex["comparison_image"]),
                        "observation": ex["observation_ko"],
                        "good_url": media_url(dataset, ex["good"]["clip_path"]),
                        "bad_url": media_url(dataset, ex["bad"]["clip_path"]),
                        "good_time": ex["good"]["evidence_time_sec"],
                        "bad_time": ex["bad"]["evidence_time_sec"],
                    }
                self.records.append(item)
                self.index[item["key"]] = item

    def filtered(self, query):
        def value(name):
            return query.get(name, [""])[0]
        dataset, exercise, split, label = [value(k) for k in ("dataset", "exercise", "split", "label")]
        search, evidence = value("q").casefold(), value("evidence") == "1"
        return [r for r in self.records
                if (not dataset or r["dataset"] == dataset)
                and (not exercise or r["exercise"] == exercise)
                and (not split or r["split"] == split)
                and (not label or label in r["labels"])
                and (not evidence or r["evidence"])
                and (not search or search in " ".join([r["video_id"], r["subject_id"], *r["labels"], *r["labels_ko"], *r["feedbacks_ko"]]).casefold())]

    def stats(self):
        return {
            "total": len(self.records), "datasets": dict(Counter(r["dataset"] for r in self.records)),
            "exercises": [{"id": k, "name": DISPLAY[k], "count": sum(r["exercise"] == k for r in self.records)} for k in DISPLAY],
            "shoulder_build": read(SHOULDER / "checks/build_report.json")["counts"],
        }


def parse_range(header, size):
    if not header:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if first:
        start, end = int(first), int(last) if last else size - 1
    else:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(size - length, 0), size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, min(end, size - 1), True


class Handler(BaseHTTPRequestHandler):
    def json_response(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def file_response(self, path):
        size = path.stat().st_size
        try:
            start, end, partial = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(end - start + 1, 0)))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/stats":
                return self.json_response(self.server.catalog.stats())
            if url.path == "/api/clips":
                rows = self.server.catalog.filtered(query)
                limit = max(1, min(int(query.get("limit", [24])[0]), 100))
                page = max(1, int(query.get("page", [1])[0]))
                start = (page - 1) * limit
                # Feedback details are fetched for the selected clip only.
                keys = ("key", "video_id", "dataset", "exercise_name", "subject_id", "view", "labels", "labels_ko", "split", "source_quality", "status")
                return self.json_response({"total": len(rows), "page": page, "items": [{k: r[k] for k in keys} for r in rows[start:start + limit]],
                                           "labels": sorted({label for r in self.server.catalog.filtered({k: v for k, v in query.items() if k != "label"}) for label in r["labels"]})})
            if url.path == "/api/clip":
                item = self.server.catalog.index.get(query.get("key", [""])[0])
                return self.json_response(item if item else {"error": "영상이 없습니다."}, 200 if item else 404)
            if url.path.startswith("/media/"):
                pieces = unquote(url.path).split("/", 3)
                if len(pieces) != 4 or pieces[2] not in ROOTS:
                    raise FileNotFoundError()
                return self.file_response(safe_path(ROOTS[pieces[2]], pieces[3]))
            if url.path == "/api/health":
                return self.json_response({"status": "ok", "clips": len(self.server.catalog.records)})
            relative = "index.html" if url.path == "/" else unquote(url.path).lstrip("/")
            return self.file_response(safe_path(APP / "web", relative))
        except (BrokenPipeError, ConnectionResetError):
            return
        except FileNotFoundError:
            self.json_response({"error": "파일을 찾을 수 없습니다."}, 404)
        except ValueError:
            self.json_response({"error": "잘못된 요청입니다."}, 400)

    def do_POST(self):
        if self.path != "/api/validate":
            return self.json_response({"error": "지원하지 않는 요청입니다."}, 404)
        # Browser button executes a fixed read-only integrity check, not input code.
        if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            return self.json_response({"error": "다른 사이트에서 검증을 실행할 수 없습니다."}, 403)
        try:
            result = subprocess.run([sys.executable, str(SHOULDER / "validate_dataset.py")], capture_output=True, text=True, timeout=60)
            if result.returncode:
                return self.json_response({"status": "failed", "error": result.stderr[-3000:]}, 500)
            return self.json_response(json.loads(result.stdout))
        except subprocess.TimeoutExpired:
            return self.json_response({"status": "failed", "error": "검증 시간이 초과되었습니다."}, 504)


def make_server(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.catalog = Catalog()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f"Dataset test UI: http://127.0.0.1:{server.server_port} ({len(server.catalog.records):,} clips)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
