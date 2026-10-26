"""교정 엔진의 실제 분석 입력을 세션별 JSONL과 확정 요약으로 저장합니다."""

import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class SessionWriter:
    def __init__(self, root, metadata):
        self.session_id = (
            time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            + "-"
            + uuid.uuid4().hex[:12]
        )
        self.directory = Path(root) / self.session_id
        self.directory.mkdir(parents=True, mode=0o700)
        self.frames = (self.directory / "frames.jsonl").open(
            "x", encoding="utf-8"
        )
        self.count = self.observed = self.predictions = 0
        self.errors = Counter()
        self.started_at = time.time()
        self.closed = False
        self.metadata = dict(
            metadata,
            schema_version=1,
            session_id=self.session_id,
            started_at=self.started_at,
            status="recording",
            landmark_order="MediaPipe Pose index 0..32",
            landmark_fields=["x", "y", "z", "visibility"],
            image_coordinates="normalized image x/y; MediaPipe relative z",
            recording_scope="Every pose-processed frame, including missing poses; not every camera capture frame",
            raw_video_saved=False,
            actual_repetition_count=None,
        )
        atomic_json(self.directory / "metadata.json", self.metadata)

    def append(self, raw, world, stamp, sequence, epoch, width, height, state):
        row = dict(
            index=self.count,
            captured_monotonic=stamp,
            written_unix=time.time(),
            camera_sequence=sequence,
            camera_epoch=epoch,
            image_width=width,
            image_height=height,
            aspect=width / height,
            landmarks=raw.tolist() if raw is not None else None,
            world_landmarks=world,
            status=state["status"],
            pose_visible=state["pose_visible"],
            buffer_frames=state["buffer_frames"],
            required_frames=state["required_frames"],
            prediction=state.get("result"),
        )
        # Missing joints remain null. No interpolation or synthetic normal label is added.
        if row["landmarks"] is not None and (
            len(row["landmarks"]) != 33
            or any(len(p) != 4 for p in row["landmarks"])
        ):
            raise ValueError("Expected 33 x 4 pose coordinates")
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self.frames.write(payload + "\n")
        self.frames.flush()
        self.count += 1
        self.observed += raw is not None
        if row["prediction"] is not None:
            self.predictions += 1
            self.errors.update(row["prediction"].get("detected_errors", []))

    def finish(self, state, reason):
        if self.closed:
            return
        self.frames.flush()
        os.fsync(self.frames.fileno())
        self.frames.close()
        self.closed = True
        summary = dict(
            self.metadata,
            status="failed" if state.get("error") else "completed",
            ended_at=time.time(),
            end_reason=reason,
            frame_count=self.count,
            observed_pose_frames=self.observed,
            missing_pose_frames=self.count - self.observed,
            prediction_count=self.predictions,
            error_window_counts=dict(self.errors),
            engine_error=state.get("error", ""),
            frames_sha256=digest(self.directory / "frames.jsonl"),
        )
        atomic_json(self.directory / "summary.json", summary)


class SessionReader:
    def __init__(self, root):
        self.root = Path(root)

    def list_sessions(self):
        result = []
        if not self.root.exists():
            return result
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            path = directory / "summary.json"
            if not path.exists():
                path = directory / "metadata.json"
            if path.exists():
                record = json.loads(path.read_text())
                if not (directory / "summary.json").exists():
                    record["status"] = "unfinalized"
                result.append(record)
        return result

    def read(self, session_id, offset=0, limit=1, verify=False):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{12}", session_id):
            raise ValueError("잘못된 세션 ID입니다.")
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("조회 범위가 올바르지 않습니다.")
        directory = self.root / session_id
        summary = directory / "summary.json"
        finalized = summary.exists()
        info = json.loads(
            (summary if finalized else directory / "metadata.json").read_text()
        )
        if not finalized:
            info["status"] = "unfinalized"
        rows = []
        with (directory / "frames.jsonl").open() as stream:
            for index, line in enumerate(stream):
                if index < offset:
                    continue
                if len(rows) == limit:
                    break
                rows.append(json.loads(line))
        matches = None
        if verify and finalized:
            matches = (
                digest(directory / "frames.jsonl") == info["frames_sha256"]
            )
        return dict(
            summary=info, frames=rows, offset=offset, integrity_matches=matches
        )
