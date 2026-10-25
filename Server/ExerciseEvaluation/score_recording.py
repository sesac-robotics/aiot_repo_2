"""실제 점수 입력·추적 신뢰도·모자이크 영상·최종 결과의 독립 기록입니다."""

import base64
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .paths import DATA_ROOT

DEFAULT_ROOT = DATA_ROOT / "score-mode-sessions"


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(
            json_value(data),
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class ScoreRecording:
    def __init__(self, context, exercise, root=DEFAULT_ROOT, provenance=None):
        session = context["session_id"]
        if not re.fullmatch("[a-f0-9]{32}", session):
            raise ValueError("잘못된 세션 ID입니다.")
        self.directory = (
            Path(root)
            / session
            / f"epoch-{context['epoch']}-{uuid.uuid4().hex[:8]}"
        )
        self.directory.mkdir(parents=True, mode=0o700)
        self.metadata = dict(
            schema_version=2,
            recording_id=f"{session}/{self.directory.name}",
            session_id=session,
            context=context,
            exercise=exercise,
            started_unix=time.time(),
            status="recording",
            raw_fields=[
                "x_normalized",
                "y_normalized",
                "z",
                "visibility",
                "presence",
            ],
            raw_scope="Every actual processed 10Hz input slot; missing slots are not synthesized",
            preview_scope="Mosaic JPEG from the exact analyzed frame; no raw video",
            provenance=provenance or {},
        )
        write_json(self.directory / "metadata.json", self.metadata)
        self.stream = (self.directory / "frames.jsonl").open("x")
        self.count = 0
        self.last_tick = None
        self.images = 0
        self.recoveries = 0
        self.finalized = False
        self.error = ""

    def append(
        self,
        tick,
        stamp,
        metadata,
        raw_video,
        raw_retry,
        chosen_raw,
        used_xy,
        quality,
        source,
        delivered,
        state,
        preview=None,
    ):
        try:
            if preview:
                image_path = self.directory / "preview"
                image_path.mkdir(exist_ok=True)
                name = f"{self.count:08d}.jpg"
                (image_path / name).write_bytes(
                    base64.b64decode(preview, validate=True)
                )
                preview_ref = "preview/" + name
                self.images += 1
            else:
                preview_ref = None
            row = dict(
                index=self.count,
                tick=tick,
                captured_monotonic=stamp,
                written_unix=time.time(),
                frame_id=metadata.get("frame_id"),
                image_width=metadata.get("shape", [0, 0])[1],
                image_height=metadata.get("shape", [0, 0])[0],
                missing_slots_before=(
                    0
                    if self.last_tick is None
                    else max(0, tick - self.last_tick - 1)
                ),
                raw_video_landmarks=raw_video,
                raw_retry_landmarks=raw_retry,
                raw_selected_landmarks=chosen_raw,
                scoring_xy=used_xy,
                extraction_source=source,
                quality=quality,
                delivered_to_tracker=delivered,
                readiness_released_actual_ticks=state.get(
                    "readiness_released_actual_ticks", []
                ),
                movement=state.get("movement"),
                scoring_readiness=state.get("scoring_readiness"),
                assessment_summary=state.get("assessment_summary"),
                preview=preview_ref,
            )
            self.stream.write(
                json.dumps(json_value(row), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            self.stream.flush()
            self.last_tick = tick
            self.count += 1
            self.recoveries += source == "independent_image_recovery"
        except Exception as exc:
            self.error = str(exc)
            raise RuntimeError("관절 기록 저장 실패: " + str(exc)) from exc

    def finish(self, state, reason):
        if self.finalized:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        complete = state.get("assessment_summary", {}).get(
            "all_detected_repetitions_scored", False
        )
        state = json_value(state)
        if state.get("recording"):
            state["recording"]["finalized"] = True
        summary = dict(
            self.metadata,
            status="failed" if self.error else "saved",
            assessment_status="complete" if complete else "not_complete",
            ended_unix=time.time(),
            end_reason=reason,
            frame_count=self.count,
            preview_count=self.images,
            recovered_frame_count=self.recoveries,
            recording_error=self.error,
            frames_sha256=sha256(self.directory / "frames.jsonl"),
            score=state,
        )
        write_json(self.directory / "summary.json", summary)
        self.finalized = True

    def snapshot(self):
        return dict(
            recording_id=self.metadata["recording_id"],
            directory=str(self.directory),
            frames=self.count,
            preview_frames=self.images,
            finalized=self.finalized,
            error=self.error,
        )


class ScoreRecordingReader:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)

    def list(self):
        result = []
        for path in sorted(
            self.root.glob("*/epoch-*/metadata.json"), reverse=True
        ):
            finalized = path.with_name("summary.json")
            row = json.loads(
                (finalized if finalized.exists() else path).read_text()
            )
            if not finalized.exists():
                row["status"] = "unfinalized"
            result.append(row)
        return sorted(result, key=lambda r: r["started_unix"], reverse=True)

    def read(self, recording_id, offset=0, verify=False):
        if (
            not re.fullmatch(
                r"[a-f0-9]{32}/epoch-[0-9]+-[a-f0-9]{8}", recording_id
            )
            or offset < 0
        ):
            raise ValueError("잘못된 기록 ID 또는 프레임 번호입니다.")
        directory = self.root / recording_id
        path = directory / "summary.json"
        info = json.loads(
            (path if path.exists() else directory / "metadata.json").read_text()
        )
        if not path.exists():
            info["status"] = "unfinalized"
        row = None
        with (directory / "frames.jsonl").open() as stream:
            for index, line in enumerate(stream):
                if index == offset:
                    row = json.loads(line)
                    break
        return dict(
            summary=info,
            frame=row,
            integrity_matches=(
                (sha256(directory / "frames.jsonl") == info["frames_sha256"])
                if verify and path.exists()
                else None
            ),
        )
