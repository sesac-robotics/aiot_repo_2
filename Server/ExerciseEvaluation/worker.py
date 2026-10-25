"""원시 BGR 프레임을 받아 선택한 평가 한 종류만 격리 실행합니다."""

import contextlib
import json
import struct
import sys

MAX_PACKET = 24_000_000


def read_exact(stream, length):
    """파이프 분할 수신을 조립하며 중간 종료를 감지합니다."""
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            raise EOFError("평가 입력이 종료됐습니다.")
        chunks.extend(part)
    return bytes(chunks)


def final_response(adapter, kind, epoch, meta):
    """중지 직전 소스가 바뀌었으면 이전 소스의 완료 결과도 전달하지 않습니다."""
    response = {
        "generation": meta["generation"],
        "epoch": meta["epoch"],
        "timestamp": meta["timestamp"],
        "status": "stopped",
        "final": True,
    }
    if adapter and kind == "score" and epoch == meta["epoch"]:
        with contextlib.redirect_stdout(sys.stderr):
            response["repetitions"] = adapter.flush()
            if hasattr(adapter, "finalize_recording"):
                adapter.finalize_recording()
            snapshot = adapter.snapshot()
            response["movement"] = snapshot["movement"]
            for key in (
                "assessment_summary",
                "recording",
                "joint_quality",
                "scoring_readiness",
            ):
                if key in snapshot:
                    response[key] = snapshot[key]
    return response


def main():
    """표준 출력은 JSON 결과 전용으로 유지하고 라이브러리 출력을 분리합니다."""
    kind, exercise = sys.argv[1:3]
    output = sys.stdout
    adapter = None
    epoch = None
    load_error = None
    try:
        while True:
            length = struct.unpack("!I", read_exact(sys.stdin.buffer, 4))[0]
            if length > MAX_PACKET:
                raise ValueError("평가 입력이 너무 큽니다.")
            metadata, raw = read_exact(sys.stdin.buffer, length).split(b"\n", 1)
            meta = json.loads(metadata)
            if meta.get("command") == "stop":
                response = final_response(adapter, kind, epoch, meta)
                output.write(
                    json.dumps(response, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                output.flush()
                break
            if meta.get("command") == "reset_motion":
                with contextlib.redirect_stdout(sys.stderr):
                    if adapter and kind == "score" and epoch == meta["epoch"]:
                        adapter.reset_motion()
                        snapshot = adapter.snapshot()
                    else:
                        snapshot = {}
                output.write(
                    json.dumps(
                        {**meta, **snapshot, "motion_reset": True},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                output.flush()
                continue
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    import numpy as np

                    if epoch != meta["epoch"]:
                        if adapter:
                            adapter.close()
                        adapter = None
                        epoch = meta["epoch"]
                        load_error = None
                        try:
                            if kind == "score":
                                from .score_adapter import ScoreAdapter

                                adapter = ScoreAdapter(
                                    exercise,
                                    session_context={
                                        "session_id": meta.get("session_id")
                                        or __import__("uuid").uuid4().hex,
                                        "generation": meta["generation"],
                                        "epoch": meta["epoch"],
                                    },
                                )
                                adapter.live_enabled = meta.get(
                                    "provisional_scores", True
                                )
                            elif kind == "correction":
                                from .correction_adapter import (
                                    CorrectionAdapter,
                                )

                                adapter = CorrectionAdapter(exercise)
                            else:
                                raise ValueError("알 수 없는 평가 종류입니다.")
                        except Exception as error:
                            load_error = str(error)
                    if load_error:
                        raise RuntimeError(load_error)
                    frame = np.frombuffer(raw, np.uint8).reshape(meta["shape"])
                    result = (
                        adapter.process(frame, meta["timestamp"], meta)
                        if kind == "score"
                        else adapter.process(frame, meta["timestamp"])
                    )
                if result is None:
                    continue
                response = {
                    "generation": meta["generation"],
                    "epoch": meta["epoch"],
                    "frame_id": meta["frame_id"],
                    "timestamp": meta["timestamp"],
                    **result,
                }
            except Exception as error:
                response = {
                    "generation": meta["generation"],
                    "epoch": meta["epoch"],
                    "timestamp": meta["timestamp"],
                    "status": "unavailable",
                    "error": str(error),
                }
            output.write(
                json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n"
            )
            output.flush()
    except EOFError:
        pass
    finally:
        if adapter:
            with contextlib.redirect_stdout(sys.stderr):
                adapter.close()


if __name__ == "__main__":
    main()
