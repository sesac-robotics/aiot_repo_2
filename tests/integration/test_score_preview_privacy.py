"""표시 영상 모자이크, 원본 입력 보존, 세션이 다른 미리보기 차단 검사입니다."""

import ast
import base64
import time
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
from Server.ExerciseEvaluation.coordinator import Coordinator
from Server.ExerciseEvaluation.paths import SCORE_LIVE
from Server.ExerciseEvaluation.preview_privacy import (
    encode_masked_preview,
    mosaic_faces,
)


def test_face_mosaic_matches_preserved_implementation():
    original = ast.parse((SCORE_LIVE / "server.py").read_text())
    function = next(
        n
        for n in original.body
        if isinstance(n, ast.FunctionDef) and n.name == "mosaic_faces"
    )
    scope = {"np": np, "cv2": cv2}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "frozen-mosaic",
            "exec",
        ),
        scope,
    )
    image = np.random.default_rng(1).integers(
        0, 255, (240, 320, 3), dtype=np.uint8
    )
    xy = np.full((33, 2), np.nan)
    xy[:11] = [[120 + i * 5, 80 + (i % 3) * 6] for i in range(11)]
    a = image.copy()
    b = image.copy()
    before = xy.copy()
    scope["mosaic_faces"](a, xy)
    assert mosaic_faces(b, xy) is True
    assert np.array_equal(a, b)
    assert np.array_equal(xy, before, equal_nan=True)
    encoded, mode = encode_masked_preview(image, xy)
    assert mode == "face"
    assert (
        cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
        ).shape
        == image.shape
    )
    assert np.array_equal(
        image,
        np.random.default_rng(1).integers(
            0, 255, (240, 320, 3), dtype=np.uint8
        ),
    )


def test_missing_face_uses_full_frame_mosaic_instead_of_raw_image():
    image = np.random.default_rng(2).integers(
        0, 255, (240, 320, 3), dtype=np.uint8
    )
    encoded, mode = encode_masked_preview(image, np.full((33, 2), np.nan))
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR
    )
    assert mode == "full_frame"
    assert not np.array_equal(decoded, image)


def test_camera_raw_frame_is_sent_to_worker_but_never_published():
    c = Coordinator({}, fixed_mode="score")
    c.state["session_id"] = "a" * 32
    image = np.random.default_rng(3).integers(
        0, 255, (240, 320, 3), dtype=np.uint8
    )
    captured = []

    class Camera:
        selected = "webcam"

        def snapshot(self):
            return 1, 1, (image, time.monotonic()), True, "", 25

    class Worker:
        def submit(self, frame, meta):
            captured.append(frame.copy())
            c.stop_event.set()

    c.camera = Camera()
    c.workers = {"score": Worker()}
    c._run()
    assert np.array_equal(captured[0], image)
    assert c.jpeg is None


def test_masked_packet_is_not_left_in_state_and_old_epoch_cannot_publish():
    c = Coordinator({}, fixed_mode="score")
    c.generation = 1
    c.state = {"running": True, "status": "running", "epoch": 2}
    encoded, _ = encode_masked_preview(
        np.zeros((240, 320, 3), np.uint8), np.full((33, 2), np.nan)
    )
    result = {
        "generation": 1,
        "epoch": 2,
        "timestamp": time.monotonic(),
        "preview_jpeg_b64": encoded,
        "face_mosaic": "full_frame",
    }
    c.receive("score", result)
    assert c.jpeg == base64.b64decode(encoded)
    assert "preview_jpeg_b64" not in c.state["score"]
    before = c.jpeg
    c.receive("score", dict(result, epoch=1, preview_jpeg_b64="wrong"))
    assert c.jpeg == before
