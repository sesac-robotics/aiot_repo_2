"""원본 교정 유지·단일 카메라·방향 정규화·정지 입력 계약 검사입니다."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from GUI.FitHelper.integrated_session import PausingReader, SharedScoreCamera


def test_shared_camera_does_not_open_or_close_the_original_camera():
    frame = np.zeros((480, 640, 3), np.uint8)
    stamp = time.monotonic()

    class Reader:
        transport = "stream"
        starts = stops = 0

        def start(self):
            self.starts += 1

        def stop(self):
            self.stops += 1

        def snapshot(self):
            return 1, 1, (frame, stamp), True, "", 30

    original = Reader()
    reader = PausingReader(original)
    b = SimpleNamespace(
        engine=SimpleNamespace(reader=reader), enabled=True, paused=False
    )
    shared = SharedScoreCamera(b)
    shared.start()
    result = shared.snapshot()
    shared.stop()
    assert result[2][0] is frame and result[2][1] == stamp
    assert original.starts == original.stops == 0
    b.paused = True
    assert shared.snapshot()[2] is None
    b.paused = False
    assert shared.snapshot()[2] is None
    original.snapshot = lambda: (2, 1, (frame, stamp + 0.1), True, "", 30)
    assert shared.snapshot()[2][0] is frame


def test_rotated_video_is_normalized_once_for_both_engines():
    frame = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    stamp = time.monotonic()
    original = SimpleNamespace(
        transport="stream",
        snapshot=lambda: (1, 1, (frame, stamp), True, "", 30),
    )
    reader = PausingReader(original)
    reader.rotation = 90
    first = reader.snapshot()[2]
    second = reader.raw_snapshot()[2]
    assert first[0].shape == (4, 3, 3)
    assert first[0] is second[0] and first[1] == stamp
    assert np.array_equal(first[0][0, -1], frame[0, 0])
    reader.paused = True
    assert reader.snapshot()[2] is None
    assert reader.raw_snapshot()[2][0] is first[0]


def test_integrated_gui_keeps_original_top_edge_without_added_banners():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from GUI.FitHelper.integrated_window import IntegratedWindow

    app = QApplication.instance() or QApplication([])
    window = IntegratedWindow(
        Path("GUI/FitHelper/workout_window.ui"), base_url="http://127.0.0.1:9"
    )
    window.timer.stop()
    assert (
        window.notice.isHidden()
        and window.quality_notice.isHidden()
        and window.correction_notice.isHidden()
    )
    assert (
        window.videoLabel.size().width() == 640
        and window.videoLabel.size().height() == 480
    )
    window.allow_close = True
    window.close()


def test_unsupported_auto_camera_does_not_allocate_or_stop_a_session():
    import pytest

    from GUI.FitHelper.integrated_session import IntegratedSession

    bridge = IntegratedSession(None, "unused-python", "/unused")
    with pytest.raises(ValueError, match="webcam 또는 esp32"):
        bridge.start({"mode": "combined", "camera": "auto"})
    assert bridge.session_id is None


def test_webcam_requests_vga_without_changing_recorded_video_or_stream():
    from GUI.FitHelper.integrated_session import WebcamVgaCapture

    calls = []

    class Capture:
        def isOpened(self):
            return True

        def set(self, key, value):
            calls.append((key, value))
            return True

    backend = SimpleNamespace(
        VideoCapture=lambda source, *a, **kw: Capture(),
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
    )
    configured = WebcamVgaCapture(backend)
    configured.VideoCapture(0)
    assert calls == [(3, 640), (4, 480)]
    calls.clear()
    configured.VideoCapture("recorded.mp4")
    configured.VideoCapture("http://camera/stream")
    assert calls == []
