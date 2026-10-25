"""카메라를 열지 않고 Qt 선택 옵션과 미평가·완료 결과 표시를 검사합니다."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from GUI.FitHelper.exercise_evaluation_window import (
    ExerciseEvaluationWindow,
    item_lines,
)


def test_unknown_points_are_not_rendered_as_zero():
    lines = item_lines(
        {
            "items": [
                {
                    "id": "return",
                    "title": "복귀",
                    "ai_points": None,
                    "rule_points": 20,
                }
            ]
        }
    )
    assert lines == ["복귀: AI 미평가 / 규칙 20"]


def test_webcam_default_and_completed_results_do_not_change_user_selection():
    app = QApplication.instance() or QApplication([])
    window = ExerciseEvaluationWindow("http://127.0.0.1:9")
    window.timer.stop()
    try:
        assert window.camera.currentData() == "webcam"
        assert window.mode.currentData() == "combined"
        state = {
            "running": False,
            "status": "idle",
            "options": {
                "camera": "webcam",
                "mode": "combined",
                "exercise": "squats",
                "webcam": "0",
            },
            "correction": {"status": "not_requested"},
            "score": {
                "status": "stopped",
                "movement": {"count": 7},
                "repetitions": [
                    {
                        "number": 7,
                        "score_100": {
                            "items": [
                                {
                                    "id": "return",
                                    "title": "복귀",
                                    "ai_points": None,
                                    "rule_points": 20,
                                }
                            ]
                        },
                    }
                ],
            },
        }
        window.render_state(state)
        assert "완료 7회" in window.result.toPlainText()
        assert "AI 미평가" in window.result.toPlainText()
        assert "선택하지 않은 평가" in window.result.toPlainText()
        window.camera.setCurrentIndex(window.camera.findData("esp32"))
        window.render_state(state)
        assert window.camera.currentData() == "esp32"
        assert app is QApplication.instance()
    finally:
        window.allow_close = True
        window.close()
