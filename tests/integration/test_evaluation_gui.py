"""점수 모드의 미평가 표시·기본 UI·결과 저장을 카메라 없이 검사합니다."""

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from GUI.FitHelper.mode_gui import ModeChooser
from GUI.FitHelper.score_mode_window import ScoreModeWindow, item_lines


def test_unknown_points_are_not_rendered_as_zero():
    assert item_lines(
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
    ) == ["복귀: AI 미평가 / 규칙 20"]


def test_score_only_uses_original_workout_ui_and_saves_completed_results(
    tmp_path,
):
    app = QApplication.instance() or QApplication([])
    ui = Path(__file__).resolve().parents[2] / "GUI/FitHelper/workout_window.ui"
    window = ScoreModeWindow(
        ui, base_url="http://127.0.0.1:9", sessions_root=tmp_path
    )
    window.timer.stop()
    try:
        state = {
            "running": False,
            "session_id": "abc123",
            "score": {
                "movement": {"count": 7},
                "repetitions": [
                    {
                        "number": 7,
                        "score_100": {
                            "items": [
                                {
                                    "id": "return",
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
        assert window.counterLabel.text() == "검출 7회"
        assert "전체 항목 채점 확정 0회" in window.result.toPlainText()
        assert "AI 미평가" in window.result.toPlainText()
        assert window.liveScoreLabel.text() == "재확인"
        window.save_result()
        assert json.loads(window.saved_path.read_text()) == state
        assert window.exercise == "squats"
    finally:
        window.allow_close = True
        window.close()


def test_mode_chooser_keeps_independent_modes_and_adds_integrated_mode():
    app = QApplication.instance() or QApplication([])
    chooser = ModeChooser(Path(__file__).resolve().parents[2] / "GUI/FitHelper")
    assert [chooser.mode.itemData(i) for i in range(chooser.mode.count())] == [
        "correction",
        "score",
        "combined",
    ]
    assert chooser.webcam.value() == 0
    chooser.close()
