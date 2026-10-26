"""현재 저장소 GUI 연결·격리 실행 환경·개인 파일 제외 경계를 검사합니다."""

import os
import subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from GUI.FitHelper import welcome_window
from GUI.FitHelper.mode_gui import create_login
from GUI.FitHelper.runtime_config import correction_environment
from Server.ExerciseEvaluation.paths import CORRECTION, REPO


def test_current_namespaced_login_reaches_mode_dashboard_without_analysis(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    original = welcome_window.WelcomeWindow
    window = create_login(
        REPO / "GUI/FitHelper", integrated_url="http://127.0.0.1:19873"
    )
    try:
        # The unchanged login uses this fully qualified module after success.
        window._login_completed(True)
        dashboard = window.welcome_window
        assert isinstance(dashboard, original)
        assert "운동 모드 선택" in dashboard.startWorkoutButton.text()
        assert dashboard.has_completed_analysis is False
        dashboard.go_to_workout()
        assert dashboard.mode_chooser.integrated_url == "http://127.0.0.1:19873"
        assert dashboard.mode_chooser.mode.count() == 3
        dashboard.mode_chooser.close()
        dashboard.close()
    finally:
        window.close()
        welcome_window.WelcomeWindow = original
        app.processEvents()


def test_legacy_import_path_is_limited_to_correction_child(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    env = correction_environment()
    assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO), str(CORRECTION)]
    assert os.environ["PYTHONPATH"] == "/unrelated/source"


def test_private_data_and_local_environments_are_ignored():
    paths = [
        ".aifit-local/AIFit.db",
        ".aifit-local/local-jwt.key",
        ".aifit-local/score-mode-sessions/example/frames.jsonl",
        ".venv-score/bin/python",
        ".venv-correction/bin/python",
        "output/private-record.png",
    ]
    ignored = subprocess.check_output(
        ["git", "check-ignore", *paths], cwd=REPO, text=True
    ).splitlines()
    assert ignored == paths
