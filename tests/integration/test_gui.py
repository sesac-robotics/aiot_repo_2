"""화면 없이 인증 상태와 Qt 요청 실행 경계를 검증합니다."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import requests
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QWidget

from GUI.FitHelper.application_context import ApplicationContext
from GUI.FitHelper.domain.auth_session import AuthSession
from GUI.FitHelper.login_client import LoginClient
from GUI.FitHelper.request_worker import RequestTask


class Response:
    def __init__(self, body, failure=False):
        self.body = body
        self.failure = failure

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.failure:
            raise requests.HTTPError("test failure")


def test_login_state_only_published_after_me_validation():
    session = AuthSession()
    session.establish("previous-token", {"user_id": 99})
    responses = iter(
        [
            Response({"success": True, "access_token": "new-token"}),
            Response({"success": False}, failure=True),
        ]
    )
    assert not LoginClient(session, lambda *a, **k: next(responses)).login(
        "user", "pw"
    )
    assert session.token is None
    assert session.user is None
    responses = iter(
        [
            Response({"success": True, "access_token": "new-token"}),
            Response({"success": True, "user": {"user_id": 1}}),
        ]
    )
    assert LoginClient(session, lambda *a, **k: next(responses)).login(
        "user", "pw"
    )
    user = session.user
    user["user_id"] = 99
    assert session.user["user_id"] == 1
    session.clear()
    assert session.token is None
    with pytest.raises(ValueError):
        session.establish("", {})


def test_qt_worker_keeps_gui_responsive_and_delivers_on_gui_thread():
    app = QApplication.instance() or QApplication([])
    app.aifit_context = ApplicationContext()
    window = QWidget()
    completed = []
    release = threading.Event()
    worker_threads = []

    def work():
        worker_threads.append(QThread.currentThread())
        assert release.wait(timeout=3)
        return 42

    def success(value):
        completed.append((value, QThread.currentThread()))

    task = RequestTask(
        window, work, success, lambda error: completed.append(error)
    )
    task.start()
    deadline = time.monotonic() + 3
    while not worker_threads and time.monotonic() < deadline:
        app.processEvents()
    assert worker_threads and worker_threads[0] != app.thread()
    assert not completed
    release.set()
    while not completed and time.monotonic() < deadline:
        app.processEvents()
    assert completed == [(42, app.thread())]
    window.close()


def test_gui_entry_imports_without_network_or_camera(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import must not start I/O")

    monkeypatch.setattr(requests, "request", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    from GUI.FitHelper import main, welcome_window, workout_window

    assert callable(main.main)
    assert welcome_window.WelcomeWindow
    assert workout_window.WorkoutWindow


def test_logout_invalidates_an_inflight_login_result():
    session = AuthSession()
    generation = session.begin_authentication()
    session.clear()
    assert (
        session.establish(
            "late-token", {"user_id": 1}, expected_revision=generation
        )
        is False
    )
    assert session.token is None
    current = session.begin_authentication()
    assert (
        session.establish(
            "current-token", {"user_id": 1}, expected_revision=current
        )
        is True
    )
