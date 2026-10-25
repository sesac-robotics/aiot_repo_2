"""세션 상태와 모델 어댑터의 의존 방향 및 지연 이벤트 차단을 검증합니다."""

import ast
from pathlib import Path

import pytest

from app.session.manager import SessionManager
from app.session.models import SessionConflict


def test_session_core_has_no_analysis_or_framework_imports():
    root = Path(__file__).resolve().parents[2] / "Server/AIService/app/session"
    for file in root.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(
                    ("app.analysis", "fastapi", "cv2", "torch")
                ), file


def test_duplicate_and_stale_events_cannot_increment_current_exercise():
    manager = SessionManager()
    manager.start_session(1)
    manager.start_exercise(11, "dumbbell_shoulder_press", 1, 10)
    context = manager.get_analysis_context()
    assert manager.record_analysis_rep(context, "rep-1") is True
    assert manager.record_analysis_rep(context, "rep-1") is False
    assert manager.get_snapshot()["exercise"]["rep_count"] == 1
    manager.pause_session()
    manager.resume_session()
    assert manager.record_analysis_rep(context, "late-rep") is False
    assert manager.get_snapshot()["exercise"]["rep_count"] == 1
    manager.finish_exercise()
    with pytest.raises(SessionConflict):
        manager.start_exercise(11, "dumbbell_shoulder_press", 2, 10)


def test_internal_api_preserves_auth_and_session_transition_contracts():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers.internal_sessions import router

    application = FastAPI()
    application.include_router(router)
    application.state.session_manager = SessionManager()
    application.state.internal_api_key = "test-internal-key"
    with TestClient(application) as client:
        assert client.post("/internal/sessions/1/start").status_code == 401
        headers = {"X-Internal-API-Key": "test-internal-key"}
        assert (
            client.post(
                "/internal/sessions/1/start", headers=headers
            ).status_code
            == 200
        )
        path = "/internal/sessions/1/exercises/11"
        response = client.post(
            path + "/start",
            headers=headers,
            json={
                "exercise_name": "dumbbell_shoulder_press",
                "order_no": 1,
                "target_reps": 10,
            },
        )
        assert response.status_code == 200, response.text
        assert (
            client.post(
                "/internal/sessions/1/finish", headers=headers
            ).status_code
            == 409
        )
        manager = application.state.session_manager
        context = manager.get_analysis_context()
        manager.record_analysis_rep(context, "rep-1")
        manager.record_analysis_rep(context, "rep-1")
        response = client.get("/internal/sessions/1/live", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["exercise"]["rep_count"] == 1
        assert client.post(path + "/finish", headers=headers).status_code == 200
        assert (
            client.post(
                "/internal/sessions/1/finish", headers=headers
            ).status_code
            == 200
        )
