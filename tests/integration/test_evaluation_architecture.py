"""순수 상태 전이·저장 실패·의존 방향의 의미 있는 경계를 검사합니다."""

import json
from types import SimpleNamespace

import pytest

from Server.ExerciseEvaluation.adapters.session_repository import (
    JsonSessionRepository,
)
from Server.ExerciseEvaluation.application.integrated_session import (
    IntegratedSession,
)
from Server.ExerciseEvaluation.domain.lifecycle import (
    SessionLifecycle,
    SessionPhase,
)
from tools.check_source_contracts import check_sources


def test_lifecycle_rejects_invalid_start_and_pause_without_mutation():
    state = SessionLifecycle()
    with pytest.raises(ValueError):
        state.pause()
    assert state.phase is SessionPhase.IDLE
    state.start()
    with pytest.raises(ValueError):
        state.start()
    assert state.phase is SessionPhase.RUNNING


def test_pause_resume_and_stop_are_idempotent_without_another_counter():
    state = SessionLifecycle()
    state.start()
    assert state.pause() is True
    assert state.pause() is False
    assert state.resume() is True
    assert state.resume() is False
    state.stop()
    state.stop()
    assert state.phase is SessionPhase.STOPPED and not state.enabled
    state.start()
    assert state.enabled


def test_repository_preserves_schema_and_rejects_cross_session_paths(tmp_path):
    repo = JsonSessionRepository(tmp_path)
    session_id = "a" * 32
    manifest = {
        "session_id": session_id,
        "schema": "integrated-session-v1",
        "score": {"recording": None},
    }
    repo.save(session_id, manifest)
    assert repo.list_sessions() == [manifest]
    with pytest.raises(ValueError):
        repo.save("../outside", manifest)
    with pytest.raises(ValueError):
        repo.save("b" * 32, manifest)
    assert len(list(tmp_path.glob("*/summary.json"))) == 1


def test_corrupt_record_is_reported_instead_of_silently_disappearing(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    (folder / "summary.json").write_text("{broken")
    with pytest.raises(ValueError):
        JsonSessionRepository(tmp_path).list_sessions()


class FakeScore:
    def __init__(self):
        self.running = False
        self.resets = 0

    def start(self, options):
        self.running = True

    def stop(self):
        self.running = False

    def bind_session(self, session_id):
        self.session_id = session_id

    def snapshot(self):
        return {"running": self.running, "score": self.retained_score()}

    def retained_score(self):
        return {"recording": {}, "movement": {"count": 2}}

    def reset_motion(self):
        self.resets += 1


class MemoryRepository:
    def __init__(self):
        self.fail = False
        self.records = {}

    def save(self, session_id, manifest):
        if self.fail:
            raise OSError("fixture disk full")
        self.records[session_id] = manifest

    def list_sessions(self):
        return list(self.records.values())

    def score_epochs(self, directory):
        return []


def test_application_uses_injected_ports_and_can_retry_a_failed_save(tmp_path):
    resets = []
    engine = SimpleNamespace(
        reader=SimpleNamespace(paused=False),
        start=lambda *args: None,
        stop=lambda: None,
        reset=lambda: resets.append("reset"),
        snapshot=lambda: {"session_path": None, "recording_error": ""},
    )
    score = FakeScore()
    repository = MemoryRepository()
    service = IntegratedSession(
        engine, tmp_path, repository, lambda bridge: score
    )
    service.start(
        {"exercise": "squats", "mode": "combined", "camera": "webcam"}
    )
    service.pause()
    service.pause()
    assert score.resets == 1 and engine.reader.paused
    service.resume()
    service.resume()
    assert score.resets == 2 and resets == ["reset"]
    assert [e["event"] for e in service.events] == ["start", "pause", "resume"]
    repository.fail = True
    with pytest.raises(OSError):
        service.stop()
    assert service.last_summary is None
    assert service.lifecycle.phase is SessionPhase.STOPPED
    repository.fail = False
    service.stop()
    service.stop()
    assert len(repository.records) == 1
    assert service.list_sessions()[0]["score"]["movement"]["count"] == 2


def test_architecture_guard_rejects_gui_in_server_and_adapters_in_application(
    tmp_path,
):
    paths = {
        "Server/ExerciseEvaluation/domain/bad.py": "import cv2",
        "Server/ExerciseEvaluation/application/bad.py": "from Server.ExerciseEvaluation.adapters import bad",
        "Server/ExerciseEvaluation/adapters/bad.py": "from GUI.FitHelper import mode_gui",
    }
    for name, body in paths.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""경계 위반 합성 입력입니다."""\n' + body + "\n")
    _, errors = check_sources(tmp_path)
    assert any("non-domain dependency cv2" in error for error in errors)
    assert any(
        "application depends on implementation" in error for error in errors
    )
    assert any(
        "evaluation depends on presentation" in error for error in errors
    )
