"""실제 저장소 대신 임시 DB로 인증·소유권·트랜잭션 계약을 검증합니다."""

import asyncio
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from Server.AIFitService.adapters.ai_gateway import GatewayError, HTTPAIGateway
from Server.AIFitService.config import settings
from Server.AIFitService.database import connection
from Server.AIFitService.domain.routine import RoutineError, day_exercises
from Server.AIFitService.main import app
from Server.AIFitService.routers import body_analysis_router, workout_router
from Server.AIFitService.services import auth_service, token_service


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(connection, "DB_PATH", tmp_path / "isolated.db")
    monkeypatch.setattr(
        settings, "JWT_SECRET", "test-key-only-never-deploy-" * 2
    )
    monkeypatch.setattr(auth_service, "JWT_SECRET", settings.JWT_SECRET)
    monkeypatch.setattr(token_service, "JWT_SECRET", settings.JWT_SECRET)
    connection.init_db()
    body_analysis_router.init_body_analysis_db()
    workout_router.init_workout_db()
    return connection


def test_transaction_closes_commits_and_rolls_back(database):
    with database.get_connection() as conn:
        conn.execute("CREATE TABLE transaction_test (value INTEGER)")
        conn.execute("INSERT INTO transaction_test VALUES (1)")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with database.get_connection() as failed:
            failed.execute("INSERT INTO transaction_test VALUES (2)")
            raise RuntimeError("simulate application failure")
    with database.get_connection() as check:
        assert [
            row[0]
            for row in check.execute("SELECT value FROM transaction_test")
        ] == [1]
    with pytest.raises(sqlite3.ProgrammingError):
        failed.execute("SELECT 1")


def test_empty_database_startup_and_auth_contract(database):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/me").status_code == 401
        payload = dict(
            username="integration-user",
            password="Testpass1!",
            email_address="integration@example.com",
            nickname="통합검증",
            gender="male",
            age=25,
            height=175,
            weight=70,
        )
        response = client.post("/api/auth/register", json=payload)
        assert response.status_code == 201, response.text
        assert (
            client.post("/api/auth/register", json=payload).status_code == 409
        )
        login = client.post(
            "/api/auth/login",
            json={
                "username": payload["username"],
                "password": payload["password"],
            },
        )
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        profile = client.get(
            "/api/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert profile.status_code == 200, profile.text
        assert "password_hash" not in profile.text
        assert (
            client.post(
                "/api/auth/login",
                json={"username": payload["username"], "password": "incorrect"},
            ).status_code
            == 401
        )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_hold_targets_cannot_reach_ai(value):
    import json

    plan = {
        "days": [
            {
                "day_no": 1,
                "exercises": [
                    {
                        "exercise_name": "plank",
                        "sets": 3,
                        "target_duration_sec": value,
                    }
                ],
            }
        ]
    }
    with pytest.raises(RoutineError):
        day_exercises(json.dumps(plan), 1)


def test_legacy_routine_and_http_error_mapping():
    plan = '{"day_1":{"exercises":[{"exercise_name":"숄더 프레스", "sets":3,"reps":10}]}}'
    result = day_exercises(plan, 1)
    assert result[0]["order_no"] == 1
    assert result[0]["reps"] == 10
    with pytest.raises(RoutineError) as error:
        day_exercises('{"days":[]}', 2)
    assert error.value.kind == "missing"
    with pytest.raises(Exception) as http_error:
        workout_router._day_exercises("invalid-json", 1)
    assert http_error.value.status_code == 500


def test_http_gateway_injected_transport_does_not_retry():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                503, json={"detail": "sensitive internal response"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            gateway = HTTPAIGateway(
                "http://ai.invalid", "test-internal-key", client
            )
            with pytest.raises(GatewayError) as error:
                await gateway.request("POST", "/internal/sessions/1/start")
            assert "sensitive" not in str(error.value)
            assert len(requests) == 1
            assert (
                requests[0].headers["X-Internal-API-Key"] == "test-internal-key"
            )
            with pytest.raises(GatewayError) as missing:
                await HTTPAIGateway("", "", client).request(
                    "GET", "/pose/latest"
                )
            assert missing.value.unavailable
            assert len(requests) == 1

    asyncio.run(scenario())


def test_workout_start_compensates_when_ai_exercise_setup_fails(
    database, monkeypatch
):
    import json
    from datetime import date

    from fastapi import HTTPException

    from Server.AIFitService.services.token_service import get_current_user

    day = date.today().isoweekday()
    plan = json.dumps(
        {
            "days": [
                {
                    "day_no": day,
                    "exercises": [
                        {
                            "exercise_name": "dumbbell_shoulder_press",
                            "sets": 3,
                            "reps": 10,
                        }
                    ],
                }
            ]
        }
    )
    with database.get_connection() as conn:
        conn.execute(
            """INSERT INTO UserInfo
            (user_id,username,password_hash,email_address,nickname,gender,age,height,weight)
            VALUES (1,'workout-test','hash','workout@example.com','test','male',25,175,70)"""
        )
        conn.execute(
            """INSERT INTO Routine
            (routine_id,routine_name,target_body_part,routine_plan_json)
            VALUES (1,'test','shoulder',?)""",
            (plan,),
        )
        conn.execute("""INSERT INTO UserBodyInfo
            (user_id,routine_id,front_photo_ref,back_photo_ref,right_photo_ref,left_photo_ref,analysis_status)
            VALUES (1,1,'f','b','r','l','completed')""")
    calls = []

    async def ai_request(method, path, payload=None):
        calls.append((method, path, payload))
        if "/exercises/" in path:
            raise HTTPException(502, "simulated AI setup failure")
        return {"success": True}

    monkeypatch.setattr(workout_router, "_ai_request", ai_request)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": 1}
    try:
        with TestClient(app) as client:
            response = client.post("/api/workout-sessions")
            assert response.status_code == 502, response.text
    finally:
        app.dependency_overrides.clear()
    assert len(calls) == 3
    assert calls[-1][1].endswith("/finish")
    with database.get_connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM ExerciseSession").fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM Exercise").fetchone()[0] == 0
