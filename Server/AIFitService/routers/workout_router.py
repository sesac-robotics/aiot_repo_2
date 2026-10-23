"""Workout session orchestration between the GUI, AIFit DB, and AI Server."""

import asyncio
import json
import hmac
import logging
from datetime import date, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from config.settings import AI_SERVER_API_KEY, AI_SERVER_BASE_URL
from database.connection import get_connection
from Server.AIFitService.services.exercise_names import canonical_exercise_name
from Server.AIFitService.services.token_service import get_current_user


router = APIRouter(prefix="/api/workout-sessions", tags=["Workout Sessions"])
logger = logging.getLogger("aifit.workout")
AI_TIMEOUT = 10.0


class ExerciseRecordInput(BaseModel):
    exercise_id: int = Field(gt=0)
    set_no: int = Field(ge=1)
    rep_no: int = Field(ge=1)
    pose_state: str = Field(min_length=1, max_length=100)
    score: float | None = None
    error_codes: list[str] | None = None
    feedback_code: str | None = Field(default=None, max_length=100)
    feedback_text: str | None = Field(default=None, max_length=2000)


def init_workout_db() -> None:
    """Add persisted orchestration state needed for set-by-set control."""
    with get_connection() as conn:
        exercise_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(Exercise)")
        }
        if "order_no" not in exercise_columns:
            conn.execute("ALTER TABLE Exercise ADD COLUMN order_no INTEGER")
        if "target_duration_sec" not in exercise_columns:
            conn.execute("ALTER TABLE Exercise ADD COLUMN target_duration_sec REAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ExerciseSessionRuntime (
                session_id INTEGER PRIMARY KEY,
                current_exercise_id INTEGER NOT NULL,
                current_set_no INTEGER NOT NULL DEFAULT 1,
                awaiting_start INTEGER NOT NULL DEFAULT 0,
                set_pause INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES ExerciseSession(session_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (current_exercise_id) REFERENCES Exercise(exercise_id)
                    ON DELETE CASCADE
            )
        """)


def _ai_url(path: str) -> str:
    if not AI_SERVER_BASE_URL or not AI_SERVER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI Server 연결 설정이 없습니다.",
        )
    return f"{AI_SERVER_BASE_URL}/{path.lstrip('/')}"


async def _ai_request(method: str, path: str, payload: dict | None = None) -> Any:
    url = _ai_url(path)
    headers = {"X-Internal-API-Key": AI_SERVER_API_KEY}
    logger.info("[AI_SERVER][REQUEST] method=%s path=%s", method, path)
    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            response = await client.request(method, url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.error(
            "[AI_SERVER][ERROR] method=%s path=%s type=connection_error error=%s",
            method, path, type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="AI Server에 연결할 수 없습니다.")

    if response.is_error:
        # Avoid logging response content; it may contain sensitive internal data.
        logger.error(
            "[AI_SERVER][ERROR] method=%s path=%s type=http_error status=%s",
            method, path, response.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail=f"AI Server 요청이 실패했습니다 (HTTP {response.status_code}).",
        )
    logger.info(
        "[AI_SERVER][SUCCESS] method=%s path=%s status=%s",
        method, path, response.status_code,
    )
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


async def _ai_optional_get(path: str) -> Any:
    """Fetch pose/metrics data without failing the live session if unavailable."""
    try:
        return await _ai_request("GET", path)
    except HTTPException as exc:
        logger.info("[AI_SERVER][OPTIONAL_UNAVAILABLE] path=%s status=%s", path, exc.status_code)
        return None


def _day_exercises(plan_json: str, day_no: int) -> list[dict]:
    try:
        plan = json.loads(plan_json)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="저장된 루틴 JSON 형식이 잘못되었습니다.")

    day_plan = None
    if isinstance(plan, dict):
        days = plan.get("days")
        if isinstance(days, list):
            day_plan = next(
                (item for item in days if isinstance(item, dict) and item.get("day_no") == day_no),
                None,
            )
        elif isinstance(days, dict):
            day_plan = days.get(str(day_no)) or days.get(f"day_{day_no}")
        if day_plan is None and plan.get("day_no") == day_no:
            day_plan = plan
        if day_plan is None:
            day_plan = plan.get(f"day_{day_no}") or plan.get(f"day{day_no}")
    elif isinstance(plan, list):
        day_plan = next(
            (item for item in plan if isinstance(item, dict) and item.get("day_no") == day_no),
            None,
        )

    exercises = day_plan.get("exercises") if isinstance(day_plan, dict) else None
    if not isinstance(exercises, list) or not exercises:
        raise HTTPException(
            status_code=404,
            detail=f"선택한 7일 루틴에 Day {day_no} 운동이 없습니다.",
        )

    normalized = []
    for index, item in enumerate(exercises, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=500, detail="루틴 운동 항목 형식이 잘못되었습니다.")
        name = canonical_exercise_name(item.get("exercise_name"))
        sets = item.get("sets")
        reps = item.get("reps")
        duration = item.get("target_duration_sec", item.get("duration_sec"))
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=500, detail="루틴에 운동 이름이 없습니다.")
        if not isinstance(sets, int) or sets < 1:
            raise HTTPException(status_code=500, detail=f"{name}: 목표 세트 수가 올바르지 않습니다.")
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                raise HTTPException(status_code=500, detail=f"{name}: 목표 유지 시간이 올바르지 않습니다.")
            if duration <= 0:
                raise HTTPException(status_code=500, detail=f"{name}: 목표 유지 시간은 0보다 커야 합니다.")
            reps = 1  # DB schema requires reps; duration is the AI Server target.
        elif not isinstance(reps, int) or reps < 1:
            raise HTTPException(status_code=500, detail=f"{name}: 목표 반복 횟수가 올바르지 않습니다.")
        order_no = item.get("order_no", index)
        if not isinstance(order_no, int) or order_no < 1:
            raise HTTPException(status_code=500, detail=f"{name}: 운동 순서가 올바르지 않습니다.")
        normalized.append({
            "exercise_name": name.strip(),
            "sets": sets,
            "reps": reps,
            "target_duration_sec": duration,
            "order_no": order_no,
        })
    return normalized


def _get_owned_session(session_id: int, user_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM ExerciseSession WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="운동 세션을 찾을 수 없습니다.")
    return dict(row)


def _current_workout(session_id: int):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT runtime.current_exercise_id, runtime.current_set_no,
                   runtime.awaiting_start, runtime.set_pause,
                   exercise.exercise_name, exercise.sets, exercise.reps,
                   exercise.order_no, exercise.target_duration_sec
            FROM ExerciseSessionRuntime AS runtime
            JOIN Exercise AS exercise
              ON exercise.exercise_id = runtime.current_exercise_id
            WHERE runtime.session_id = ?
        """, (session_id,)).fetchone()
    return dict(row) if row else None


def _ai_exercise_payload(exercise: dict) -> dict:
    return {
        "exercise_name": canonical_exercise_name(exercise["exercise_name"]),
        "order_no": exercise["order_no"],
        "target_reps": None if exercise["target_duration_sec"] is not None else exercise["reps"],
        "target_duration_sec": exercise["target_duration_sec"],
    }


async def _start_ai_exercise(session_id: int, exercise_id: int, exercise: dict):
    return await _ai_request(
        "POST",
        f"/internal/sessions/{session_id}/exercises/{exercise_id}/start",
        _ai_exercise_payload(exercise),
    )


@router.post("", status_code=201)
async def start_workout(current_user: dict = Depends(get_current_user)):
    """Create today's DB session and start its first routine exercise on AI Server."""
    user_id = current_user["user_id"]
    day_no = date.today().isoweekday()  # Monday=1 ... Sunday=7, matching Routine Day 1-7.

    with get_connection() as conn:
        active = conn.execute("""
            SELECT session_id FROM ExerciseSession
            WHERE user_id = ? AND session_status IN ('IN_PROGRESS', 'PAUSED')
            LIMIT 1
        """, (user_id,)).fetchone()
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"이미 진행 중인 운동 세션이 있습니다 (session_id={active['session_id']}).",
            )

        body = conn.execute("""
            SELECT routine.routine_id, routine.routine_plan_json
            FROM UserBodyInfo AS body
            JOIN Routine AS routine ON routine.routine_id = body.routine_id
            WHERE body.user_id = ? AND body.routine_id IS NOT NULL
            ORDER BY body.created_at DESC, body.body_id DESC
            LIMIT 1
        """, (user_id,)).fetchone()
        if body is None:
            raise HTTPException(status_code=409, detail="사용자에게 연결된 7일 운동 루틴이 없습니다.")

        exercises = _day_exercises(body["routine_plan_json"], day_no)
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        cursor = conn.execute("""
            INSERT INTO ExerciseSession (user_id, routine_id, day_no, started_at, session_status)
            VALUES (?, ?, ?, ?, 'IN_PROGRESS')
        """, (user_id, body["routine_id"], day_no, now))
        session_id = cursor.lastrowid

        inserted_exercises = []
        for item in exercises:
            cursor = conn.execute("""
                INSERT INTO Exercise (
                    session_id, exercise_name, sets, reps, order_no, target_duration_sec
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                item["exercise_name"],
                item["sets"],
                item["reps"],
                item["order_no"],
                item["target_duration_sec"],
            ))
            inserted_exercises.append({**item, "exercise_id": cursor.lastrowid})

        first = inserted_exercises[0]
        conn.execute("""
            INSERT INTO ExerciseSessionRuntime (
                session_id, current_exercise_id, current_set_no, awaiting_start, set_pause
            ) VALUES (?, ?, 1, 0, 0)
        """, (session_id, first["exercise_id"]))

    try:
        ai_session = await _ai_request("POST", f"/internal/sessions/{session_id}/start")
        ai_exercise = await _start_ai_exercise(session_id, first["exercise_id"], first)
    except HTTPException:
        # If session creation succeeded but exercise setup failed, best-effort
        # cleanup prevents the AI Server from retaining an orphan session.
        try:
            await _ai_request("POST", f"/internal/sessions/{session_id}/finish")
        except HTTPException:
            logger.warning("Could not clean up AI Server session %s", session_id)
        # Do not leave an AIFit session behind if AI Server could not start it.
        with get_connection() as conn:
            conn.execute("DELETE FROM ExerciseSession WHERE session_id = ?", (session_id,))
        raise

    return {
        "session_id": session_id,
        "routine_id": body["routine_id"],
        "day_no": day_no,
        "session_status": "IN_PROGRESS",
        "current_set_no": 1,
        "exercise": first,
        "exercises": inserted_exercises,
        "ai_server": {"session": ai_session, "exercise": ai_exercise},
    }


@router.post("/{session_id}/pause")
async def pause_workout(session_id: int, current_user: dict = Depends(get_current_user)):
    session = _get_owned_session(session_id, current_user["user_id"])
    if session["session_status"] != "IN_PROGRESS":
        raise HTTPException(status_code=409, detail="진행 중인 세션만 일시정지할 수 있습니다.")
    ai_result = await _ai_request("POST", f"/internal/sessions/{session_id}/pause")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            UPDATE ExerciseSession
            SET session_status='PAUSED', paused_at=?
            WHERE session_id=? AND user_id=? AND session_status='IN_PROGRESS'
        """, (now, session_id, current_user["user_id"]))
        conn.execute(
            "UPDATE ExerciseSessionRuntime SET awaiting_start=0, set_pause=0 WHERE session_id=?",
            (session_id,),
        )
    return {"session_id": session_id, "session_status": "PAUSED", "ai_server": ai_result}


@router.post("/{session_id}/start")
async def resume_or_advance_workout(
    session_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Resume manual pauses, or start the next set/exercise after an auto-pause."""
    user_id = current_user["user_id"]
    session = _get_owned_session(session_id, user_id)
    if session["session_status"] != "PAUSED":
        raise HTTPException(status_code=409, detail="일시정지된 세션만 Start로 재개할 수 있습니다.")
    runtime = _current_workout(session_id)
    if runtime is None:
        raise HTTPException(status_code=409, detail="세션의 현재 운동 상태를 찾을 수 없습니다.")

    next_exercise = None
    next_set_no = runtime["current_set_no"]
    if runtime["set_pause"]:
        if runtime["current_set_no"] < runtime["sets"]:
            next_set_no += 1
        else:
            with get_connection() as conn:
                next_exercise = conn.execute("""
                    SELECT * FROM Exercise
                    WHERE session_id = ? AND order_no > ?
                    ORDER BY order_no LIMIT 1
                """, (session_id, runtime["order_no"])).fetchone()
            if next_exercise is None:
                raise HTTPException(
                    status_code=409,
                    detail="오늘 루틴의 모든 운동이 완료되었습니다. 세션 종료 요청이 필요합니다.",
                )
            next_exercise = dict(next_exercise)
            next_set_no = 1

    # AI's documented resume endpoint reopens the paused session; starting the
    # exercise again at a completed-set boundary resets its rep/hold counter.
    ai_session = await _ai_request("POST", f"/internal/sessions/{session_id}/resume")
    ai_exercise = None
    if runtime["set_pause"]:
        if next_exercise is None:
            with get_connection() as conn:
                next_exercise = conn.execute(
                    "SELECT * FROM Exercise WHERE exercise_id=?", (runtime["current_exercise_id"],)
                ).fetchone()
            next_exercise = dict(next_exercise)
        try:
            ai_exercise = await _start_ai_exercise(
                session_id, next_exercise["exercise_id"], next_exercise
            )
        except HTTPException:
            try:
                await _ai_request("POST", f"/internal/sessions/{session_id}/pause")
            except HTTPException:
                logger.exception("Could not restore AI Server pause after exercise start failed")
            raise

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            UPDATE ExerciseSession
            SET session_status='IN_PROGRESS',
                paused_duration=paused_duration + CASE
                    WHEN paused_at IS NOT NULL THEN MAX(0, CAST((julianday(?) - julianday(paused_at)) * 86400 AS INTEGER))
                    ELSE 0 END,
                paused_at=NULL
            WHERE session_id=? AND user_id=?
        """, (now, session_id, user_id))
        if runtime["set_pause"]:
            conn.execute("""
                UPDATE ExerciseSessionRuntime
                SET current_exercise_id=?, current_set_no=?, awaiting_start=0, set_pause=0
                WHERE session_id=?
            """, (next_exercise["exercise_id"], next_set_no, session_id))

    return {
        "session_id": session_id,
        "session_status": "IN_PROGRESS",
        "current_set_no": next_set_no,
        "exercise_id": next_exercise["exercise_id"] if next_exercise else runtime["current_exercise_id"],
        "ai_server": {"session": ai_session, "exercise": ai_exercise},
    }


@router.get("/{session_id}/live")
async def get_live_workout(session_id: int, current_user: dict = Depends(get_current_user)):
    """Proxy AI Server's live session snapshot and pause when this set reaches its goal."""
    session = _get_owned_session(session_id, current_user["user_id"])
    runtime = _current_workout(session_id)
    if runtime is None:
        raise HTTPException(status_code=409, detail="세션의 현재 운동 상태를 찾을 수 없습니다.")
    if session["session_status"] == "COMPLETED":
        return {
            "session_id": session_id,
            "session_status": "COMPLETED",
            "current_exercise_id": runtime["current_exercise_id"],
            "current_set_no": runtime["current_set_no"],
            "awaiting_start": False,
            "session_complete": True,
            "ai_server": None,
        }
    snapshot, latest_pose, metrics = await asyncio.gather(
        _ai_request("GET", f"/internal/sessions/{session_id}/live"),
        _ai_optional_get("/pose/latest"),
        _ai_optional_get("/metrics"),
    )

    live_exercise = snapshot.get("exercise") if isinstance(snapshot, dict) else None
    if isinstance(live_exercise, dict) and not runtime["awaiting_start"]:
        rep_count = live_exercise.get("rep_count")
        hold_seconds = live_exercise.get("current_hold_sec")
        reps_done = (
            isinstance(rep_count, (int, float))
            and runtime["target_duration_sec"] is None
            and rep_count >= runtime["reps"]
        )
        hold_done = (
            isinstance(hold_seconds, (int, float))
            and runtime["target_duration_sec"] is not None
            and hold_seconds >= runtime["target_duration_sec"]
        )
        if reps_done or hold_done:
            final_set = runtime["current_set_no"] >= runtime["sets"]
            with get_connection() as conn:
                following = conn.execute(
                    "SELECT 1 FROM Exercise WHERE session_id=? AND order_no>? LIMIT 1",
                    (session_id, runtime["order_no"]),
                ).fetchone() if final_set else True
            final_exercise = final_set and following is None

            if final_exercise:
                await _ai_request(
                    "POST",
                    f"/internal/sessions/{session_id}/exercises/{runtime['current_exercise_id']}/finish",
                )
                await _ai_request("POST", f"/internal/sessions/{session_id}/finish")
                now = datetime.now().astimezone().isoformat(timespec="seconds")
                with get_connection() as conn:
                    conn.execute("""
                        UPDATE ExerciseSession SET session_status='COMPLETED', ended_at=?
                        WHERE session_id=? AND session_status='IN_PROGRESS'
                    """, (now, session_id))
                session["session_status"] = "COMPLETED"
            else:
                await _ai_request("POST", f"/internal/sessions/{session_id}/pause")
                now = datetime.now().astimezone().isoformat(timespec="seconds")
                with get_connection() as conn:
                    conn.execute("""
                        UPDATE ExerciseSession SET session_status='PAUSED', paused_at=?
                        WHERE session_id=? AND session_status='IN_PROGRESS'
                    """, (now, session_id))
                    conn.execute("""
                        UPDATE ExerciseSessionRuntime
                        SET awaiting_start=1, set_pause=1
                        WHERE session_id=?
                    """, (session_id,))
                runtime["awaiting_start"] = 1
                runtime["set_pause"] = 1

    with get_connection() as conn:
        latest_record = conn.execute("""
            SELECT set_no, rep_no, pose_state, score, error_codes, feedback_code, feedback_text
            FROM ExerciseRecord
            WHERE exercise_id=? AND set_no=?
            ORDER BY rep_no DESC LIMIT 1
        """, (runtime["current_exercise_id"], runtime["current_set_no"])).fetchone()
    record_payload = dict(latest_record) if latest_record else None
    if record_payload and record_payload["error_codes"]:
        try:
            record_payload["error_codes"] = json.loads(record_payload["error_codes"])
        except json.JSONDecodeError:
            record_payload["error_codes"] = []

    return {
        "session_id": session_id,
        "session_status": "PAUSED" if runtime["awaiting_start"] else session["session_status"],
        "current_exercise_id": runtime["current_exercise_id"],
        "current_set_no": runtime["current_set_no"],
        "target_sets": runtime["sets"],
        "target_reps": None if runtime["target_duration_sec"] is not None else runtime["reps"],
        "target_duration_sec": runtime["target_duration_sec"],
        "awaiting_start": bool(runtime["awaiting_start"]),
        "session_complete": session["session_status"] == "COMPLETED",
        "latest_record": record_payload,
        "ai_server": snapshot,
        "pose": latest_pose,
        "metrics": metrics,
    }


@router.post("/{session_id}/records", status_code=201)
def receive_exercise_record(
    session_id: int,
    payload: ExerciseRecordInput,
    x_internal_api_key: str | None = Header(default=None),
):
    """Persist one AI-evaluated repetition; GUI reads it through the live endpoint."""
    if not AI_SERVER_API_KEY:
        raise HTTPException(status_code=503, detail="AI Server 인증 키가 설정되지 않았습니다.")
    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, AI_SERVER_API_KEY):
        raise HTTPException(status_code=401, detail="내부 인증 키가 올바르지 않습니다.")

    error_codes = json.dumps(payload.error_codes, ensure_ascii=False) if payload.error_codes is not None else None
    with get_connection() as conn:
        exercise = conn.execute(
            "SELECT sets, reps, target_duration_sec FROM Exercise WHERE exercise_id=? AND session_id=?",
            (payload.exercise_id, session_id),
        ).fetchone()
        if exercise is None:
            raise HTTPException(status_code=404, detail="세션에 속한 운동을 찾을 수 없습니다.")
        if payload.set_no > exercise["sets"]:
            raise HTTPException(status_code=422, detail="목표 세트 수를 초과한 결과입니다.")
        if exercise["target_duration_sec"] is None and payload.rep_no > exercise["reps"]:
            raise HTTPException(status_code=422, detail="목표 반복 횟수를 초과한 결과입니다.")
        conn.execute("""
            INSERT INTO ExerciseRecord (
                exercise_id, set_no, rep_no, pose_state, score,
                error_codes, feedback_code, feedback_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exercise_id, set_no, rep_no) DO UPDATE SET
                pose_state=excluded.pose_state,
                score=excluded.score,
                error_codes=excluded.error_codes,
                feedback_code=excluded.feedback_code,
                feedback_text=excluded.feedback_text
        """, (
            payload.exercise_id, payload.set_no, payload.rep_no, payload.pose_state,
            payload.score, error_codes, payload.feedback_code, payload.feedback_text,
        ))

    return {
        "accepted": True,
        "session_id": session_id,
        "exercise_id": payload.exercise_id,
        "set_no": payload.set_no,
        "rep_no": payload.rep_no,
    }


@router.post("/{session_id}/finish")
async def finish_workout(session_id: int, current_user: dict = Depends(get_current_user)):
    session = _get_owned_session(session_id, current_user["user_id"])
    if session["session_status"] == "COMPLETED":
        return {
            "session_id": session_id,
            "session_status": "COMPLETED",
            "already_completed": True,
        }

    runtime = _current_workout(session_id)
    if runtime is None:
        raise HTTPException(status_code=409, detail="세션의 현재 운동 상태를 찾을 수 없습니다.")

    # AI Server requires the active exercise to be closed before its session.
    ai_exercise_result = await _ai_request(
        "POST",
        f"/internal/sessions/{session_id}/exercises/{runtime['current_exercise_id']}/finish",
    )
    ai_session_result = await _ai_request(
        "POST",
        f"/internal/sessions/{session_id}/finish",
    )
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            UPDATE ExerciseSession
            SET session_status='COMPLETED', ended_at=?,
                paused_duration=paused_duration + CASE
                    WHEN paused_at IS NOT NULL THEN MAX(0, CAST((julianday(?) - julianday(paused_at)) * 86400 AS INTEGER))
                    ELSE 0 END,
                paused_at=NULL
            WHERE session_id=? AND user_id=?
        """, (now, now, session_id, current_user["user_id"]))
    return {
        "session_id": session_id,
        "session_status": "COMPLETED",
        "ai_server": {
            "exercise": ai_exercise_result,
            "session": ai_session_result,
        },
    }
