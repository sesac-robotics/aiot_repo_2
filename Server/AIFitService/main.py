import uvicorn
import asyncio
import logging
from fastapi import FastAPI

from config.settings import (
    AI_SERVER_API_KEY,
    AI_SERVER_BASE_URL,
    APP_HOST,
    APP_PORT,
)
from database.connection import init_db
from routers.auth_router import router as auth_router
from routers.profile_router import router as profile_router
from routers.body_analysis_router import (
    router as body_analysis_router,
    init_body_analysis_db,
)
from routers.exercise_report_router import router as exercise_report_router
from routers.workout_router import (
    router as workout_router,
    init_workout_db,
    workout_session_watchdog,
)
from routers.video_stream_router import (
    receive_ai_stream,
    router as video_stream_router,
)

app = FastAPI(title="AIFit Server", version="1.1.0")
logger = logging.getLogger("aifit.gui_api")


@app.middleware("http")
async def log_gui_api_result(request, call_next):
    """Log API callers and outcomes without logging payloads or credentials."""
    path = request.url.path
    if not (path.startswith("/api/") or path.startswith("/body-analysis") or path.startswith("/exercise-report")):
        return await call_next(request)

    caller = "AI_SERVER" if path.endswith("/records") or path == "/api/video/frame" else "GUI"
    logger.info("[%s][REQUEST] method=%s path=%s", caller, request.method, path)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            "[%s][ERROR] method=%s path=%s type=unhandled_exception error=%s",
            caller, request.method, path, type(exc).__name__,
        )
        raise

    status = response.status_code
    if status >= 500:
        logger.error("[%s][ERROR] method=%s path=%s type=server_error status=%s", caller, request.method, path, status)
    elif status >= 400:
        logger.warning("[%s][ERROR] method=%s path=%s type=client_error status=%s", caller, request.method, path, status)
    else:
        logger.info("[%s][SUCCESS] method=%s path=%s status=%s", caller, request.method, path, status)
    return response

app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(body_analysis_router)
app.include_router(exercise_report_router)
app.include_router(workout_router)
app.include_router(video_stream_router)


@app.on_event("startup")
def on_startup():
    init_db()
    init_body_analysis_db()
    init_workout_db()


@app.on_event("startup")
async def start_ai_video_receiver():
    if AI_SERVER_BASE_URL and AI_SERVER_API_KEY:
        app.state.ai_video_receiver_task = asyncio.create_task(receive_ai_stream())
    else:
        print(
            "[VIDEO] AI_SERVER_BASE_URL 또는 AI_SERVER_API_KEY가 없어 "
            "AI Server 스트림 연결을 시작하지 않았습니다."
        )


@app.on_event("startup")
async def start_workout_session_watchdog():
    app.state.workout_session_watchdog_task = asyncio.create_task(
        workout_session_watchdog()
    )


@app.on_event("shutdown")
async def stop_ai_video_receiver():
    task = getattr(app.state, "ai_video_receiver_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.on_event("shutdown")
async def stop_workout_session_watchdog():
    task = getattr(app.state, "workout_session_watchdog_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.get("/health")
def health():
    return {"success": True, "message": "AIFit Server is running"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )
