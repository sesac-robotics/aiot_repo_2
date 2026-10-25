"""웹 라우터와 앱 수명을 조립하는 백엔드 진입점입니다. 설정 검증과 스키마 초기화 후 선택적 영상 수신 작업을 만들고 종료 시 취소·회수합니다."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from Server.AIFitService.adapters.frame_buffer import LatestFrameBuffer
from Server.AIFitService.config.settings import (
    AI_SERVER_API_KEY,
    AI_SERVER_BASE_URL,
    APP_HOST,
    APP_PORT,
    validate_settings,
)
from Server.AIFitService.database.connection import init_db
from Server.AIFitService.routers.auth_router import router as auth_router
from Server.AIFitService.routers.body_analysis_router import (
    init_body_analysis_db,
)
from Server.AIFitService.routers.body_analysis_router import (
    router as body_analysis_router,
)
from Server.AIFitService.routers.exercise_report_router import (
    router as exercise_report_router,
)
from Server.AIFitService.routers.profile_router import router as profile_router
from Server.AIFitService.routers.video_stream_router import (
    receive_ai_stream,
)
from Server.AIFitService.routers.video_stream_router import (
    router as video_stream_router,
)
from Server.AIFitService.routers.workout_router import init_workout_db
from Server.AIFitService.routers.workout_router import router as workout_router


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """설정 검증, 스키마 초기화, 수신 작업 종료를 하나의 수명으로 관리합니다."""
    validate_settings()
    init_db()
    init_body_analysis_db()
    init_workout_db()
    application.state.frame_buffer = LatestFrameBuffer()
    task = None
    if AI_SERVER_BASE_URL and AI_SERVER_API_KEY:
        task = asyncio.create_task(
            receive_ai_stream(application.state.frame_buffer)
        )
    application.state.ai_video_receiver_task = task
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Cancellation here is the expected shutdown result.
                pass


app = FastAPI(title="AIFit Server", version="1.1.0", lifespan=lifespan)
logger = logging.getLogger("aifit.gui_api")


@app.middleware("http")
async def log_gui_api_result(request, call_next):
    """Log API callers and outcomes without logging payloads or credentials."""
    path = request.url.path
    if not (
        path.startswith("/api/")
        or path.startswith("/body-analysis")
        or path.startswith("/exercise-report")
    ):
        return await call_next(request)

    caller = (
        "AI_SERVER"
        if path.endswith("/records") or path == "/api/video/frame"
        else "GUI"
    )
    logger.info("[%s][REQUEST] method=%s path=%s", caller, request.method, path)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            "[%s][ERROR] method=%s path=%s type=unhandled_exception error=%s",
            caller,
            request.method,
            path,
            type(exc).__name__,
        )
        raise

    status = response.status_code
    if status >= 500:
        logger.error(
            "[%s][ERROR] method=%s path=%s type=server_error status=%s",
            caller,
            request.method,
            path,
            status,
        )
    elif status >= 400:
        logger.warning(
            "[%s][ERROR] method=%s path=%s type=client_error status=%s",
            caller,
            request.method,
            path,
            status,
        )
    else:
        logger.info(
            "[%s][SUCCESS] method=%s path=%s status=%s",
            caller,
            request.method,
            path,
            status,
        )
    return response


app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(body_analysis_router)
app.include_router(exercise_report_router)
app.include_router(workout_router)
app.include_router(video_stream_router)


@app.get("/health")
def health():
    return {"success": True, "message": "AIFit Server is running"}


if __name__ == "__main__":
    uvicorn.run(
        "Server.AIFitService.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )
