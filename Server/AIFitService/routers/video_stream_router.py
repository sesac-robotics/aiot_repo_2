"""내부 키로 수신한 영상 또는 AI 스트림 영상을 앱별 최신 프레임 버퍼에 게시합니다. GUI에는 JWT로 보호된 MJPEG 스트림을 제공합니다.

Receive JPEG frames from AI Server and expose them as an MJPEG stream."""

import asyncio
import hmac
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from Server.AIFitService.adapters.frame_buffer import LatestFrameBuffer
from Server.AIFitService.config.settings import (
    AI_SERVER_API_KEY,
    AI_SERVER_BASE_URL,
    AI_SERVER_STREAM_PATH,
    VIDEO_INGEST_API_KEY,
)
from Server.AIFitService.services.token_service import get_current_user
from shared.mjpeg import JPEGFrameParser

router = APIRouter(prefix="/api/video", tags=["Video Stream"])

MAX_FRAME_BYTES = 8 * 1024 * 1024
logger = logging.getLogger("aifit.video")


async def receive_ai_stream(buffer: LatestFrameBuffer) -> None:
    """Pull JPEG frames directly from AI Server and publish them to GUI streams."""
    if not AI_SERVER_BASE_URL or not AI_SERVER_API_KEY:
        return

    stream_url = urljoin(
        AI_SERVER_BASE_URL + "/",
        AI_SERVER_STREAM_PATH.lstrip("/"),
    )
    headers = {
        "X-Internal-API-Key": AI_SERVER_API_KEY,
        "Accept": "multipart/x-mixed-replace",
    }
    timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                logger.info(
                    "Connecting to AI Server video stream: %s", stream_url
                )
                async with client.stream(
                    "GET", stream_url, headers=headers
                ) as response:
                    response.raise_for_status()
                    logger.info("AI Server video stream connected")
                    parser = JPEGFrameParser(MAX_FRAME_BYTES)

                    async for chunk in response.aiter_bytes(
                        chunk_size=16 * 1024
                    ):
                        if not chunk:
                            continue
                        for frame in parser.feed(chunk):
                            await buffer.publish(frame)

            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "AI Server stream returned HTTP %s; reconnecting",
                    exc.response.status_code,
                )
            except httpx.RequestError as exc:
                logger.warning("AI Server stream connection failed: %s", exc)

            await asyncio.sleep(2)


@router.post("/frame", status_code=202)
async def receive_frame(
    request: Request,
    x_internal_api_key: str | None = Header(default=None),
):
    """Accept one JPEG frame as the raw HTTP request body from AI Server."""
    if not VIDEO_INGEST_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AIFIT_VIDEO_API_KEY is not configured on AIFit Server.",
        )
    if not x_internal_api_key or not hmac.compare_digest(
        x_internal_api_key, VIDEO_INGEST_API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FRAME_BYTES:
                raise HTTPException(
                    status_code=413, detail="Frame is too large."
                )
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid Content-Length."
            )

    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].lower()
    )
    if content_type not in {"image/jpeg", "application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="Send JPEG bytes with image/jpeg or application/octet-stream.",
        )

    frame = await request.body()
    if not frame:
        raise HTTPException(status_code=400, detail="Frame body is empty.")
    if len(frame) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame is too large.")
    if not (frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")):
        raise HTTPException(
            status_code=400, detail="Request body is not a complete JPEG frame."
        )

    sequence = await request.app.state.frame_buffer.publish(frame)

    return {"accepted": True, "sequence": sequence}


@router.get("/stream")
async def stream_frames(
    request: Request, current_user: dict = Depends(get_current_user)
):
    """Stream the newest received JPEG frames to GUI clients as MJPEG."""

    async def frame_generator():
        sequence = 0
        while True:
            sequence, frame = await request.app.state.frame_buffer.next_frame(
                sequence
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status")
async def video_status(request: Request):
    """현재 앱에서 영상을 수신했는지 보고합니다. 다른 앱 실행과 상태를 공유하지 않습니다."""
    return await request.app.state.frame_buffer.status()
