"""운동 진단 로그만 JSONL 파일로 분리합니다. 기존 디버그 플래그/판정은 유지합니다."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "debug.jsonl"
DEBUG_TAGS = (
    "SquatDebug",
    "ShoulderPressDebug",
    "PushUpDebug",
    "YTWDebug",
    "PlankDebug",
)


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        # 기존 JSON 본문을 중첩 객체로 보존하여 관절/이벤트 필드를 바로 조회할 수 있습니다.
        message = record.getMessage()
        tag, _, body = message.partition("]")
        try:
            data = json.loads(body.strip(), parse_constant=lambda _: None)
        except (ValueError, TypeError):
            data = {"message": body.strip()}
        return json.dumps(
            {
                "logged_at": datetime.fromtimestamp(
                    record.created, timezone.utc
                ).isoformat(),
                "debug_type": tag.lstrip("["),
                "level": record.levelname,
                "data": data,
            },
            ensure_ascii=False,
            allow_nan=False,
        )


class _DebugFileFilter(logging.Filter):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def filter(self, record):
        # logger 단계에서 분리하므로 Uvicorn/root의 터미널 핸들러로 전파되지 않습니다.
        if record.getMessage().startswith(
            tuple("[" + tag + "]" for tag in DEBUG_TAGS)
        ):
            self.handler.handle(record)
            return False
        return True


@contextmanager
def debug_log_files(log_path: Path = DEBUG_LOG_PATH):
    """앱 수명에 귀속된 필터·파일 핸들러를 만들고 종료 시 해제합니다.

    전역 사용자 수와 필터 포인터를 공유하지 않습니다. 중첩 수명에서는 먼저
    설치한 필터가 진단 로그를 처리하며, 각 범위는 자신이 만든 자원만 회수합니다.
    """
    logger = logging.getLogger("uvicorn.error")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(_JsonFormatter())
    debug_filter = _DebugFileFilter(handler)
    logger.addFilter(debug_filter)
    try:
        yield
    finally:
        logger.removeFilter(debug_filter)
        handler.close()
