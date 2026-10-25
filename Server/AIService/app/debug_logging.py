"""운동 진단 로그만 JSONL 파일로 분리합니다. 기존 디버그 플래그/판정은 유지합니다."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading


DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / 'logs' / 'debug.jsonl'
DEBUG_TAGS = ('SquatDebug', 'ShoulderPressDebug', 'PushUpDebug', 'YTWDebug', 'PlankDebug')


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        # 기존 JSON 본문을 중첩 객체로 보존하여 관절/이벤트 필드를 바로 조회할 수 있습니다.
        message = record.getMessage()
        tag, _, body = message.partition(']')
        try:
            data = json.loads(body.strip(), parse_constant=lambda _: None)
        except (ValueError, TypeError):
            data = {'message': body.strip()}
        return json.dumps({
            'logged_at': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'debug_type': tag.lstrip('['),
            'level': record.levelname,
            'data': data,
        }, ensure_ascii=False, allow_nan=False)


class _DebugFileFilter(logging.Filter):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def filter(self, record):
        # logger 단계에서 분리하므로 Uvicorn/root의 터미널 핸들러로 전파되지 않습니다.
        if record.getMessage().startswith(tuple('[' + tag + ']' for tag in DEBUG_TAGS)):
            self.handler.handle(record)
            return False
        return True


_lock = threading.RLock()
_users = 0
_filter = None


@contextmanager
def debug_log_files():
    """서버 수명 동안만 분리하고, 워커 종료 후 flush/close 및 기존 logger 복원을 수행합니다."""
    global _users, _filter
    logger = logging.getLogger('uvicorn.error')
    with _lock:
        if _users == 0:
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            # 파일당 10 MiB, 백업 5개. DEBUG가 꺼져 있으면 빈 파일도 만들지 않습니다.
            handler = RotatingFileHandler(
                DEBUG_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5,
                encoding='utf-8', delay=True,
            )
            handler.setFormatter(_JsonFormatter())
            _filter = _DebugFileFilter(handler)
            logger.addFilter(_filter)
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0:
                logger.removeFilter(_filter)
                _filter.handler.close()
                _filter = None
