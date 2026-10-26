"""진단 로그 어댑터가 앱 수명 밖에 필터나 파일 핸들러를 남기지 않는지 검사합니다."""

import json
import logging

from app.debug_logging import debug_log_files


def test_debug_handler_is_local_restored_and_does_not_duplicate(tmp_path):
    logger = logging.getLogger("uvicorn.error")
    previous = list(logger.filters)
    record = logging.LogRecord(
        "uvicorn.error",
        logging.INFO,
        "",
        0,
        '[ShoulderPressDebug] {"event":"test"}',
        (),
        None,
    )
    path = tmp_path / "debug.jsonl"
    with debug_log_files(path):
        with debug_log_files(path):
            logger.handle(record)
        logger.handle(record)
    assert logger.filters == previous
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["data"]["event"] == "test"
