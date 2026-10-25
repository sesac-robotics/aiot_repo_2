"""SQLite 연결의 트랜잭션과 수명을 함께 관리하는 영속성 어댑터입니다."""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from Server.AIFitService.config.settings import DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """외래 키를 적용한 연결을 제공하고 성공 시 커밋, 실패 시 롤백합니다.

    SQLite 자체 with 구문은 연결을 닫지 않으므로 finally에서 반드시 닫습니다.
    호출자는 with 블록 밖에서 cursor나 연결을 재사용하지 않아야 합니다.

    Yields:
        sqlite3.Connection: 행 이름으로 조회할 수 있는 단일 트랜잭션 연결입니다.

    Raises:
        sqlite3.Error: 연결·쿼리·커밋 실패를 호출자에게 전달합니다.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """UserInfo 스키마를 없을 때만 생성합니다. 기존 사용자 데이터는 보존합니다."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS UserInfo (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                email_address TEXT NOT NULL UNIQUE,
                nickname TEXT NOT NULL,
                gender TEXT NOT NULL,
                age INTEGER NOT NULL,
                height REAL NOT NULL,
                weight REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
    schema_path = Path(__file__).resolve().parents[2] / "AIFitDB" / "AIFit.sql"
    with get_connection() as conn:
        # executescript starts a transaction explicitly because SQLite otherwise
        # commits an outstanding transaction before running the script.
        conn.executescript(
            "BEGIN;\n" + schema_path.read_text(encoding="utf-8") + "\nCOMMIT;"
        )
    logging.getLogger(__name__).info("Base schema ready")
