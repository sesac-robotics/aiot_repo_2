"""사용자 조회·등록·수정·연관 데이터 삭제 SQL을 모읍니다. 매 호출은 연결 컨텍스트 안에서 실행하며 외래 키와 트랜잭션을 적용합니다."""

import sqlite3

from Server.AIFitService.database.connection import get_connection


def find_by_username(username: str) -> sqlite3.Row | None:
    """로그인 검증에 필요한 사용자 행을 반환합니다. 비밀번호 해시는 API 응답에 포함하지 않습니다."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT user_id, username, password_hash, nickname "
            "FROM UserInfo WHERE username = ?",
            (username,),
        ).fetchone()


def find_public_by_id(user_id: int) -> sqlite3.Row | None:
    """비밀번호 해시를 제외한 사용자 행을 반환합니다. 사용자가 없으면 None입니다."""
    with get_connection() as conn:
        return conn.execute(
            """SELECT user_id, username, email_address, nickname, gender,
                      age, height, weight, created_at
               FROM UserInfo WHERE user_id = ?""",
            (user_id,),
        ).fetchone()


def username_exists(username: str) -> bool:
    """아이디의 존재 여부를 매개변수 SQL로 검사합니다. 입력을 SQL에 직접 연결하지 않습니다."""
    with get_connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM UserInfo WHERE username = ?", (username,)
            ).fetchone()
            is not None
        )


def email_exists(email: str) -> bool:
    """이메일의 대소문자를 구분하지 않고 중복 여부를 검사합니다."""
    with get_connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM UserInfo WHERE lower(email_address) = lower(?)",
                (email,),
            ).fetchone()
            is not None
        )


def email_exists_for_other_user(email: str, user_id: int) -> bool:
    """프로필 변경 시 자기 계정을 제외한 이메일 중복을 검사합니다."""
    with get_connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM UserInfo WHERE lower(email_address) = lower(?) AND user_id != ?",
                (email, user_id),
            ).fetchone()
            is not None
        )


def create_user(
    username: str,
    password_hash: str,
    email_address: str,
    nickname: str,
    gender: str,
    age: int,
    height: float,
    weight: float,
) -> int | None:
    """검증된 사용자 정보와 비밀번호 해시를 저장하고 SQLite 삽입 ID를 반환합니다."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO UserInfo
            (username, password_hash, email_address, nickname, gender, age, height, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                username,
                password_hash,
                email_address,
                nickname,
                gender,
                age,
                height,
                weight,
            ),
        )
        return cursor.lastrowid


def get_password_hash(user_id: int) -> str | None:
    """비밀번호 변경·탈퇴 검증용 해시를 조회합니다. 사용자가 없으면 None입니다."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM UserInfo WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["password_hash"] if row else None


def update_profile(
    user_id: int,
    email_address: str,
    nickname: str,
    gender: str,
    age: int,
    height: float,
    weight: float,
) -> bool:
    """지정 사용자 프로필을 갱신하고 실제 변경 대상이 있으면 True를 반환합니다."""
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE UserInfo
               SET email_address = ?, nickname = ?, gender = ?, age = ?, height = ?, weight = ?
               WHERE user_id = ?""",
            (email_address, nickname, gender, age, height, weight, user_id),
        )
        return cursor.rowcount > 0


def update_password(user_id: int, password_hash: str) -> bool:
    """검증을 마친 새 비밀번호 해시를 저장합니다. 원문 비밀번호를 받지 않습니다."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE UserInfo SET password_hash = ? WHERE user_id = ?",
            (password_hash, user_id),
        )
        return cursor.rowcount > 0


def delete_user_and_related_data(user_id: int) -> bool:
    """Delete a user's records while retaining routines still used by others."""
    with get_connection() as conn:
        user = conn.execute(
            "SELECT 1 FROM UserInfo WHERE user_id = ?", (user_id,)
        ).fetchone()
        if user is None:
            return False

        routine_rows = conn.execute(
            """SELECT routine_id FROM UserBodyInfo WHERE user_id = ? AND routine_id IS NOT NULL
               UNION SELECT routine_id FROM BodyAnalysis WHERE user_id = ? AND routine_id IS NOT NULL""",
            (user_id, user_id),
        ).fetchall()
        routine_ids = [row["routine_id"] for row in routine_rows]

        session_rows = conn.execute(
            "SELECT session_id FROM ExerciseSession WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        session_ids = [row["session_id"] for row in session_rows]

        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            exercise_rows = conn.execute(
                f"SELECT exercise_id FROM Exercise WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()
            exercise_ids = [row["exercise_id"] for row in exercise_rows]
            if exercise_ids:
                exercise_placeholders = ",".join("?" for _ in exercise_ids)
                conn.execute(
                    f"DELETE FROM ExerciseRecord WHERE exercise_id IN ({exercise_placeholders})",
                    exercise_ids,
                )
                conn.execute(
                    f"DELETE FROM Exercise WHERE exercise_id IN ({exercise_placeholders})",
                    exercise_ids,
                )
            conn.execute(
                f"DELETE FROM ExerciseReport WHERE session_id IN ({placeholders})",
                session_ids,
            )
            conn.execute(
                f"DELETE FROM ExerciseSession WHERE session_id IN ({placeholders})",
                session_ids,
            )

        conn.execute("DELETE FROM BodyAnalysis WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM UserBodyInfo WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM UserInfo WHERE user_id = ?", (user_id,))

        for row in routine_ids:
            routine_id = row
            still_referenced = conn.execute(
                """SELECT 1 FROM UserBodyInfo WHERE routine_id = ?
                   UNION ALL SELECT 1 FROM BodyAnalysis WHERE routine_id = ?
                   UNION ALL SELECT 1 FROM ExerciseSession WHERE routine_id = ?
                   LIMIT 1""",
                (routine_id, routine_id, routine_id),
            ).fetchone()
            if still_referenced is None:
                conn.execute(
                    "DELETE FROM Routine WHERE routine_id = ?", (routine_id,)
                )

        return True
