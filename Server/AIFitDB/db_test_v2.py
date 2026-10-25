import sqlite3
import json
from pathlib import Path

# 이 파일과 같은 폴더에 테스트 DB를 생성합니다.
DB_PATH = Path(__file__).resolve().parent / "aifit_test.db"

REQUIRED_TABLES = {
    "UserInfo",
    "UserBodyInfo",
    "Routine",
    "ExerciseSession",
    "Exercise",
    "ExerciseRecord",
    "ExerciseReport",
}

def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")

def main():
    # 테스트 전용 DB를 새로 만들기 위해 기존 테스트 DB만 초기화합니다.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")
        check(conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1,
              "외래 키 검사 활성화")

        # aifit.sql이 이 파일과 같은 폴더에 있어야 합니다.
        schema_path = Path(__file__).resolve().parent / "AIFit.sql"
        if not schema_path.exists():
            raise FileNotFoundError(
                f"{schema_path} 파일을 찾을 수 없습니다. "
                "aifit.sql을 이 파이썬 파일과 같은 폴더에 두세요."
            )

        conn.executescript(schema_path.read_text(encoding="utf-8"))
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        check(REQUIRED_TABLES.issubset(tables), "필수 테이블 7개 생성")

        # 1. 사용자
        cur = conn.execute("""
            INSERT INTO UserInfo
                (username, password_hash, email_address, nickname,
                 gender, age, height, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "db_test_user", "test_hash", "db_test@example.com",
            "DB테스트", "M", 25, 175.0, 70.0
        ))
        user_id = cur.lastrowid
        check(conn.execute(
            "SELECT COUNT(*) FROM UserInfo WHERE user_id=?",
            (user_id,)
        ).fetchone()[0] == 1, "사용자 저장 및 조회")

        # 2. 공용 루틴
        routine_plan = {
            "1": [
                {"exercise_name": "Air Squat", "sets": 3, "reps": 12},
                {"exercise_name": "Deadlift", "sets": 3, "reps": 10}
            ]
        }
        cur = conn.execute("""
            INSERT INTO Routine
                (routine_name, description, target_body_part, routine_plan_json)
            VALUES (?, ?, ?, ?)
        """, (
            "DB 테스트 루틴", "테스트용 전신 루틴", "전신",
            json.dumps(routine_plan, ensure_ascii=False)
        ))
        routine_id = cur.lastrowid
        check(json.loads(conn.execute(
            "SELECT routine_plan_json FROM Routine WHERE routine_id=?",
            (routine_id,)
        ).fetchone()[0]) == routine_plan, "루틴 및 JSON 구성 저장·조회")

        # 3. 체형 분석
        cur = conn.execute("""
            INSERT INTO UserBodyInfo
                (user_id, routine_id, front_photo_ref, back_photo_ref,
                 side1_photo_ref, side2_photo_ref, chest_status,
                 shoulder_status, arm_status, back_status, leg_status,
                 analysis_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, routine_id,
            "photos/front.jpg", "photos/back.jpg",
            "photos/side1.jpg", "photos/side2.jpg",
            "정상", "보완 필요", "정상", "정상", "보완 필요",
            "COMPLETED"
        ))
        body_id = cur.lastrowid
        check(conn.execute(
            "SELECT COUNT(*) FROM UserBodyInfo WHERE body_id=? AND user_id=?",
            (body_id, user_id)
        ).fetchone()[0] == 1, "체형 분석 결과 저장 및 사용자 연결")

        # 4. 하루 세션
        cur = conn.execute("""
            INSERT INTO ExerciseSession
                (user_id, routine_id, day_no, started_at, session_status)
            VALUES (?, ?, ?, datetime('now', 'localtime'), 'IN_PROGRESS')
        """, (user_id, routine_id, 1))
        session_id = cur.lastrowid
        check(conn.execute(
            "SELECT COUNT(*) FROM ExerciseSession WHERE session_id=?",
            (session_id,)
        ).fetchone()[0] == 1, "운동 세션 저장")

        # 5. 세션 안의 개별 운동
        exercise_ids = []
        for name, sets, reps in [
            ("Air Squat", 3, 12),
            ("Deadlift", 3, 10),
        ]:
            cur = conn.execute("""
                INSERT INTO Exercise (session_id, exercise_name, sets, reps)
                VALUES (?, ?, ?, ?)
            """, (session_id, name, sets, reps))
            exercise_ids.append(cur.lastrowid)
        check(conn.execute(
            "SELECT COUNT(*) FROM Exercise WHERE session_id=?",
            (session_id,)
        ).fetchone()[0] == 2, "세션에 개별 운동 2개 저장")

        # 6. 반복별 자세 기록
        sample_scores = [80.0, 90.0, 100.0]
        for rep_no, score in enumerate(sample_scores, start=1):
            conn.execute("""
                INSERT INTO ExerciseRecord
                    (exercise_id, set_no, rep_no, pose_state, score,
                     error_codes, feedback_code, feedback_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exercise_ids[0], 1, rep_no, "GOOD", score,
                "[]", "GOOD_POSTURE", "테스트 피드백"
            ))

        check(conn.execute(
            "SELECT COUNT(*) FROM ExerciseRecord WHERE exercise_id=?",
            (exercise_ids[0],)
        ).fetchone()[0] == 3, "반복별 운동 기록 3개 저장")

        # 중복 반복 기록은 UNIQUE 제약조건으로 거부되어야 함
        duplicate_rejected = False
        try:
            conn.execute("""
                INSERT INTO ExerciseRecord
                    (exercise_id, set_no, rep_no, pose_state)
                VALUES (?, 1, 1, 'BAD')
            """, (exercise_ids[0],))
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        check(duplicate_rejected, "중복 세트·반복 기록 차단")

        # 존재하지 않는 사용자 FK는 거부되어야 함
        fk_rejected = False
        try:
            conn.execute("""
                INSERT INTO ExerciseSession
                    (user_id, routine_id, day_no, started_at)
                VALUES (?, ?, 1, datetime('now'))
            """, (999999, routine_id))
        except sqlite3.IntegrityError:
            fk_rejected = True
        check(fk_rejected, "존재하지 않는 사용자 참조 차단")

        # 7. 실제 기록에서 평균을 계산해 리포트 저장
        avg_score = conn.execute("""
            SELECT AVG(er.score)
            FROM ExerciseRecord er
            JOIN Exercise e ON e.exercise_id = er.exercise_id
            WHERE e.session_id = ?
        """, (session_id,)).fetchone()[0]

        report_json = {
            "summary": "테스트 운동 완료",
            "exercise_count": 2,
            "valid_record_count": 3,
            "avg_score": round(avg_score, 2)
        }

        conn.execute("""
            UPDATE ExerciseSession
            SET ended_at = datetime('now', 'localtime'),
                session_status = 'COMPLETED'
            WHERE session_id = ?
        """, (session_id,))

        conn.execute("""
            INSERT INTO ExerciseReport (session_id, avg_score, report_json)
            VALUES (?, ?, ?)
        """, (
            session_id, avg_score,
            json.dumps(report_json, ensure_ascii=False)
        ))
        conn.commit()

        saved_report = conn.execute("""
            SELECT avg_score, report_json
            FROM ExerciseReport
            WHERE session_id = ?
        """, (session_id,)).fetchone()
        check(saved_report is not None, "최종 운동 리포트 저장 및 조회")
        check(abs(saved_report["avg_score"] - 90.0) < 0.001,
              "리포트 평균 점수 계산 확인 (예상 90.0)")
        check(
            abs(json.loads(saved_report["report_json"])["avg_score"] -
                saved_report["avg_score"]) < 0.001,
            "리포트 JSON과 avg_score 일치"
        )

        # 날짜별 세션 조회 확인
        daily_sessions = conn.execute("""
            SELECT s.session_id, date(s.started_at) AS exercise_date,
                   r.routine_name, s.day_no, s.session_status
            FROM ExerciseSession s
            JOIN Routine r ON r.routine_id = s.routine_id
            WHERE s.user_id = ?
            ORDER BY s.started_at DESC
        """, (user_id,)).fetchall()
        check(len(daily_sessions) == 1, "사용자별 날짜·세션 조회")

        # DB 무결성 검사
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        check(integrity == "ok", "SQLite DB 무결성 검사")

        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        check(len(fk_errors) == 0, "외래 키 무결성 검사")

        # 사람이 결과를 확인할 수 있도록 리포트 출력
        print("\n===== 최종 리포트 테스트 결과 =====")
        print(f"DB 파일: {DB_PATH}")
        print(f"사용자 ID: {user_id}")
        print(f"체형 분석 ID: {body_id}")
        print(f"루틴 ID: {routine_id}")
        print(f"세션 ID: {session_id}")
        print(f"세션 상태: COMPLETED")
        print(f"유효 운동 기록 수: {len(sample_scores)}")
        print(f"평균 자세 점수: {saved_report['avg_score']:.2f}")
        print("리포트 JSON:")
        print(json.dumps(
            json.loads(saved_report["report_json"]),
            ensure_ascii=False, indent=2
        ))
        print("\n모든 DB 테스트가 통과했습니다.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
