"""AIFit DB 테스트 (aifit.db에 실제로 저장됨)

python test_aifit.py          # 테스트 데이터를 aifit.db에 저장 (실행할 때마다 세션이 하나씩 추가됨)
python test_aifit.py --clean  # 테스트 사용자(tester)와 딸린 세션/기록/리포트를 모두 삭제
"""

import json
import os
import sqlite3
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "aifit_test.db")
SCHEMA = os.path.join(BASE, "AIFit.sql")


def main() -> None:
    """별도 데모 DB에서 제약 조건을 확인합니다. 명시적으로 실행할 때만 저장합니다."""
    conn = sqlite3.connect(DB)
    conn.executescript(
        open(SCHEMA, encoding="utf-8").read()
    )  # 테이블이 없으면 만들고, 있으면 그대로 둠
    conn.execute("PRAGMA foreign_keys = ON")  # 연결마다 필요

    if "--clean" in sys.argv:
        conn.execute(
            "DELETE FROM UserInfo WHERE username = 'tester'"
        )  # CASCADE로 딸린 데이터도 삭제
        conn.commit()
        print(
            "tester 사용자와 그 세션/기록/리포트를 삭제했습니다. (Exercise 'squat'은 남아 있음)"
        )
        raise SystemExit

    def run(title, sql, params=()):
        try:
            cur = conn.execute(sql, params)
            print(f"[OK]   {title}")
            return cur
        except sqlite3.IntegrityError as e:
            print(f"[막힘] {title} -> {e}")

    def now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 사용자/운동 종류: 이미 있으면 재사용 (여러 번 실행해도 중복 에러가 나지 않게)
    row = conn.execute(
        "SELECT user_id FROM UserInfo WHERE username='tester'"
    ).fetchone()
    if row:
        user_id = row[0]
        print("[OK]   사용자 tester 이미 있음, 재사용")
    else:
        user_id = run(
            "사용자 등록",
            "INSERT INTO UserInfo(username,password_hash,email_address,nickname,gender,age,height,weight) VALUES(?,?,?,?,?,?,?,?)",
            (
                "tester",
                "hash123",
                "tester@example.com",
                "테스터",
                "M",
                25,
                175.0,
                70.0,
            ),
        ).lastrowid
        run(
            "체형 등록",
            "INSERT INTO UserBodyInfo(user_id,front_photo_ref,back_photo_ref,side1_photo_ref,side2_photo_ref,analysis_status) VALUES(?,?,?,?,?,?)",
            (user_id, "f.jpg", "b.jpg", "s1.jpg", "s2.jpg", "PENDING"),
        )

    row = conn.execute(
        "SELECT exercise_id FROM Exercise WHERE name='squat'"
    ).fetchone()
    exercise_id = (
        row[0]
        if row
        else run(
            "운동 종류 등록", "INSERT INTO Exercise(name) VALUES('squat')"
        ).lastrowid
    )

    # 정상 흐름: 매번 새 세션 하나 + 1회 결과 3개 + 리포트
    session_id = run(
        "세션 시작",
        "INSERT INTO ExerciseSession(user_id,exercise_id,started_at,session_status) VALUES(?,?,?,?)",
        (user_id, exercise_id, now(), "IN_PROGRESS"),
    ).lastrowid
    for n in (1, 2, 3):
        run(
            f"{n}회 결과 저장",
            "INSERT INTO ExerciseRecord(session_id,rep_no,pose_state,score,grade,error_codes) VALUES(?,?,?,?,?,?)",
            (session_id, n, "OK", 80.0 + n, "B", json.dumps(["KNEE_IN"])),
        )
    run(
        "세션 종료 갱신",
        "UPDATE ExerciseSession SET ended_at=?, rep_count=3, session_status='COMPLETED' WHERE session_id=?",
        (now(), session_id),
    )
    run(
        "리포트 저장",
        "INSERT INTO ExerciseReport(session_id,report_json) VALUES(?,?)",
        (session_id, json.dumps({"avg_score": 82})),
    )

    # 막혀야 정상인 경우 (실패하므로 DB에는 아무것도 남지 않음)
    run(
        "같은 rep_no 중복 저장",
        "INSERT INTO ExerciseRecord(session_id,rep_no,pose_state) VALUES(?,1,'OK')",
        (session_id,),
    )
    run(
        "없는 세션에 결과 저장",
        "INSERT INTO ExerciseRecord(session_id,rep_no,pose_state) VALUES(999999,1,'OK')",
    )
    run(
        "같은 username 가입",
        "INSERT INTO UserInfo(username,password_hash,email_address,nickname,gender,age,height,weight) VALUES('tester','h','other@example.com','n','M',20,170,60)",
    )
    run(
        "같은 email 가입",
        "INSERT INTO UserInfo(username,password_hash,email_address,nickname,gender,age,height,weight) VALUES('tester2','h','tester@example.com','n','M',20,170,60)",
    )
    run(
        "email 없이 가입",
        "INSERT INTO UserInfo(username,password_hash,nickname,gender,age,height,weight) VALUES('tester3','h','n','M',20,170,60)",
    )
    run(
        "세션에 리포트 2개",
        "INSERT INTO ExerciseReport(session_id,report_json) VALUES(?,'{}')",
        (session_id,),
    )

    conn.commit()  # 이 줄이 있어야 aifit.db에 실제로 저장됨
    print("\n저장 완료 -> python db_viewer.py 로 확인하세요")


if __name__ == "__main__":
    main()
