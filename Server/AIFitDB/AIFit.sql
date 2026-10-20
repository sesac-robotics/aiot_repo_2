-- AIFit DB schema (SQLite)
-- 실행: sqlite3 aifit.db < aifit_schema.sql
-- 주의: foreign_keys 설정은 DB 연결마다 켜야 하므로 앱 코드에서도 연결 직후 실행할 것

PRAGMA foreign_keys = ON;

-- 1. 사용자 정보 (SR-001)
CREATE TABLE IF NOT EXISTS UserInfo (
    user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    email_address  TEXT    NOT NULL UNIQUE,
    nickname       TEXT    NOT NULL,
    gender         TEXT    NOT NULL,
    age            INTEGER NOT NULL,
    height         REAL    NOT NULL,   -- cm
    weight         REAL    NOT NULL,   -- kg
    created_at     TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 2. 체형 분석 정보 (SR-002)
CREATE TABLE IF NOT EXISTS UserBodyInfo (
    body_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                   INTEGER NOT NULL,
    front_photo_ref           TEXT    NOT NULL,   -- 사진 파일 경로 (파일 자체는 폴더에 저장)
    back_photo_ref            TEXT    NOT NULL,
    side1_photo_ref           TEXT    NOT NULL,
    side2_photo_ref           TEXT    NOT NULL,
    body_type_classification  TEXT,
    chest_status              TEXT,
    shoulder_status           TEXT,
    arm_status                TEXT,
    back_status               TEXT,
    leg_status                TEXT,
    analysis_status           TEXT    NOT NULL,
    created_at                TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES UserInfo(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bodyinfo_user_created ON UserBodyInfo(user_id, created_at);

-- 3. 운동 종류
CREATE TABLE IF NOT EXISTS Exercise (
    exercise_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE
);

-- 4. 운동 세션
CREATE TABLE IF NOT EXISTS ExerciseSession (
    session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    exercise_id     INTEGER NOT NULL,
    started_at      TEXT    NOT NULL,
    ended_at        TEXT,
    rep_count       INTEGER NOT NULL DEFAULT 0 CHECK (rep_count >= 0),
    session_status  TEXT    NOT NULL,
    FOREIGN KEY (user_id)     REFERENCES UserInfo(user_id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_id) REFERENCES Exercise(exercise_id)
);
CREATE INDEX IF NOT EXISTS idx_session_user_started ON ExerciseSession(user_id, started_at);

-- 5. 1회별 운동 결과 (유효한 1회만 저장)
CREATE TABLE IF NOT EXISTS ExerciseRecord (
    record_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL,
    rep_no         INTEGER NOT NULL CHECK (rep_no >= 1),
    pose_state     TEXT    NOT NULL,
    score          REAL,
    grade          TEXT,
    error_codes    TEXT,               -- JSON 문자열
    feedback_code  TEXT,
    feedback_text  TEXT,
    recorded_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (session_id, rep_no),
    FOREIGN KEY (session_id) REFERENCES ExerciseSession(session_id) ON DELETE CASCADE
);

-- 6. 최종 운동 리포트 (세션당 1개)
CREATE TABLE IF NOT EXISTS ExerciseReport (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL UNIQUE,
    report_json  TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES ExerciseSession(session_id) ON DELETE CASCADE
);