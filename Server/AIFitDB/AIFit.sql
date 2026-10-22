-- AIFit DB schema (final v2, SQLite)
-- 실행: sqlite3 aifit.db < aifit.sql
-- 주의: foreign_keys 설정은 DB 연결마다 켜야 하므로 앱 코드에서도 연결 직후 실행할 것
-- 구조: Routine(루틴 목록) / UserInfo - UserBodyInfo(1:1) / ExerciseSession(Day 1개) - Exercise - ExerciseRecord / ExerciseReport

PRAGMA foreign_keys = ON;

-- 1. 사용자 정보
CREATE TABLE IF NOT EXISTS UserInfo (
    user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    email_address  TEXT    UNIQUE,
    nickname       TEXT    NOT NULL,
    gender         TEXT    NOT NULL,
    age            INTEGER NOT NULL,
    height         REAL    NOT NULL,               -- cm
    weight         REAL    NOT NULL,               -- kg
    created_at     TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 2. 운동 루틴 : 루틴 목록
--    AI가 체형 분석 결과에 따라 이 목록 중 적합한 루틴을 선택
CREATE TABLE IF NOT EXISTS Routine (
    routine_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    routine_name       TEXT    NOT NULL,
    description        TEXT,
    target_body_part   TEXT    NOT NULL,
    routine_plan_json  TEXT    NOT NULL,   -- 7일(Day 1~7) 운동 구성 JSON
    created_at         TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 3. 체형 분석 정보
--    분석 실패 후 다시 시도할 때는 새 행이 아니라 기존 행을 UPDATE
CREATE TABLE IF NOT EXISTS UserBodyInfo (
    body_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    routine_id       INTEGER,                -- AI가 추천한 루틴 (분석 직후에는 NULL)
    front_photo_ref  TEXT    NOT NULL,       -- 사진 파일 경로 (파일 자체는 폴더에 저장)
    back_photo_ref   TEXT    NOT NULL,
    right_photo_ref  TEXT    NOT NULL,
    left_photo_ref   TEXT    NOT NULL,
    chest_status     TEXT,
    shoulder_status  TEXT,
    arm_status       TEXT,
    back_status      TEXT,
    leg_status       TEXT,
    analysis_status  TEXT    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id)    REFERENCES UserInfo(user_id) ON DELETE CASCADE,
    FOREIGN KEY (routine_id) REFERENCES Routine(routine_id)   -- 추천에 쓰인 루틴은 삭제 불가
);

-- 4. 운동 세션 = 루틴의 Day 1개 (완료한 세션만 남기고, 중도 종료 시 세션 삭제)
CREATE TABLE IF NOT EXISTS ExerciseSession (
    session_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    routine_id       INTEGER NOT NULL,
    day_no           INTEGER NOT NULL CHECK (day_no BETWEEN 1 AND 7),
    started_at       TEXT    NOT NULL,
    ended_at         TEXT,
    session_status   TEXT    NOT NULL DEFAULT 'IN_PROGRESS',   -- IN_PROGRESS / PAUSED / COMPLETED
    paused_at        TEXT,                                      -- 일시정지 시작 시각 (재개하면 NULL)
    paused_duration  INTEGER NOT NULL DEFAULT 0 CHECK (paused_duration >= 0),  -- 누적 일시정지 시간(초)
    FOREIGN KEY (user_id)    REFERENCES UserInfo(user_id) ON DELETE CASCADE,
    FOREIGN KEY (routine_id) REFERENCES Routine(routine_id)   -- 사용 중인 루틴은 삭제 불가
);
CREATE INDEX IF NOT EXISTS idx_session_user_started ON ExerciseSession(user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_session_user_routine_day ON ExerciseSession(user_id, routine_id, day_no);

-- 5. 세션 안의 개별 운동 (세션 시작 시 routine_plan_json의 해당 Day 운동을 복사해 저장)
CREATE TABLE IF NOT EXISTS Exercise (
    exercise_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL,
    exercise_name  TEXT    NOT NULL,                      -- AI 서버에 전달하는 고정 목록의 운동 이름
    sets           INTEGER NOT NULL CHECK (sets >= 1),    -- 목표 세트 수
    reps           INTEGER NOT NULL CHECK (reps >= 1),    -- 세트당 목표 반복 횟수
    FOREIGN KEY (session_id) REFERENCES ExerciseSession(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_exercise_session ON Exercise(session_id);

-- 6. 1회별 운동 결과 (유효한 1회만 저장)
CREATE TABLE IF NOT EXISTS ExerciseRecord (
    record_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    exercise_id    INTEGER NOT NULL,
    set_no         INTEGER NOT NULL CHECK (set_no >= 1),
    rep_no         INTEGER NOT NULL CHECK (rep_no >= 1),
    pose_state     TEXT    NOT NULL,
    score          REAL,
    error_codes    TEXT,               -- JSON 문자열 (오류 없으면 NULL 또는 '[]' 중 하나로 통일)
    feedback_code  TEXT,
    feedback_text  TEXT,
    UNIQUE (exercise_id, set_no, rep_no),
    FOREIGN KEY (exercise_id) REFERENCES Exercise(exercise_id) ON DELETE CASCADE
);

-- 7. 최종 운동 리포트 (세션당 1개)
CREATE TABLE IF NOT EXISTS ExerciseReport (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL UNIQUE,
    avg_score    REAL,
    report_json  TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES ExerciseSession(session_id) ON DELETE CASCADE
);