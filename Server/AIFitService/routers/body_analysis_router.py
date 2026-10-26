import json
import random
import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from config.settings import BASE_DIR
from database.connection import get_connection
from services.exercise_names import canonical_exercise_name, canonicalize_routine_exercises
from services.token_service import get_current_user

router = APIRouter(tags=["Body Analysis"])
UPLOAD_DIR = BASE_DIR / "body_analysis_uploads"
LATEST_DIR = UPLOAD_DIR / "latest"
PROCESSING_DIR = UPLOAD_DIR / "processing"

PHOTO_FIELDS = {
    "front_photo": "front",
    "left_photo": "left",
    "right_photo": "right",
    "back_photo": "back",
}
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
EXERCISE_BY_PART = {
    "chest_status": ("가슴", "push_up"),
    "shoulder_status": ("어깨", "dumbbell_shoulder_press"),
    "arm_status": ("팔", "plank"),
    "back_status": ("등", "ytw_raise"),
    "leg_status": ("하체", "squat"),
}
ALLOWED_EXERCISES = {exercise for _, exercise in EXERCISE_BY_PART.values()}


def init_body_analysis_db():
    with get_connection() as conn:
        # The workout flow stores the generated plan in these application tables.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Routine (
                routine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                routine_name TEXT NOT NULL,
                description TEXT,
                target_body_part TEXT NOT NULL,
                routine_plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS UserBodyInfo (
                body_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                routine_id INTEGER,
                front_photo_ref TEXT NOT NULL,
                back_photo_ref TEXT NOT NULL,
                right_photo_ref TEXT NOT NULL,
                left_photo_ref TEXT NOT NULL,
                chest_status TEXT,
                shoulder_status TEXT,
                arm_status TEXT,
                back_status TEXT,
                leg_status TEXT,
                analysis_status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES UserInfo(user_id) ON DELETE CASCADE,
                FOREIGN KEY (routine_id) REFERENCES Routine(routine_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS BodyAnalysis (
                body_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'processing',
                analysis_status TEXT NOT NULL DEFAULT 'processing',
                front_photo_path TEXT NOT NULL,
                left_photo_path TEXT NOT NULL,
                right_photo_path TEXT NOT NULL,
                back_photo_path TEXT NOT NULL,
                body_type_classification TEXT,
                recognized INTEGER,
                failure_reason TEXT,
                chest_status TEXT,
                shoulder_status TEXT,
                arm_status TEXT,
                back_status TEXT,
                leg_status TEXT,
                analysis_prompt TEXT,
                routine_id INTEGER,
                routine_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        # Upgrade existing SQLite databases without removing previous records.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(BodyAnalysis)")}
        migrations = {
            "analysis_status": "TEXT NOT NULL DEFAULT 'processing'",
            "recognized": "INTEGER",
            "failure_reason": "TEXT",
            "shoulder_status": "TEXT",
            "leg_status": "TEXT",
            "analysis_prompt": "TEXT",
            "routine_id": "INTEGER",
            "routine_json": "TEXT",
        }
        for name, declaration in migrations.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE BodyAnalysis ADD COLUMN {name} {declaration}")
        conn.execute("UPDATE BodyAnalysis SET analysis_status = status WHERE analysis_status = 'processing' AND status != 'processing'")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_body_analysis_user
            ON BodyAnalysis(user_id)
        """)


def _validated_routine(result: dict) -> dict | None:
    """Validate the model plan against the workout and weak-area rules."""
    if not result.get("recognized"):
        return None
    routine = result.get("routine")
    if not isinstance(routine, dict):
        raise ValueError("체형 분석 결과에 7일 루틴이 없습니다.")

    days = routine.get("days")
    if not isinstance(days, list) or len(days) != 7:
        raise ValueError("7일 루틴에는 Day 1부터 Day 7까지 모두 있어야 합니다.")
    by_day = {}
    for day in days:
        if not isinstance(day, dict) or day.get("day_no") not in range(1, 8):
            raise ValueError("루틴의 day_no는 1부터 7까지여야 합니다.")
        day_no = day["day_no"]
        if day_no in by_day:
            raise ValueError(f"Day {day_no}가 중복되었습니다.")
        exercises = day.get("exercises")
        if not isinstance(exercises, list):
            raise ValueError(f"Day {day_no}의 exercises 형식이 잘못되었습니다.")
        by_day[day_no] = exercises
    if set(by_day) != set(range(1, 8)):
        raise ValueError("Day 1부터 Day 7까지 빠짐없이 있어야 합니다.")

    normalized_days = []
    selected_by_day = {}
    for day_no in range(1, 8):
        exercises = by_day[day_no]
        expected_count = 1 if day_no <= 5 else None
        if expected_count == 1 and len(exercises) != 1:
            raise ValueError(f"평일 Day {day_no}에는 운동을 정확히 하나 넣어야 합니다.")
        if day_no >= 6 and len(exercises) > 1:
            raise ValueError("토요일과 일요일에는 각각 운동을 최대 하나만 넣을 수 있습니다.")

        normalized_exercises = []
        for item in exercises:
            if not isinstance(item, dict):
                raise ValueError(f"Day {day_no}의 운동 항목 형식이 잘못되었습니다.")
            name = canonical_exercise_name(item.get("exercise_name"))
            sets = item.get("sets")
            reps = item.get("reps")
            duration = item.get("target_duration_sec")
            if name not in ALLOWED_EXERCISES:
                raise ValueError(f"허용되지 않은 운동입니다: {name}")
            if not isinstance(sets, int) or sets < 1:
                raise ValueError(f"{name}의 세트 수는 1 이상이어야 합니다.")
            if name == "plank":
                if reps is not None or not isinstance(duration, (int, float)) or duration <= 0:
                    raise ValueError("플랭크는 reps=null과 양수 target_duration_sec가 필요합니다.")
            elif not isinstance(reps, int) or reps < 1 or duration is not None:
                raise ValueError(f"{name}은 양수 reps와 null target_duration_sec가 필요합니다.")
            normalized_exercises.append({
                "exercise_name": name,
                "sets": sets,
                "reps": reps,
                "target_duration_sec": duration,
            })
        selected_by_day[day_no] = [item["exercise_name"] for item in normalized_exercises]
        normalized_days.append({"day_no": day_no, "exercises": normalized_exercises})

    # No identical exercise may appear on two adjacent calendar days.
    for day_no in range(1, 7):
        if selected_by_day[day_no] and selected_by_day[day_no + 1] and selected_by_day[day_no][0] == selected_by_day[day_no + 1][0]:
            raise ValueError("같은 운동이 연속된 날에 배치되었습니다.")

    weak_parts = [
        key for key in EXERCISE_BY_PART
        if result.get(key) == "bad"
    ]
    expected_weekend = []
    if len(weak_parts) == 1:
        expected_weekend = [(6, weak_parts[0])]
    elif len(weak_parts) == 2:
        expected_weekend = [(6, weak_parts[0]), (7, weak_parts[1])]
    elif len(weak_parts) >= 3:
        expected_weekend = [(6, None), (7, None)]

    if len(weak_parts) <= 2:
        expected_by_day = dict(expected_weekend)
        for day_no in (6, 7):
            expected = expected_by_day.get(day_no)
            actual = selected_by_day[day_no]
            if expected is None and actual:
                raise ValueError(f"Day {day_no}는 휴식일이어야 합니다.")
            if expected is not None and actual != [EXERCISE_BY_PART[expected][1]]:
                raise ValueError(f"Day {day_no}에 약점 부위의 지정 운동이 필요합니다.")
    else:
        weekend_names = [selected_by_day[6][0] if selected_by_day[6] else None,
                         selected_by_day[7][0] if selected_by_day[7] else None]
        weak_exercises = {EXERCISE_BY_PART[key][1] for key in weak_parts}
        if any(name not in weak_exercises for name in weekend_names) or len(set(weekend_names)) != 2:
            raise ValueError("약점이 3개 이상이면 서로 다른 약점 운동 2개를 주말에 하나씩 배치해야 합니다.")

    routine_name = routine.get("routine_name")
    description = routine.get("description")
    if not isinstance(routine_name, str) or not routine_name.strip():
        raise ValueError("루틴 이름이 비어 있습니다.")
    if not isinstance(description, str):
        raise ValueError("루틴 설명 형식이 잘못되었습니다.")
    target_body_part = ",".join(EXERCISE_BY_PART[key][0] for key in weak_parts) or "balanced"
    return {
        "routine_name": routine_name.strip(),
        "description": description.strip(),
        "target_body_part": target_body_part,
        "days": normalized_days,
    }


def _build_routine_from_statuses(result: dict) -> dict:
    """Create a valid fallback plan using only the five supported exercises."""
    weak_parts = [key for key in EXERCISE_BY_PART if result.get(key) == "bad"]
    if len(weak_parts) == 1:
        saturday_part = sunday_part = None
        saturday_part = weak_parts[0]
    elif len(weak_parts) == 2:
        saturday_part, sunday_part = weak_parts
    elif len(weak_parts) >= 3:
        saturday_part, sunday_part = random.sample(weak_parts, 2)
    else:
        saturday_part = sunday_part = None

    saturday_name = EXERCISE_BY_PART[saturday_part][1] if saturday_part else None
    weekend_parts = [EXERCISE_BY_PART[key][0] for key in weak_parts]
    target_body_part = ",".join(weekend_parts) or "balanced"

    weekdays = []
    exercises = list(ALLOWED_EXERCISES)

    def add_weekday(day_no: int) -> bool:
        if day_no > 5:
            return True
        previous = weekdays[-1]["exercise_name"] if weekdays else None
        choices = [name for name in exercises if name != previous]
        if day_no == 5 and saturday_name:
            choices = [name for name in choices if name != saturday_name]
        random.shuffle(choices)
        for name in choices:
            item = {
                "exercise_name": name,
                "sets": 3,
                "reps": None if name == "plank" else 10,
                "target_duration_sec": 30 if name == "plank" else None,
            }
            weekdays.append(item)
            if add_weekday(day_no + 1):
                return True
            weekdays.pop()
        return False

    if not add_weekday(1):
        raise ValueError("다섯 평일 운동을 연속 중복 없이 구성하지 못했습니다.")

    days = [
        {"day_no": day_no, "exercises": [exercise]}
        for day_no, exercise in enumerate(weekdays, start=1)
    ]
    days.extend([
        {
            "day_no": 6,
            "exercises": ([{
                "exercise_name": saturday_name,
                "sets": 3,
                "reps": None if saturday_name == "plank" else 10,
                "target_duration_sec": 30 if saturday_name == "plank" else None,
            }] if saturday_name else []),
        },
        {
            "day_no": 7,
            "exercises": ([{
                "exercise_name": EXERCISE_BY_PART[sunday_part][1],
                "sets": 3,
                "reps": None if EXERCISE_BY_PART[sunday_part][1] == "plank" else 10,
                "target_duration_sec": 30 if EXERCISE_BY_PART[sunday_part][1] == "plank" else None,
            }] if sunday_part else []),
        },
    ])
    return {
        "routine_name": "체형 맞춤 7일 운동 루틴",
        "description": "체형 분석 상태를 반영해 구성한 주간 운동 루틴입니다.",
        "target_body_part": target_body_part,
        "days": days,
    }


def cleanup_old_uploads():
    """Remove legacy per-request photo folders once no analysis is running."""
    if not UPLOAD_DIR.exists():
        return
    with get_connection() as conn:
        pending = conn.execute(
            "SELECT 1 FROM BodyAnalysis WHERE analysis_status = 'processing' LIMIT 1"
        ).fetchone()
    if pending:
        return

    for directory in UPLOAD_DIR.iterdir():
        if not directory.is_dir() or directory.name in {"latest", "processing"}:
            continue
        try:
            UUID(directory.name)
        except ValueError:
            continue
        shutil.rmtree(directory, ignore_errors=True)


def run_body_analysis(body_id: str, processing_paths: dict[str, str]):
    """Analyze saved photos and persist the structured result for GUI polling."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT analysis_prompt, user_id, front_photo_path, left_photo_path,
                      right_photo_path, back_photo_path
               FROM BodyAnalysis WHERE body_id = ?""",
            (body_id,),
        ).fetchone()
    if row is None:
        shutil.rmtree(PROCESSING_DIR / body_id, ignore_errors=True)
        return

    try:
        from services.body_analysis_service import analyze_body_photos

        result = analyze_body_photos(
            processing_paths,
            row["analysis_prompt"],
        )
        print(
            f"[BODY_ANALYSIS] ChatGPT response body_id={body_id}: "
            f"{json.dumps(result, ensure_ascii=False)}",
            flush=True,
        )
        try:
            routine = _validated_routine(result)
        except ValueError as exc:
            if not result.get("recognized"):
                raise
            print(
                f"[BODY_ANALYSIS][ROUTINE_FALLBACK] body_id={body_id}: "
                f"{str(exc)[:300]}; constructing a valid schedule from body statuses",
                flush=True,
            )
            routine = _build_routine_from_statuses(result)
        with get_connection() as conn:
            routine_id = None
            routine_json = None
            if routine is not None:
                routine_json = json.dumps(routine, ensure_ascii=False)
                cursor = conn.execute(
                    """INSERT INTO Routine (
                           routine_name, description, target_body_part, routine_plan_json
                       ) VALUES (?, ?, ?, ?)""",
                    (routine["routine_name"], routine["description"],
                     routine["target_body_part"], routine_json),
                )
                routine_id = cursor.lastrowid
                print(
                    f"[BODY_ANALYSIS][ROUTINE_SAVED] body_id={body_id} "
                    f"routine_id={routine_id} day_count={len(routine['days'])}",
                    flush=True,
                )

            body_info = conn.execute(
                """SELECT body_id FROM UserBodyInfo WHERE user_id = ?
                   ORDER BY created_at DESC, body_id DESC LIMIT 1""",
                (row["user_id"],),
            ).fetchone()
            body_values = (
                routine_id,
                row["front_photo_path"],
                row["back_photo_path"],
                row["right_photo_path"],
                row["left_photo_path"],
                result["chest_status"],
                result["shoulder_status"],
                result["arm_status"],
                result["back_status"],
                result["leg_status"],
                "completed",
            )
            if body_info:
                conn.execute(
                    """UPDATE UserBodyInfo SET
                           routine_id=?, front_photo_ref=?, back_photo_ref=?,
                           right_photo_ref=?, left_photo_ref=?, chest_status=?,
                           shoulder_status=?, arm_status=?, back_status=?, leg_status=?,
                           analysis_status=?, created_at=datetime('now', 'localtime')
                       WHERE body_id=?""",
                    (*body_values, body_info["body_id"]),
                )
            else:
                conn.execute(
                    """INSERT INTO UserBodyInfo (
                           user_id, routine_id, front_photo_ref, back_photo_ref,
                           right_photo_ref, left_photo_ref, chest_status,
                           shoulder_status, arm_status, back_status, leg_status,
                           analysis_status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row["user_id"], *body_values),
                )

            conn.execute(
                """UPDATE BodyAnalysis
                   SET analysis_status='completed', status='completed',
                       recognized=?, failure_reason=?,
                       chest_status=?, shoulder_status=?, arm_status=?,
                       back_status=?, leg_status=?, routine_id=?, routine_json=?,
                       error_message=NULL
                   WHERE body_id=?""",
                (int(result["recognized"]), result["failure_reason"],
                 result["chest_status"], result["shoulder_status"],
                 result["arm_status"], result["back_status"],
                 result["leg_status"], routine_id, routine_json, body_id),
            )
    except Exception as exc:
        # Show local validation messages for diagnosis, while keeping provider
        # request/response internals and credentials out of logs.
        detail = str(exc)[:300] if isinstance(exc, ValueError) else type(exc).__name__
        print(
            f"[BODY_ANALYSIS] Analysis failed for body_id={body_id}: "
            f"{type(exc).__name__}: {detail}",
            flush=True,
        )
        with get_connection() as conn:
            conn.execute(
                """UPDATE BodyAnalysis SET analysis_status='failed', status='failed',
                   error_message='체형 분석에 실패했습니다. 서버 설정과 로그를 확인해 주세요.'
                   WHERE body_id=?""",
                (body_id,),
            )
    finally:
        shutil.rmtree(PROCESSING_DIR / body_id, ignore_errors=True)
        cleanup_old_uploads()


@router.post("/body-analysis", status_code=202)
async def upload_body_analysis(
    background_tasks: BackgroundTasks,
    front_photo: UploadFile = File(...),
    left_photo: UploadFile = File(...),
    right_photo: UploadFile = File(...),
    back_photo: UploadFile = File(...),
    prompt: str | None = Form(default=None, max_length=4000),
    current_user: dict = Depends(get_current_user),
):
    body_id = str(uuid4())
    user_id = current_user["user_id"]
    body_dir = PROCESSING_DIR / body_id
    body_dir.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_paths = {}
    processing_paths = {}
    files = {
        "front_photo": front_photo,
        "left_photo": left_photo,
        "right_photo": right_photo,
        "back_photo": back_photo,
    }

    try:
        for field, file in files.items():
            extension = ALLOWED_TYPES.get(file.content_type)
            if not extension:
                raise HTTPException(
                    status_code=415,
                    detail=f"{field}: JPG, PNG, WEBP 파일만 업로드할 수 있습니다.",
                )

            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail=f"{field} 파일이 비어 있습니다.")
            if len(contents) > 20 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{field} 파일이 너무 큽니다.")

            side = PHOTO_FIELDS[field]
            processing_path = body_dir / f"{side}{extension}"
            processing_path.write_bytes(contents)
            processing_paths[field] = str(processing_path)

        # Commit all four latest images together after validating the full upload.
        for field, processing_path_string in processing_paths.items():
            side = PHOTO_FIELDS[field]
            processing_path = Path(processing_path_string)
            extension = processing_path.suffix
            for old_file in LATEST_DIR.glob(f"{side}.*"):
                old_file.unlink(missing_ok=True)
            latest_path = LATEST_DIR / f"{side}{extension}"
            shutil.copyfile(processing_path, latest_path)
            uploaded_paths[field] = str(latest_path)

        with get_connection() as conn:
            conn.execute("""
                INSERT INTO BodyAnalysis (
                    body_id, user_id, status,
                    front_photo_path, left_photo_path,
                right_photo_path, back_photo_path, analysis_prompt
            ) VALUES (?, ?, 'processing', ?, ?, ?, ?, ?)
            """, (
                body_id,
                user_id,
                uploaded_paths["front_photo"],
                uploaded_paths["left_photo"],
                uploaded_paths["right_photo"],
                uploaded_paths["back_photo"],
                prompt,
            ))

    except Exception:
        # 저장이나 DB 기록이 실패하면 이번 업로드 파일을 정리합니다.
        shutil.rmtree(body_dir, ignore_errors=True)
        raise
    finally:
        for file in files.values():
            await file.close()

    background_tasks.add_task(run_body_analysis, body_id, processing_paths)

    return {"body_id": body_id, "analysis_status": "processing", "status": "processing"}


@router.get("/body-analysis/{body_id}")
def get_body_analysis(
    body_id: str,
    current_user: dict = Depends(get_current_user),
):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT body_id, analysis_status, recognized, failure_reason,
                   chest_status, shoulder_status, arm_status, back_status,
                   leg_status, routine_id, routine_json, error_message
            FROM BodyAnalysis
            WHERE body_id = ? AND user_id = ?
        """, (body_id, current_user["user_id"])).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="체형 분석 기록을 찾을 수 없습니다.")

    response = {
        "body_id": row["body_id"],
        "analysis_status": row["analysis_status"],
        "status": row["analysis_status"],
    }

    if row["analysis_status"] == "completed":
        routine = None
        if row["routine_json"]:
            try:
                routine = canonicalize_routine_exercises(json.loads(row["routine_json"]))
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="저장된 7일 루틴 JSON 형식이 잘못되었습니다.")
        response["result"] = {
            "recognized": bool(row["recognized"]) if row["recognized"] is not None else None,
            "failure_reason": row["failure_reason"],
            "chest_status": row["chest_status"],
            "shoulder_status": row["shoulder_status"],
            "arm_status": row["arm_status"],
            "back_status": row["back_status"],
            "leg_status": row["leg_status"],
            "routine_id": row["routine_id"],
            "routine": routine,
        }
    elif row["analysis_status"] == "failed":
        response["error"] = row["error_message"]

    return response
