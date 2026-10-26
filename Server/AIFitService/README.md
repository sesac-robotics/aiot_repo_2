# AIFit Server: 인증·체형 사진 분석

## 설치 및 실행 (Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r Server/AIFitService/requirements.txt
export AIFIT_DB_PATH="$(pwd)/Server/AIFitDB/aifit.db"
export AIFIT_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export OPENAI_API_KEY="발급받은_API_키"
# AI Server의 기존 영상 스트림을 AIFit Server가 직접 읽기 위한 설정
export AI_SERVER_BASE_URL="http://192.168.15.208"
export AIFIT_HOST="192.168.15.148"
export AI_SERVER_API_KEY="<AI Server가 허용하는 키>"
export AI_SERVER_STREAM_PATH="/stream"
# 선택 설정
export OPENAI_MODEL="gpt-4o-mini"
python -m Server.AIFitService.main
```

`AIFIT_JWT_SECRET`은 서버가 실행되는 동안 고정되어야 합니다. 실제 운영에서는 안전한 환경 변수/비밀 관리 기능을 사용하고 개발용 기본 키를 사용하지 마세요.

## 엔드포인트
- `GET /health`: 서버 상태 (토큰 불필요)
- `POST /api/auth/register`: 회원가입 (토큰 불필요)
- `POST /api/auth/login`: 로그인 및 JWT 발급 (토큰 불필요)
- `GET /api/me`: JWT가 있어야 접근 가능한 예시 보호 API
- `POST /body-analysis`: 인증 후 정면·좌측·우측·후면 사진과 선택 프롬프트를 업로드합니다.
- `GET /body-analysis/{body_id}`: 해당 분석 작업의 상태, 부위 평가, 생성된 7일 루틴을 조회합니다.
- `GET /exercise-report?date=YYYY-MM-DD`: 인증한 사용자의 날짜별 완료 운동 리포트를 조회합니다. 같은 날짜에 리포트가 여러 개면 가장 최근 리포트를 반환합니다.
- `POST /api/workout-sessions`: 인증한 사용자의 오늘 루틴 세션과 첫 운동을 만들고 AI Server에 시작을 요청합니다.
- `GET /api/workout-sessions/{session_id}/live`: AI Server의 현재 세션 상태를 GUI에 전달하고 목표 반복/유지 시간에 도달하면 자동 일시정지합니다.
- `POST /api/workout-sessions/{session_id}/records`: 내부 키로 인증한 AI Server에서 1회 운동 평가 결과를 받습니다.
- `POST /api/workout-sessions/{session_id}/pause`: GUI의 Pause 요청을 AI Server에 전달합니다.
- `POST /api/workout-sessions/{session_id}/start`: 일시정지를 재개합니다. 세트 완료 일시정지였다면 다음 세트 또는 운동의 카운터를 초기화합니다.
- `POST /api/workout-sessions/{session_id}/finish`: AI Server와 AIFit DB의 세션을 완료 처리합니다.
- `POST /api/video/frame`: 선택 기능. 내부 API 키로 인증한 발신자가 JPEG 바이너리 프레임을 보냅니다.
- `GET /api/video/stream`: JWT 인증 후 가장 최근 프레임들을 GUI에 MJPEG 스트림으로 전달합니다.
- `GET /api/video/status`: 프레임 수신 여부와 마지막 수신 시각을 반환합니다.

영상 수신은 AIFit Server가 직접 AI Server의 `AI_SERVER_BASE_URL` +
`AI_SERVER_STREAM_PATH` 주소에 연결해 처리합니다. AI Server가 요구하는 키를
`AI_SERVER_API_KEY`에 설정하면 서버 시작 시 연결을 시작하고 연결이 끊기면 자동으로
재연결합니다. 별도의 `video_frame_forwarder.py` 실행은 필요하지 않습니다.
GUI는 로그인 응답의 JWT를 `Authorization: Bearer <token>` 헤더로 보내
`GET /api/video/stream`에 연결합니다. 프레임은 AIFit Server 메모리에 최신 1장만
보관하므로 서버를 재시작하면 새 프레임을 기다립니다. 여러 Uvicorn worker를 사용할
경우 worker끼리 프레임 메모리를 공유하지 않으므로 단일 worker로 실행하거나 별도
공유 저장소를 구성해야 합니다.

`POST /api/video/frame` 방식으로 AI Server가 프레임을 직접 업로드하도록 구성한 경우에만
`AIFIT_VIDEO_API_KEY`를 AIFit Server에 설정하고, 동일한 키를 발신 요청의
`X-Internal-API-Key` 헤더에 넣어야 합니다. 직접 AI 스트림을 읽는 기본 방식에는 이 키가
필요하지 않습니다.

운동 세션 시작은 로그인 JWT를 넣어 `POST /api/workout-sessions`를 호출합니다.
AIFit Server는 사용자의 최신 `UserBodyInfo.routine_id`에 연결된 루틴에서 오늘 요일의
운동을 읽고 `ExerciseSession` 및 그날의 `Exercise` 행을 생성합니다. Day 번호는
월요일=1부터 일요일=7까지입니다. DB에서 생성한 `session_id`와 첫 `exercise_id`를
AI Server의 `POST /internal/sessions/{session_id}/start` 및
`POST /internal/sessions/{session_id}/exercises/{exercise_id}/start`에 전달합니다.
각 운동 요청에는 루틴의 운동 이름, 순서, 목표 반복 횟수 또는 `target_duration_sec`를
전달합니다. `GET .../{session_id}/live`를 GUI가 주기적으로 호출하면 AI Server의 세션,
`/pose/latest`, `/metrics` 응답과 저장된 최신 반복 평가 결과를 받을 수 있습니다. 목표에
도달한 세트는 AIFit Server가 AI
Server에 pause를 보내고 GUI의 `start` 요청을 기다립니다. 다음 세트나 운동은 `start`
요청 때 시작합니다. 사용자가 직접 Pause를 누르면 현재 반복 카운터를 보존하고, Start를
누르면 카운터를 초기화하지 않고 재개합니다.

반복별 자세 평가를 저장하려면 AI Server가 평가할 때마다 아래 주소로 결과를 보내야
합니다. AI Server와 AIFit Server는 동일한 내부 키를 `X-Internal-API-Key`로 사용합니다.
현재 AI Server OpenAPI 화면에는 이 callback 경로가 포함되어 있지 않으므로 AIFit Server의
수신 기능만 준비된 상태이며, AI Server가 결과를 보내도록 연결하는 작업은 별도로
필요합니다.

```http
POST /api/workout-sessions/{session_id}/records
X-Internal-API-Key: <AI_SERVER_API_KEY>
Content-Type: application/json

{
  "exercise_id": 123,
  "set_no": 1,
  "rep_no": 1,
  "pose_state": "good",
  "score": 92.5,
  "error_codes": [],
  "feedback_code": null,
  "feedback_text": "좋은 자세입니다."
}
```

루틴 JSON은 `{"days": [{"day_no": 1, "exercises": [...]}]}` 또는 단일 Day 객체
형태를 지원합니다. 세션 시작 전 사용자에게 `UserBodyInfo.routine_id`가 연결되어 있어야
하고 오늘 Day의 운동 목록이 존재해야 합니다. 현재 `Exercise` 테이블에 없는
`order_no`, `target_duration_sec`, 그리고 세션 진행 상태 저장 테이블은 서버 시작 시
호환성 마이그레이션으로 추가됩니다.

GUI API 호출 로그는 `[GUI][REQUEST]`, `[GUI][SUCCESS]`, `[GUI][ERROR]`로 구분하며,
AI Server 호출 로그는 `[AI_SERVER][REQUEST]`, `[AI_SERVER][SUCCESS]`,
`[AI_SERVER][ERROR]`로 구분합니다. 요청 본문, JWT, 내부 API 키는 로그에 기록하지 않습니다.

사진 업로드는 `multipart/form-data`이며 필드는 `front_photo`, `left_photo`,
`right_photo`, `back_photo`, 선택 필드 `prompt`입니다. 성공하면 `body_id`와
`analysis_status: processing`을 반환합니다. GUI는 이 ID로 조회를 반복하고,
완료 시 `result` 안의 `chest_status`, `shoulder_status`, `arm_status`,
`back_status`, `leg_status`와 `routine_id`, `routine`을 사용하면 됩니다. 각 상태값은 `great`, `good`, `bad`, `null` 중 하나입니다. 사진의 방향이나 서로 같은 사진인지 검증하지 않고, 보이는 근육을 종합 평가합니다. `recognized`는 근육 평가 자체가 가능한지 나타내며, 평가할 수 없는 경우 사유는 `failure_reason`에 담깁니다. 인식에 성공하면 생성된 루틴은 `Routine.routine_plan_json`에 저장되고, 해당 루틴의 ID는 사용자의 `UserBodyInfo.routine_id`에 연결됩니다. GUI 응답의 `result.routine`에는 이름, 설명, 목표 부위, Day 1~7 운동 목록이 포함됩니다. 평가 자체가 불가능하면 routine은 `null`이고 DB에 새 Routine을 만들지 않습니다. API 처리 자체가 실패하면
`analysis_status: failed`와 일반 오류 안내가 반환됩니다.

기본 분석 지침은 `config/settings.py`의 `BODY_ANALYSIS_PROMPT` 기본 문구로
정의되어 있으며, `AIFIT_BODY_ANALYSIS_PROMPT` 환경 변수로 바꿀 수 있습니다.
업로드 요청의 `prompt` 필드는 추가 지침으로 전달됩니다. 분석은 실제 OpenAI API
호출을 하므로 `OPENAI_API_KEY`가 필요하며, 요청 사진은 분석을 위해 OpenAI로
전송됩니다. 모델은 `OPENAI_MODEL`로 설정할 수 있습니다.

7일 루틴 생성 지침과 응답 JSON Schema는 `services/body_analysis_service.py`에
정의되어 있습니다. 월~금에는 하루 한 가지씩 다섯 허용 운동 중 하나를 무작위로
선택하고, 같은 운동을 연속된 날에 배치하지 않습니다. `bad`로 평가된 약점은
가슴=푸쉬업, 어깨=덤벨 숄더 프레스, 팔=플랭크, 등=YTW 레이즈, 하체=스쿼트로
연결합니다. 약점이 1개면 토요일, 2개면 토·일요일에 순서대로 하나씩 추가하고,
3개 이상이면 약점 중 서로 다른 두 부위를 토·일요일에 하나씩 추가합니다.
생성된 계획은 서버에서 날짜, 운동 이름, 세트/반복 또는 플랭크 유지 시간, 약점
배치 규칙을 다시 검증한 뒤 저장합니다. GUI 응답은 다음 형태입니다.

```json
{
  "body_id": "analysis-uuid",
  "analysis_status": "completed",
  "result": {
    "recognized": true,
    "failure_reason": null,
    "chest_status": "good",
    "shoulder_status": "bad",
    "arm_status": "good",
    "back_status": "good",
    "leg_status": "great",
    "routine_id": 12,
    "routine": {
      "routine_name": "7일 맞춤 루틴",
      "description": "...",
      "target_body_part": "어깨",
      "days": [
        {"day_no": 1, "exercises": [{"exercise_name": "스쿼트", "sets": 3, "reps": 10, "target_duration_sec": null}]},
        {"day_no": 2, "exercises": [{"exercise_name": "푸쉬업", "sets": 3, "reps": 10, "target_duration_sec": null}]},
        {"day_no": 3, "exercises": [{"exercise_name": "플랭크", "sets": 3, "reps": null, "target_duration_sec": 30}]},
        {"day_no": 4, "exercises": [{"exercise_name": "YTW 레이즈", "sets": 3, "reps": 10, "target_duration_sec": null}]},
        {"day_no": 5, "exercises": [{"exercise_name": "스쿼트", "sets": 3, "reps": 10, "target_duration_sec": null}]},
        {"day_no": 6, "exercises": [{"exercise_name": "덤벨 숄더 프레스", "sets": 3, "reps": 10, "target_duration_sec": null}]},
        {"day_no": 7, "exercises": []}
      ]
    }
  }
}
```

서버 시작 시 기존 `BodyAnalysis` 테이블에 `analysis_status`, `recognized`,
`failure_reason`, `shoulder_status`, `leg_status`, `analysis_prompt`, `routine_id`,
`routine_json` 컬럼을 추가합니다. `Routine`과 `UserBodyInfo` 테이블이 없는 DB에는
필요한 구조를 생성합니다. 기존 `status` 값도 호환을 위해 함께 갱신됩니다.

업로드 사진은 `body_analysis_uploads/latest/`에 면별 최신 파일로 덮어씁니다.
분석 중에는 요청별 임시 파일을 사용해 동시에 들어온 요청의 사진이 섞이지 않게
하고, 분석이 끝나면 임시 파일을 삭제합니다. 이전 방식으로 만들어진 요청별
사진 폴더는 진행 중인 분석이 없을 때 정리됩니다.

API 문서: `http://127.0.0.1:8000/docs`

## JWT 검증 테스트
로그인 응답의 access_token 값을 TOKEN에 넣으세요.
```bash
curl -i http://127.0.0.1:8000/api/me
# 토큰 없이 요청하면 401

curl -i http://127.0.0.1:8000/api/me \
  -H "Authorization: Bearer TOKEN"
# 유효한 토큰이면 200, 사용자 정보 반환
```

## DB 확인
```bash
python - <<'PY'
import sqlite3
c = sqlite3.connect("users.db")
print("User count:", c.execute("SELECT COUNT(*) FROM UserInfo").fetchone()[0])
for row in c.execute("""
SELECT user_id, username, email_address, nickname, gender, age, height, weight, created_at
FROM UserInfo ORDER BY user_id
"""):
    print(row)
c.close()
PY
```

<!-- integration-standards-20261025 -->

## 통합 구조와 책임

저장소 루트에서 `python -m Server.AIFitService.main` 또는 `python -m uvicorn Server.AIFitService.main:app --host 127.0.0.1 --port 8000`으로 실행합니다. 파일 단위 실행과 패키지 절대 경로가 섞여 있던 import를 루트 기준으로 통일했습니다. `config/settings.py`는 비어 있던 설정 계약을 복구한 곳이며 `.env.example`의 값을 실제 프로세스 환경으로 공급합니다. `.env.example`을 복사하는 것만으로 자동 로드되지는 않습니다.

`AIFIT_JWT_SECRET`은 최소 32바이트입니다. 설정 파일은 키를 생성하거나 개발용 키를 기본값으로 사용하지 않습니다. `AIFIT_DB_PATH`가 없으면 Server/AIFitDB/aifit.db를 사용합니다. `AIFIT_HOST`, `AIFIT_PORT`, `AIFIT_TOKEN_EXPIRE_MINUTES`를 우선하며 APP_HOST/APP_PORT/JWT_SECRET 등 비접두 호환 이름도 허용합니다. 외부 사진 분석은 OPENAI_API_KEY와 OPENAI_MODEL을 명시해야 합니다. 빈 모델 이름을 임의 모델로 대체하지 않습니다.

lifespan은 설정 검증 → 기본 스키마 생성 → 분석/운동 부가 스키마 초기화 → 앱별 영상 버퍼 생성 → 선택적 AI 영상 수신 순서입니다. 종료 시 수신 태스크를 취소하고 기다립니다. import만으로 DB·네트워크를 열지 않습니다. 기존 DB 스키마를 통째로 삭제하지 않고 CREATE IF NOT EXISTS와 기존 추가 컬럼 마이그레이션을 유지합니다.

HTTP 라우터는 전송·인증과 업무 흐름 조정을 담당합니다. domain/routine.py는 목표 검증과 구버전 루틴 정규화를 담당하고 HTTP 상태 코드에 의존하지 않습니다. RoutineError의 missing/invalid를 라우터가 404/500으로 매핑합니다. 시간 목표의 NaN·무한대·bool을 거부해 잘못된 목표가 AI 서버로 전달되는 것을 막습니다.

AIGateway는 외부 요청 포트이고 HTTPAIGateway는 HTTPX 구현입니다. URL·키·클라이언트를 생성 시 전달합니다. 테스트에서는 MockTransport로 실패를 재현합니다. start/finish 재전송의 중복 실행 위험 때문에 어댑터는 상태 변경 요청을 자동 재시도하지 않습니다. 두 서버의 상태와 SQLite 커밋은 하나의 원자적 트랜잭션이 아니므로 라우터의 보상 처리가 필요하고, 프로세스 장애까지 완전히 복구하는 outbox는 아직 없습니다.

사용자 CRUD는 user_repository가 SQL을 소유합니다. get_connection은 연결마다 외래 키를 켜고 성공 시 커밋·실패 시 롤백한 뒤 반드시 닫습니다. sqlite3.Connection의 기본 with만으로는 연결이 닫히지 않으므로 이 컨텍스트를 사용합니다. 커서와 연결을 블록 밖에서 재사용하지 않습니다. 회원가입 중복은 사전 검사와 DB UNIQUE 제약을 함께 사용해 경합 시에도 409로 처리합니다.

LatestFrameBuffer는 앱마다 새로 생성하고 최신 JPEG 하나만 보관합니다. 서버 재시작과 다중 Uvicorn worker 사이에는 메모리가 공유되지 않습니다. JPEGFrameParser는 shared 패키지의 순수 바이트 파서입니다. 상태 로그에는 비밀번호·토큰·외부 분석 응답 본문을 넣지 않습니다.

`tests/integration/test_backend.py`, `test_streaming.py`는 임시 DB·가짜 전송으로 실행합니다. 실제 사용자 DB, 외부 사진 API, 카메라를 사용하지 않습니다. 테스트 통과는 운영 인증 키 설정이나 실제 사진 분석 결과의 정확성을 대신하지 않습니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [adapters/ai_gateway.py](adapters/ai_gateway.py) | AI 서버의 HTTP 통신을 업무 흐름에서 분리한 어댑터입니다. | `GatewayError`, `AIGateway`, `HTTPAIGateway` |
| [adapters/frame_buffer.py](adapters/frame_buffer.py) | 표시용 최신 프레임 한 개를 보관하는 비동기 공유 객체입니다. | `LatestFrameBuffer` |
| [config/settings.py](config/settings.py) | 환경 변수에서 백엔드 설정을 읽습니다. | `validate_settings` |
| [database/connection.py](database/connection.py) | SQLite 연결의 트랜잭션과 수명을 함께 관리하는 영속성 어댑터입니다. | `get_connection`, `init_db` |
| [domain/__init__.py](domain/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [domain/exercise_names.py](domain/exercise_names.py) | 화면 표시 이름을 시스템 운동 식별자로 변환합니다. 외부 I/O는 수행하지 않습니다. | `canonical_exercise_name`, `canonicalize_routine_exercises` |
| [domain/routine.py](domain/routine.py) | 운동 루틴의 순수 업무 규칙입니다. HTTP 상태 결정은 라우터가 담당합니다. | `RoutineError`, `day_exercises` |
| [main.py](main.py) | 웹 라우터와 앱 수명을 조립하는 백엔드 진입점입니다. 설정 검증과 스키마 초기화 후 선택적 영상 수신 작업을 만들고 종료 시 취소·회수합니다. | `lifespan`, `log_gui_api_result`, `health` |
| [models/auth_models.py](models/auth_models.py) | 회원가입·로그인·프로필 API의 입력과 출력 DTO를 정의합니다. Pydantic 검증은 전송 형식의 유효성 검사이며 인증과 영속성은 서비스가 담당합니다. | `RegisterRequest`, `LoginRequest`, `ProfileUpdateRequest`, `PasswordChangeRequest`, `DeleteAccountRequest` |
| [repositories/user_repository.py](repositories/user_repository.py) | 사용자 조회·등록·수정·연관 데이터 삭제 SQL을 모읍니다. 매 호출은 연결 컨텍스트 안에서 실행하며 외래 키와 트랜잭션을 적용합니다. | `find_by_username`, `find_public_by_id`, `username_exists`, `email_exists`, `email_exists_for_other_user`, `create_user`, `get_password_hash`, `update_profile`, `update_password`, `delete_user_and_related_data` |
| [routers/auth_router.py](routers/auth_router.py) | 회원가입·로그인 HTTP 요청을 인증 서비스로 전달하고 업무 오류를 상태 코드로 변환합니다. 비밀번호 해시 구현을 라우터에 복제하지 않습니다. | `register`, `login` |
| [routers/body_analysis_router.py](routers/body_analysis_router.py) | 사진 수신·백그라운드 분석·DB 저장·조회 API를 연결합니다. 사진은 외부 분석 제공자로 전송되며 생성 루틴은 업무 규칙 검증을 통과한 뒤 저장합니다. | `init_body_analysis_db`, `cleanup_old_uploads`, `run_body_analysis`, `upload_body_analysis`, `get_body_analysis` |
| [routers/exercise_report_router.py](routers/exercise_report_router.py) | 인증 사용자의 완료 세션을 날짜별로 조회해 운동 리포트 응답을 만듭니다. 다른 사용자의 결과가 섞이지 않도록 조회 조건에 사용자 ID를 포함합니다. | `get_exercise_report` |
| [routers/profile_router.py](routers/profile_router.py) | JWT로 확인한 사용자의 프로필·비밀번호·계정을 관리합니다. 요청 본문의 임의 사용자 ID 대신 인증 의존성이 반환한 사용자를 기준으로 처리합니다. | `get_my_profile`, `update_my_profile`, `change_my_password`, `delete_my_account` |
| [routers/video_stream_router.py](routers/video_stream_router.py) | 내부 키로 수신한 영상 또는 AI 스트림 영상을 앱별 최신 프레임 버퍼에 게시합니다. GUI에는 JWT로 보호된 MJPEG 스트림을 제공합니다. | `receive_ai_stream`, `receive_frame`, `stream_frames`, `video_status` |
| [routers/workout_router.py](routers/workout_router.py) | GUI 명령, DB 세션 상태, AI 서버의 세션 명령을 조정합니다. 루틴 규칙은 domain, HTTP 전송은 adapters에 위임하며 두 서버 간 보상 처리와 실패 응답을 담당합니다. | `ExerciseRecordInput`, `init_workout_db`, `start_workout`, `pause_workout`, `resume_or_advance_workout`, `get_live_workout`, `receive_exercise_record`, `finish_workout` |
| [services/auth_service.py](services/auth_service.py) | 사용자 등록과 로그인을 처리합니다. PBKDF2 비밀번호 해시 검증 후 JWT를 발급하고 SQL은 사용자 저장소에 위임합니다. | `AuthError`, `hash_password`, `verify_password`, `register_user`, `login_user` |
| [services/body_analysis_service.py](services/body_analysis_service.py) | 네 방향 사진을 외부 Responses API 입력으로 변환하고 구조화된 응답을 검증합니다. 외부 모델 이름은 환경 변수로 지정하며 사진으로 의료 진단을 확정하지 않습니다. | `analyze_body_photos` |
| [services/exercise_names.py](services/exercise_names.py) | 기존 서비스 import 경로를 유지하는 운동 식별자 호환 어댑터입니다. | 패키지 경계/선언 |
| [services/token_service.py](services/token_service.py) | Bearer JWT의 서명·발급 시각·만료와 사용자 존재를 검증합니다. 프레임워크 의존성으로 보호 API에 인증 사용자를 전달합니다. | `get_current_user` |
