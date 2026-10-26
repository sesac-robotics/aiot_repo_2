# AIFit → AI Server 내부 세션 HTTP 규격

## 연결 및 인증

- Base URL: 배포한 AI Server 주소(로컬 기본 `http://localhost:8000`).
- 모든 내부 요청에 `X-Internal-API-Key: <공유 비밀키>` Header를 보냅니다.
- AI Server의 환경변수 또는 `.env`에 `INTERNAL_API_KEY`를 설정합니다. 실제 키는 코드나 저장소에 넣지 않습니다.
- AIFit에도 동일한 값을 설정합니다. 키는 서버 시작 시 읽으므로 변경 시 재시작합니다.
- 키 누락/불일치: 401. 서버 키 미설정/공백: 503. 내부 API는 차단되고 기존 영상 API는 유지됩니다.
- 운동 시작 Body는 `Content-Type: application/json`입니다. 그 외 요청은 Body가 필요 없습니다.
- 인메모리 관리자 하나를 `app.state.session_manager`에 생성하므로 `--workers 1`을 사용합니다.
- 향후 PoseStreamService에는 이 인스턴스를 주입합니다. 이번 변경에서는 추론 워커를 수정하지 않았습니다.

## 엔드포인트

성공은 모두 **200 OK**, JSON Snapshot을 반환합니다.

| Method | 경로 | 동작 |
|---|---|---|
| POST | `/internal/sessions/{session_id}/start` | 세션 시작 |
| POST | `/internal/sessions/{session_id}/exercises/{exercise_id}/start` | 운동 시작 |
| POST | `/internal/sessions/{session_id}/pause` | 세션 일시정지 |
| POST | `/internal/sessions/{session_id}/resume` | 세션 재개 |
| POST | `/internal/sessions/{session_id}/exercises/{exercise_id}/finish` | 현재 운동 종료 |
| POST | `/internal/sessions/{session_id}/finish` | 세션 종료 |
| GET | `/internal/sessions/{session_id}/live` | 현재/마지막 종료 세션 Snapshot 조회 |

`session_id`, `exercise_id`는 AIFit이 발급하는 양의 정수입니다. AI Server는 ID를 생성하거나
DB 존재 여부를 조회하지 않습니다. 운동 종료는 현재 운동 ID와 일치해야 합니다.
관리자에서 보관하지 않는 세션/운동에 대한 요청은 404입니다.

운동 시작 Body:

```json
{"exercise_name":"squat","order_no":1,"target_reps":10}
```

- `exercise_name`: 공백을 제외한 비어 있지 않은 문자열.
- `order_no`: 1 이상의 정수. 문자열/boolean/실수는 허용하지 않습니다.
- `target_reps`: 생략/null 또는 1 이상의 정수. 목표 도달 시 자동 종료하지 않습니다.
- 정의하지 않은 추가 필드, 잘못된 JSON, 필수 필드 누락은 400입니다.
- 기존 UUID 기반 `app/schemas/exercise.py`는 이 API에 사용하지 않습니다.

## 성공 응답 예시

운동 시작 직후의 예시입니다. 시각/시간 값은 요청 시점에 따라 달라집니다.

```json
{
  "session": {
    "session_id": 101,
    "session_status": "running",
    "started_at": "2026-10-21T05:00:00+00:00",
    "ended_at": null,
    "paused_at": null,
    "paused_duration_sec": 0.0,
    "active_duration_sec": 0.25
  },
  "exercise": {
    "exercise_id": 201,
    "exercise_name": "squat",
    "order_no": 1,
    "target_reps": 10,
    "rep_count": 0,
    "exercise_status": "running"
  }
}
```

운동 시작 전 `exercise`는 null입니다. 종료한 운동은 다음 운동 시작 전까지 Snapshot에
남으며 `exercise_status`는 `finished`입니다. Pause는 세션 상태로 표시하므로 현재 운동의
상태는 running을 유지합니다. UTC 시각은 ISO 8601, 시간은 소수 초입니다.
활동 시간은 운동 사이 대기를 포함한 세션 running 시간이며 Pause 중에는 증가하지 않습니다.

## 상태·오류·재시도

- 활성 세션이 있는데 같은/다른 ID로 start: 409.
- 중복 pause/resume: 200, 기존 정책에 따라 시간 중복 계산 없이 처리.
- 중복 운동 시작/종료 또는 세션 종료: 409. 종료 결과를 두 번 생성하지 않습니다.
- 운동 진행 중 세션 종료: 409. 운동을 먼저 종료해야 합니다.
- Pause 중 새 운동 시작: 409. 현재 운동 종료와 세션 종료는 허용합니다.
- 종료된 세션 상태 변경: 409. 현재 종료 세션 live 조회는 200.
- 새 세션 시작 후 이전 세션 조회: 404. 과거 이력을 보관하지 않습니다.
- 종료한 세션 ID 및 현재 세션에서 사용한 운동 ID 재사용은 기존 관리자 정책대로 거부합니다.
- 응답 유실 후 start/finish 재전송은 409가 될 수 있습니다. live로 현재 상태를 확인하세요.
  이 API는 DB 저장이나 결과 전달의 exactly-once를 보장하지 않습니다.
- 서버 재시작 시 모든 상태가 사라집니다. 단일 프로세스 전제이며 여러 인스턴스 간 공유하지 않습니다.

오류 형식은 공통으로 `{"detail":"설명"}`입니다.

| HTTP | 의미 |
|---|---|
| 400 | 경로/Body/도메인 입력 오류 |
| 401 | 키 누락 또는 불일치 |
| 404 | 현재 보관한 세션 또는 운동과 대상 ID 불일치/대상 없음 |
| 409 | 중복 요청 또는 허용되지 않은 상태 전이 |
| 503 | AI Server 내부 API 키 미설정 |

ID 검증과 변경은 관리자의 동일 RLock 안에서 처리합니다. 세션 교체와 요청이 경합해도
이전 ID를 검증한 요청이 새 세션을 변경하지 않도록 합니다. 이 인증·400 변환은 내부
세션 라우터에 한정되며 기존 `/stream`, `/pose/latest`, `/health`, `/metrics` 계약은 유지합니다.

## curl 순서 테스트

프로젝트 디렉터리에서 키를 환경변수로 입력합니다. `.env`만 사용하는 경우 curl을 실행하는
셸에도 같은 키를 설정해야 합니다.

```bash
read -rs -p 'Internal API key: ' INTERNAL_API_KEY
export INTERNAL_API_KEY
# 기존 GPU/모델/카메라 설정을 유지하고 서버를 --workers 1로 실행합니다.

BASE_URL=http://localhost:8000
curl -sS -X POST "$BASE_URL/internal/sessions/101/start" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"

curl -sS -X POST "$BASE_URL/internal/sessions/101/exercises/201/start" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"exercise_name":"squat","order_no":1,"target_reps":10}'

curl -sS -X POST "$BASE_URL/internal/sessions/101/pause" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"
curl -sS -X POST "$BASE_URL/internal/sessions/101/resume" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"
curl -sS "$BASE_URL/internal/sessions/101/live" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"

curl -sS -X POST "$BASE_URL/internal/sessions/101/exercises/201/finish" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"
curl -sS -X POST "$BASE_URL/internal/sessions/101/exercises/202/start" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"exercise_name":"push_up","order_no":2}'
curl -sS -X POST "$BASE_URL/internal/sessions/101/exercises/202/finish" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"
curl -sS -X POST "$BASE_URL/internal/sessions/101/finish" \
  -H "X-Internal-API-Key: $INTERNAL_API_KEY"
```

## 테스트

FastAPI/Pydantic 및 TestClient 의존성만 사용하며 카메라/GPU를 호출하지 않습니다.

```bash
python -m unittest discover -s tests -p 'test_*session*.py' -v
```

API 12개와 기존 SessionManager 18개, 총 30개가 통과했습니다.
`tests/`는 기존 .gitignore의 제외 대상이므로 커밋에 포함하려면 명시적으로 추가해야 합니다.
