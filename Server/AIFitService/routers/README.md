# API 라우터

이 폴더는 GUI와 AI Server가 사용하는 FastAPI 엔드포인트를 기능별로 관리합니다. 각 라우터는 요청을 검증하고, 필요한 사용자 인증 또는 내부 API 키를 확인한 뒤 서비스와 데이터베이스를 연결해 응답합니다.

## 라우터 구성

| 파일 | 담당 기능 | 주요 엔드포인트 |
|---|---|---|
| `auth_router.py` | 회원가입 및 로그인 | `POST /api/auth/register`, `POST /api/auth/login` |
| `profile_router.py` | 내 프로필 조회·수정, 비밀번호 변경, 계정 삭제 | `GET/PUT /api/me`, `POST /api/me/change-password`, `POST /api/me/delete-account` |
| `body_analysis_router.py` | 체형 사진 접수, 분석 상태와 결과 조회 | `POST /body-analysis`, `GET /body-analysis/{body_id}` |
| `exercise_report_router.py` | 날짜별 운동 결과 리포트 조회 | `GET /exercise-report` |
| `workout_router.py` | 운동 세션 생성·재개·일시정지·기록·종료 | 아래 운동 세션 API 참고 |
| `video_stream_router.py` | AI Server 영상 프레임 수신 및 GUI 스트리밍 | `POST /api/video/frame`, `GET /api/video/stream`, `GET /api/video/status` |

## 운동 세션 API

- `POST /api/workout-sessions` — 새 운동 세션을 만들거나 기존 `PAUSED` 세션을 재개합니다.
- `POST /api/workout-sessions/{session_id}/pause` — 진행 중인 세션을 일시정지합니다.
- `POST /api/workout-sessions/{session_id}/start` — 일시정지한 운동을 재개하거나 자동 일시정지 뒤 다음 세트·운동을 시작합니다.
- `POST /api/workout-sessions/{session_id}/heartbeat` — GUI 연결 상태를 기록합니다. 진행 중 `GET .../live` 요청도 heartbeat 시각을 갱신합니다.
- `GET /api/workout-sessions/{session_id}/live` — AI Server의 실시간 운동 상태와 자세·점수·피드백을 GUI에 전달합니다.
- `POST /api/workout-sessions/{session_id}/records` — 내부 API 키를 확인한 뒤 AI Server의 반복별 운동 평가를 저장합니다.
- `POST /api/workout-sessions/{session_id}/finish` — 세션을 완료하고 필요한 운동 기록이 모두 있으면 결과 리포트를 생성합니다.

## 연동 및 접근 제어

사용자 데이터가 필요한 API는 JWT로 로그인 사용자를 확인합니다. AI Server의 내부 콜백은 내부 API 키를 검사합니다. 체형 분석 라우터는 분석 결과로 생성된 운동 루틴도 저장 전에 검증합니다.
