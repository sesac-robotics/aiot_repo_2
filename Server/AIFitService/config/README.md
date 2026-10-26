# 설정 관리

이 폴더는 AIFit Server가 실행될 환경과 외부 서비스 연결에 필요한 설정을 관리합니다. 환경에 따라 달라지는 값을 코드와 분리해 개발·운영 환경에서 같은 서버 코드를 사용할 수 있도록 구성했습니다.

## `settings.py`

다음 설정을 한곳에서 관리합니다.

- **AI Server 연동:** 서버 주소, 내부 API 인증 키, 영상 스트림 경로
- **데이터베이스:** SQLite 파일 위치와 `AIFIT_DB_PATH` 경로 재정의
- **서버 실행:** 호스트와 포트
- **운동 세션 연결 감시:** GUI heartbeat가 끊긴 세션을 일시정지하기까지의 제한 시간
- **사용자 인증:** JWT 비밀 키, 서명 알고리즘, 토큰 유효 시간
- **체형 분석:** OpenAI API 키, 모델, 사진 분석 지침

## 주요 환경 변수

| 변수 | 용도 | 기본값 또는 조건 |
|---|---|---|
| `AI_SERVER_BASE_URL` | AI Server 주소 | `http://192.168.15.208` |
| `AI_SERVER_API_KEY` | AI Server 내부 API 인증 | 환경 변수 설정 권장 |
| `AIFIT_DB_PATH` | SQLite 데이터베이스 경로 | `AIFitDB/AIFit.db` |
| `AIFIT_HOST`, `AIFIT_PORT` | 서버 바인딩 주소와 포트 | `0.0.0.0`, `8000` |
| `AIFIT_SESSION_HEARTBEAT_TIMEOUT_SECONDS` | 연결 끊김 판단 시간 | 30초, 최소 10초 |
| `AIFIT_JWT_SECRET` | JWT 서명 비밀 키 | **필수** |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | 체형 분석 API 설정 | 모델 기본값 `gpt-4o-mini` |

비밀 키는 소스 코드나 저장소에 넣지 말고 실행 환경에서 제공해야 합니다. `AIFIT_JWT_SECRET`이 설정되지 않으면 서버가 시작되지 않습니다.

## 설정 사용 흐름

`main.py`와 각 서비스·라우터가 이 설정을 가져와 데이터베이스, 인증, OpenAI, AI Server에 연결합니다. 운동 세션 라우터는 heartbeat 제한 시간을 사용해 GUI 연결이 끊긴 세션을 `PAUSED`로 전환합니다.
