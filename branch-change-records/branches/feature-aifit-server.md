# feature/aifit-server 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/aifit-server` / `origin/feature/aifit-server`
- 가져온 원본 커밋 S: [`4a41665`](https://github.com/sesac-robotics/aiot_repo_2/commit/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc)
- 원본 스냅샷 전체 파일: 31개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`fd3f979`](https://github.com/sesac-robotics/aiot_repo_2/commit/fd3f97901244d18149e0e055845230762088bc38) / 첫 번째 부모 [`6e529b9`](https://github.com/sesac-robotics/aiot_repo_2/commit/6e529b9efebecec6b8f545631ce28e1bb2926e87)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: 백엔드 인증·사진 분석·루틴·운동 조정·영상 중계와 DB 스키마

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

대문자 브랜치 다음에 병합했습니다. 이 브랜치가 제공한 라우터·서비스·저장소 구조를 기반으로 실행 문제를 수정하고 실제 경계에 필요한 도메인과 어댑터를 추가했습니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `fd3f979`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 21개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/6e529b9efebecec6b8f545631ce28e1bb2926e87...fd3f97901244d18149e0e055845230762088bc38)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

초기 독립 GUI 파일 삭제와 대부분의 인증·운동·사진 분석 API 구현은 원본 브랜치 작업입니다. domain/adapters 분리와 아래 수명·설정 변경이 통합 후의 직접 수정입니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 환경 설정 복구 | config/settings.py가 빈 파일이어서 APP_HOST, JWT_SECRET, DB_PATH 등 참조가 해결되지 않았습니다. | 환경 변수 기반 설정, AIFIT_ 접두 이름 및 호환 이름, 키·포트·유효 시간 검증, .env.example을 추가했습니다. 기본 인증 비밀 키를 하드코딩하지 않습니다. | [PY-05 · 외부 입력 검증](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 패키지 import와 시작 수명 | config/database/routers 등의 짧은 경로와 Server.AIFitService 절대 경로가 섞였고 startup/shutdown 핸들러가 분리되어 있었습니다. | 루트 기준 패키지 경로와 lifespan으로 통일했습니다. 설정 검증·기본/부가 스키마·버퍼·수신 태스크의 시작 및 종료를 한 곳에서 조립합니다. | [PY-03 · 절대 import·sys.path 변경 금지](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| SQLite 연결 종료와 빈 DB 시작 | get_connection이 sqlite3.Connection을 반환했습니다. SQLite 자체 with는 커밋/롤백을 해도 연결을 닫지 않습니다. 기본 초기화는 UserInfo만 만들었습니다. | contextmanager가 성공 시 커밋, 실패 시 롤백 후 finally에서 닫습니다. 기본 AIFit.sql과 기존 부가 마이그레이션을 순서대로 적용해 빈 DB 시작을 복구했습니다. SQL 스키마 원문은 수정하지 않았습니다. | [PY-08 · 자원 수명 정리](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 루틴 규칙 분리 | workout_router._day_exercises 안에 JSON 정규화·목표 검증과 HTTPException이 함께 있었습니다. | domain/routine.py로 순수 규칙을 옮겼습니다. RoutineError의 missing/invalid를 라우터가 기존 404/500으로 변환하고 NaN·무한대·bool 목표를 거부합니다. | [ARC-01 · 도메인의 외부 프레임워크 의존 금지](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| 운동 식별자 규칙 | 서비스 모듈 안에 공통 운동 이름 변환 구현이 있었습니다. | domain/exercise_names.py에 규칙을 두고 기존 서비스 경로는 명시적 재내보내기로 유지했습니다. | [ARC-02 · 책임과 의존 방향](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| AI HTTP 포트/어댑터 | 라우터 안에서 HTTPX 요청과 오류·응답 처리를 직접 수행했습니다. | AIGateway/HTTPAIGateway와 GatewayError를 추가했습니다. URL·키·AsyncClient를 주입하고 라우터가 503/502를 결정합니다. 자동 상태 변경 재시도는 하지 않습니다. | [ARC-05 · 명시적 포트와 전달 계약](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| 최신 영상 상태 | 라우터 모듈의 전역 프레임·시퀀스·조건 변수에 상태가 있었습니다. | lifespan마다 LatestFrameBuffer를 생성해 app.state로 전달합니다. 공통 JPEGFrameParser는 크기·청크 경계를 처리합니다. 운동 이벤트 저장소와는 별개입니다. | [ARC-08 · 제한된 영상 버퍼와 별도 완료 이벤트](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| 로그와 설치 목록 | 인증 이벤트와 외부 분석 결과 본문을 print했고 사진 업로드 의존성 선언이 빠져 있었습니다. | 운영 로그를 logging으로 옮기고 토큰·비밀번호·분석 응답 본문을 남기지 않습니다. python-multipart==0.0.32를 requirements에 선언했습니다. | [PY-10 · 민감 본문을 제외한 운영 로그](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |

### 공통으로 적용한 기준

- [PY-02 · 4칸/80열과 도구 설정](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3)
- [PY-03 · import 그룹·경로](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3)
- [PY-09 · 역할·계약 설명](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3)
- [COMMENT · §2/§3 모듈·공개 경계 설명](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834) (참조 버전 2)
- [GIT-01 · 기능 브랜치와 PR](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3)
- [GITNEW · §5 feature/ 이름·§7 영문 커밋](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6947031) (참조 버전 4)

## 원본 관련 파일의 전체 추적표

S의 main 대비 변경 경로와 실제 merge 전달 경로의 합집합입니다. `원본 상태`는 main C→S, `merge 상태`는 G의 첫 번째 부모→G입니다. A는 추가, M은 수정, D는 제거, —는 해당 구간 변화 없음입니다. 유사한 빈 파일을 Git이 자동 rename으로 추정하는 혼동을 피하기 위해 `--no-renames`로 집계했습니다.

원본→전체 통합 칸의 차이는 다른 작성자의 브랜치를 합친 결과를 포함합니다. 마지막 칸만 M→R의 직접 수정입니다. 설명·import·타입을 제외한 AST 동일 판정은 전체 런타임 동등성이나 타입 검증 완료를 뜻하지 않습니다.

| 원본 경로 | 원본 상태 | merge 상태 | S→M | M→R |
|---|---|---|---|---|
| [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/.gitignore) | M | M | 다른 원본 이력 반영 | 설정·의존성 또는 펌웨어 변경 · [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.gitignore) |
| [Server/AIFitDB/AIFit.sql](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitDB/AIFit.sql) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIFitDB/AIFit.sql](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitDB/AIFit.sql) |
| [Server/AIFitService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/README.md) | M | M | 원본과 동일 | README 설명 보강 · [Server/AIFitService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/README.md) |
| [Server/AIFitService/config/settings.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/config/settings.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/config/settings.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/config/settings.py) |
| [Server/AIFitService/database/connection.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/database/connection.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/database/connection.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/database/connection.py) |
| [Server/AIFitService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/main.py) | M | M | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/main.py) |
| [Server/AIFitService/models/auth_models.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/models/auth_models.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/models/auth_models.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/models/auth_models.py) |
| [Server/AIFitService/repositories/user_repository.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/repositories/user_repository.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/repositories/user_repository.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/repositories/user_repository.py) |
| [Server/AIFitService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/requirements.txt) | A | M | 원본과 동일 | 설정·의존성 또는 펌웨어 변경 · [Server/AIFitService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/requirements.txt) |
| [Server/AIFitService/routers/auth_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/auth_router.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/routers/auth_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/auth_router.py) |
| [Server/AIFitService/routers/body_analysis_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/body_analysis_router.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/routers/body_analysis_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/body_analysis_router.py) |
| [Server/AIFitService/routers/exercise_report_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/exercise_report_router.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/routers/exercise_report_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/exercise_report_router.py) |
| [Server/AIFitService/routers/profile_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/profile_router.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/routers/profile_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/profile_router.py) |
| [Server/AIFitService/routers/video_stream_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/video_stream_router.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/routers/video_stream_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/video_stream_router.py) |
| [Server/AIFitService/routers/workout_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/routers/workout_router.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/routers/workout_router.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/routers/workout_router.py) |
| [Server/AIFitService/services/auth_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/services/auth_service.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/services/auth_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/services/auth_service.py) |
| [Server/AIFitService/services/body_analysis_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/services/body_analysis_service.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitService/services/body_analysis_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/services/body_analysis_service.py) |
| [Server/AIFitService/services/exercise_names.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/services/exercise_names.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/services/exercise_names.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/services/exercise_names.py) |
| [Server/AIFitService/services/token_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc/Server/AIFitService/services/token_service.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitService/services/token_service.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/services/token_service.py) |
| `Server/AIFitService/userLogin.py` | — | D | 해당 경로 없음 | 통합 후 수정 없음 |
| `Server/AIFitService/video_gui.py` | — | D | 해당 경로 없음 | 통합 후 수정 없음 |

## 통합 후 새로 추가한 관련 경계와 검사

아래 파일은 원본 S에서 가져온 것으로 귀속하지 않습니다. 통합 완료 M 뒤에 추가한 구현·설명·시험입니다.

| 새 파일 | 변경 종류 | 직접 변경 커밋 |
|---|---|---|
| [Server/AIFitService/.env.example](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/.env.example) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [Server/AIFitService/adapters/ai_gateway.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/adapters/ai_gateway.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [Server/AIFitService/adapters/frame_buffer.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/adapters/frame_buffer.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [Server/AIFitService/domain/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/domain/__init__.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [Server/AIFitService/domain/exercise_names.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/domain/exercise_names.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [Server/AIFitService/domain/routine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/domain/routine.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [shared/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/shared/README.md) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [shared/mjpeg.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/shared/mjpeg.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [tests/integration/test_backend.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tests/integration/test_backend.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [tests/integration/test_streaming.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tests/integration/test_streaming.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) | refactor(server): isolate rules and restore backend contracts | 26 |
| [`8a6773a`](https://github.com/sesac-robotics/aiot_repo_2/commit/8a6773ad65bf7872baf5504eb97eb94f7afe40f8) | chore(tooling): enforce formatting and architecture contracts | 1 |
| [`aa4c9c2`](https://github.com/sesac-robotics/aiot_repo_2/commit/aa4c9c20b719e303c56f31c31e7ffa117b4796c5) | fix(server): declare multipart upload dependency | 1 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

tests/integration/test_backend.py(임시 DB·인증·AI 시작 실패 보상), test_streaming.py(청크·크기·앱별 상태), 순수 도메인 mypy 및 최종 CI의 백엔드/세션/전송 23개 검사에 연결됩니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

운동 라우터의 조정 SQL은 전부 Repository로 옮기지 않았습니다. 분산 변경은 보상 처리이며 durable outbox가 없습니다. ARC-09의 단일 DB Writer를 전체 서비스에 구현한 것은 아닙니다. 전체 공개 함수의 타입/구체 예외 정비도 완료하지 않았습니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.
