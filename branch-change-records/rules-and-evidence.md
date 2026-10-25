# 적용 규칙의 원문·버전과 검증 범위

[전체 기록으로 돌아가기](README.md)

## 원문 위치

아래는 이전 통합 구현 작업에서 참조한 2026-10-25 수집본의 문서 버전입니다. 링크가 열리는 사이트의 현재 버전이 이 수집본과 항상 같다고 주장하지 않습니다. 원문 전체를 이 저장소에 복사하지 않고 페이지 ID·참조 버전·절 이름으로 근거를 연결합니다.

| 문서 | Confluence 위치 | 참조 버전 | 이 기록의 사용 범위 |
|---|---|---|---|
| 팀 공통 기준 | [01-team-conventions / 3932327](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3932327) | 6 | 기존 Device/GUI/Server 경계, 공용 문서 위치, 최소 코딩·의존·상태 기준 |
| Python | [00-python-coding-conventions / 3604597](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) | 3 | PY-01~12, 검사·리뷰 |
| C++ | [01-cpp-coding-conventions / 3670121](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3670121) | 3 | CPP-01~12, 직접 작성 코드와 SDK의 검토 범위 |
| 설계·모듈화·DDD | [02-design-and-modularity / 3866722](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) | 3 | ARC-01~10, 패턴과 책임 소유 |
| Git 협업 | [03-git-collaboration / 3899580](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) | 3 | GIT-01~08, 영문 Conventional Commits |
| 주석·docstring | [04-code-comments-and-docstrings / 6979834](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834) | 2 | §2 설명 범위, §3 Google docstring, §4 Doxygen |
| 최신 Git 입문 절차 | [05-git-beginner-guide / 6947031](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6947031) | 4 | §5 feature/ 이름 통일, §7 커밋, §9 PR |

Python·C++·설계·Git의 번호는 원문 번호입니다. COMMENT·GITNEW는 각 문서를 연결하기 위한 이 기록의 표식이며 원문에 그런 규칙 번호가 있는 것은 아닙니다. 주석 문서는 절 이름으로 구체 범위를 명시합니다.

Git 협업의 오래된 GIT-02 예제는 feat/ 등을 사용하지만, 이후 최신 Git 입문 절차의 feature/ 통일 방침을 적용했습니다. 따라서 새 브랜치는 feature/team-integration-standards입니다. 다른 사람의 기존 브랜치 이름은 변경하지 않았습니다.

## 규칙과 실제 수정의 대응

| 규칙 | 실제 적용 경로 | 해석·한계 |
|---|---|---|
| PY-02/PY-03 | [pyproject.toml](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/pyproject.toml), [requirements-dev.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/requirements-dev.txt), Python 소스 | 공통 서식·import 정리입니다. import 위치 변경은 실행 수명에도 영향을 줄 수 있으므로 단순 외관 변경과 같지 않습니다. |
| PY-04 | [Server/AIFitService/domain/routine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/domain/routine.py), [GUI/FitHelper/domain/auth_session.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/domain/auth_session.py), [Server/AIService/app/session/manager.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/session/manager.py) | 핵심 경계 타입을 보강했습니다. 전체 동적 UI·연구 함수에 타입이 완성된 것은 아닙니다. |
| PY-05 | 루틴 입력·설정 검증 | bool/NaN/무한대 같은 외부 입력을 if로 거부합니다. SessionManager의 pause 시각 assert는 내부 상태 전제이며 입력 검증 대신 쓰지 않습니다. |
| PY-07 | GUI AuthSession, 앱별 LatestFrameBuffer, 로컬 진단 필터 | 상태 소유자를 정리했습니다. 프로세스별 _worker_resources 부트스트랩 등 가변 상태가 남아 있어 전체 전역 상태 제거 완료로 표현하지 않습니다. |
| PY-08 | SQLite 컨텍스트, RequestTask, 추출기 finally/Finalize | DB·파일·워커 수명을 정리했습니다. 모든 화면·외부 도구의 자원 수락 시험을 완료한 것은 아닙니다. |
| PY-09/주석 §2·§3 | 모듈 역할, 주요 공개 계약, 구성요소 README | 176개 Python 파일에 한국어 모듈 책임을 연결했습니다. 모든 함수에 완전한 Google docstring을 붙였다는 의미는 아닙니다. |
| PY-10/PY-12 | 인증·사진 분석 logging, 명시적 main/lifespan | 일부 운영 출력을 정리했습니다. CLI 데모 print와 일부 기존 예외 처리·진단 코드는 남아 있습니다. |
| ARC-01/ARC-02 | backend domain, AI session/catalog·projections | 순수 규칙과 프레임워크/모델 의존을 분리했습니다. 애플리케이션의 모든 파일을 동일한 계층 폴더로 재배치하지 않았습니다. |
| ARC-03/ARC-05 | HTTPAIGateway의 클라이언트 주입, 앱 진입점 조립 | 일부 경계의 DI입니다. GUI get_context의 QApplication 조회는 남은 Service Locator 성격의 호환 경로이므로 ARC-03 전체 충족이 아닙니다. |
| ARC-06/ARC-07 | 기존 event_id/revision 중복·지연 검사, 관측 보류 의미 보존 | 기존 반복 소유권을 보강·검증했습니다. correct/incorrect/unknown 전체 집계식을 새로 구현하지 않았습니다. |
| ARC-08 | 제한된 최신 프레임 버퍼·별도 Rep, Qt worker | 영상과 이벤트를 구분합니다. AI→백엔드 반복 결과 callback의 기존 미연결은 남아 있습니다. |
| ARC-09 | SQLite 연결·트랜잭션 경계 | 단일 DB Writer 또는 MCU 단일 DeviceAdapter 전체를 새로 구현하지 않았습니다. |
| CPP-01/CPP-02/CPP-03 | config{}, enum class, uint32_t, constexpr | 직접 작성한 연결 상태와 초기화 부분에 적용했습니다. SDK 전체의 모든 반환값·할당을 재작성하지 않았습니다. |
| CPP-04/CPP-06/CPP-08/CPP-12 | [Device/ImageStreamer/connection_state.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/connection_state.h) | 새 상태 전이에는 동적 할당이 없고 밀리초 제한과 중복 시작 차단이 있습니다. SDK의 힙 사용이나 모든 통신 검증을 제거한 것은 아닙니다. |
| CPP-10/주석 §4 | 헤더 가드·namespace, .clang-format | 헤더/타입 일부 적용입니다. 실제 들여쓰기 2칸은 원문 4칸과 불일치합니다. 일부 새 펌웨어 설명도 영어이며 Doxygen 전체 한국어 정비는 끝나지 않았습니다. |
| GIT-01/04/05/06 | 기능 브랜치·merge 원본 이력·Draft PR #4 | main 직접 push·자가 승인·강제 이력 재작성을 하지 않았습니다. 다른 팀원의 실제 승인 완료와는 다릅니다. |
| GIT-03 | 별도 style 커밋과 기능 커밋 | 일부 기능 변경에 설명·서식도 함께 포함되어 완전한 목적별 분리는 아닙니다. 154b890에는 서버 수정과 스케치 이름 변경도 함께 기록됐습니다. |
| GIT-07/08 | 동결 자산 해시·작은 HTTP Range fixture | 기존 원본 브랜치에 있던 모델·데이터는 보존했습니다. 원래 이력의 대형 파일·라이선스 조건 전체가 이번에 승인됐다는 의미가 아닙니다. |

## 원본 문서 보관 방침과 이번 Markdown의 범위

공통 기준 §2는 프로젝트 공용 문서를 Confluence에서 관리하고 Git docs를 생성하지 않도록 정합니다. 이번에는 사용자가 `feature/team-integration-standards`에서 이전 브랜치별 수정 내용을 MD 파일로 남기도록 직접 요청했습니다. 이 요청 범위에 따라 이 브랜치의 통합·변경 감사 기록을 branch-change-records 아래에 저장합니다. 원문 기준서 전문과 개인 문서 아카이브는 복사하지 않습니다. 이 기록 추가가 팀 문서 보관 방침 전체의 폐지를 의미하지 않습니다.

## 검증 증거의 구분

기준 R은 [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)입니다. [최종 CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)는 새 Python 3.12 환경에서 소스 구조·Black/isort·mypy, 백엔드/전송/세션 23개 및 C++17 상태 시험을 확인합니다. [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 로컬 범위를 함께 기록했습니다.

| 검증 | 이전 구현 작업에서 확인한 결과 | 보장하지 않는 것 |
|---|---|---|
| 루트 pytest | 68개 통과 + 별도 subtests 13개 | 전체 장치·독립 최종 모델 정확도 |
| 유선·공유 도구 | 14개 통과 / 1개 skip | 실제 두 PC 네트워크와 팀 계정 전송 |
| 연구 저장 모델/API | 22개 통과 / 1개 skip | 미사용 독립 최종 Test |
| 데이터 검토 UI | 15개 통과 | 실제 비디오 전수 검수·디코딩 정확도 |
| 합계 | 119개 통과 / 2개 skip, subtests 별도 | 119개의 모델 성능 실험 |
| 코드·타입 | 176개 소스 구조, 순수 경계 mypy 9개, Black/isort | 모든 레거시 경고·타입의 정리 |
| 펌웨어·C | C++17 호스트 상태, ESP32 전체 빌드, C EGL 컴파일 | 장치 업로드·실제 영상·MISRA 인증 |
| 보존 | 모델·라벨·분할·보고 113개 해시 및 스쿼트 고정 커밋 보존 | 새로운 학습·성능 향상 |

이번 후속 작업에서는 고정 커밋·파일 blob·문서 링크·원본/통합/수정 구분을 검증합니다. 실행 소스가 바뀌지 않으므로 이전 모델/기능 시험을 새로 수행한 것으로 보고하지 않습니다.

## 후속 규칙 재확인 — 2026-10-25

자세 점수 작업을 통합하면서 공통 v6·Python v3·설계 v3·Git v3·주석 v2·Git 입문 v4를 원격 원문으로 재확인했습니다. 버전은 위 표와 같습니다. application/domain/adapter, 타입 포트·생성자 DI·실행 수명·저장 실패의 후속 적용은 [해당 브랜치 기록](branches/feature-exercise-posture-scoring.md#2026-10-25-후속-동기화와-규칙-적용)을 따릅니다. 기존 레거시의 타입·호환 전역 상태·동결 모델 자산이 모두 규칙에 맞게 재작성됐다는 의미는 아닙니다. C++ 소스는 이번 범위에서 수정하지 않았습니다.
