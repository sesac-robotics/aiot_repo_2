# aiot_repo_2
AIoT 프로젝트 2팀. AI 기반 운동 자세 분석 및 운동 피드백 시스템

## 자세 오류 분류 및 ESP32-CAM 테스트

5종목 라벨링·CNN/GRU 학습 결과·실시간 카메라 테스트·추론 API는 [Server/PostureAnalysis](Server/PostureAnalysis/README.md)에 정리했습니다. 원본 운동 영상은 별도로 확보해야 하며, 어깨 상세 라벨과 실시간 정확도는 추가 검증이 필요합니다.

<!-- integration-standards-20261025 -->

## 통합 브랜치의 구조와 검증 범위

이 설명은 `feature/team-integration-standards`의 소스 구조를 기준으로 합니다. 통합 시 확인한 원격 브랜치 9개의 최신 커밋을 모두 이력에 포함합니다. 기존 작업 디렉터리와 원격 main에는 변경을 적용하지 않습니다. 기존 모델·라벨·분할·실험 보고는 그대로 보존하며 이번 작업에서 새 성능을 측정하거나 운영 서비스에 배포하지 않습니다.

### 구성요소 경계

| 구성요소 | 소유하는 책임 | 다른 부분과의 연결 | 구현 설명 |
|---|---|---|---|
| GUI/FitHelper | 사용자 입력·화면·인증 세션·비동기 요청 | 백엔드 HTTP, JWT, MJPEG | [GUI](GUI/FitHelper/README.md) |
| Server/AIFitService | 인증·루틴·영속 세션·리포트 | SQLite, AI 내부 HTTP, 외부 사진 분석 API | [백엔드](Server/AIFitService/README.md) |
| Server/AIService | 관절 추론·운동 완료 이벤트·실시간 세션 | 카메라, 운동 전략, 자세 분류기 | [AI 서비스](Server/AIService/README.md) |
| Server/ExerciseEvaluation | 기존 교정·점수의 선택/통합 실행·개인 기록 | 생성자 주입 포트, 원본 모델·카메라·파일 어댑터 | [실행·책임 경계](Server/ExerciseEvaluation/README.md) |
| Server/PostureAnalysis | 제공자 데이터 처리·연구 학습·저장 결과 검토 | 원본·관절 캐시·동결 실험 파일 | [연구 코드](Server/PostureAnalysis/README.md) |
| Server/AIFitDB | 관계·외래 키·반복 중복 제약 | 백엔드 저장 어댑터 | [DB](Server/AIFitDB/README.md) |
| Device/ImageStreamer | 카메라 초기화·영상 HTTP 제공 | ESP32 보드·Wi-Fi | [장치](Device/ImageStreamer/README.md) |
| Device/CoachingController | 제어기 자리표시자 | 현재 `.ino` 구현 없음 | [제어기](Device/CoachingController/README.md) |
| shared | 프레임 경계 바이트 처리 | GUI·백엔드가 같은 순수 파서 사용 | [공통 코드](shared/README.md) |
| tools | 유선 연결·파일 공유 시험 도구 | OS 명령·SSH/HTTP 어댑터 | [도구](tools/README.md) |

### 경량 DDD와 패턴 적용

DDD는 폴더 수를 늘리는 규칙이 아니라 **업무 용어와 상태의 소유자를 일치시키는 설계**입니다. Session은 한 번의 운동 세션, Exercise는 세션 안의 종목, Rep는 완결된 한 반복, Hold는 연속 유지 시간, Verdict는 관측 가능한 입력에 대한 판정입니다. 화면 갱신·영상 프레임 도착·HTTP 조회는 Rep가 아닙니다. 관절 미관측은 잘못된 자세의 확정 근거가 아닙니다.

백엔드의 `domain/routine.py`는 JSON에서 루틴 목표를 검증하지만 HTTP 상태를 정하지 않습니다. 라우터가 업무 오류를 기존 404/500 응답으로 바꿉니다. `HTTPAIGateway`는 외부 호출을 맡고, `user_repository`는 SQL을 맡습니다. SQL 전체를 무조건 Repository 클래스로 바꾸지 않습니다. 한 라우터의 트랜잭션 단위인 운동 조정 SQL은 가까운 위치에 유지합니다. 분산 서버 상태 변경은 하나의 DB 트랜잭션이 아니며 실패 시 보상 처리가 필요합니다.

AI 계층의 `SessionManager`는 세션 상태와 중복 완료 이벤트를 소유합니다. 운동 이름·유형은 모델 없는 `session/catalog.py`에서 정의하며 구체 운동 분석기는 `analysis/registry.py`가 생성합니다. Strategy는 종목별 analyze/reset 인터페이스, Factory는 이름에 맞는 새 분석기 생성, Adapter는 외부 모델·HTTP·영상 변환에 적용합니다. Qt Signal/Slot은 작업 결과를 화면 스레드로 전달하는 Observer 역할입니다. MCU 연결은 enum 기반 State로 분리합니다. Singleton이나 형식적인 빈 계층은 추가하지 않습니다.

### 코딩 기준의 실제 적용

Python은 4칸 들여쓰기와 Black 80열·isort를 사용합니다. 공개 경계에는 타입과 실패 계약을 명시하고 순수 도메인은 mypy로 검사합니다. 동적으로 생성되는 Qt 위젯과 연구용 스크립트까지 엄격한 타입 검증을 완료했다고 주장하지 않습니다. 모듈 설명은 역할·입력 경로·상태 소유자를 설명하고, 복잡한 공개 메서드는 Args/Returns/Raises를 사용합니다. 코드 한 줄을 번역하는 주석보다 중복 제거·문맥 revision·관측 보류·트랜잭션 경계의 이유를 설명합니다.

네트워크·DB·카메라 작업은 명시적 함수나 앱 수명 안에서 시작합니다. DB 연결은 커밋/롤백 뒤 닫습니다. GUI 로그인·회원가입·프로필 요청은 스레드 풀에서 실행하고 화면 스레드에서 결과를 반영합니다. 영상 버퍼는 최신 한 프레임만 유지하지만 완료 이벤트의 중복 기록은 SessionManager가 따로 관리합니다.

C++의 프로젝트 소유 연결 상태 코드는 C++17, 고정 폭 정수, constexpr, enum class, 헤더 가드와 동적 할당 없는 순수 전이를 사용합니다. Espressif 예제 기반 `app_httpd.cpp`, 보드 핀 목록과 압축 웹 자산은 SDK 호환 코드로 유지합니다. 이를 프로젝트 전체 MISRA 준수나 힙 미사용 펌웨어로 표현하지 않습니다. `nvidia_egl_device.c`는 C 어댑터이며 C++로 변환하지 않습니다.

### 기준 문서

기준의 원문은 팀 Confluence에서 관리합니다. Git에는 이 구성요소 README와 코드 계약을 두며 전체 공용 문서 보관함을 복사하지 않습니다.

- [팀 공통 기준](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3932327)
- [Python 코딩 기준](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597)
- [C++ 코딩 기준](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3670121)
- [설계·모듈화·DDD](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722)
- [Git 협업](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580)
- [주석·Docstring](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834)
- [최신 Git 입문 절차](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6947031)

### 로컬 검증

저장소 루트에서 실행합니다. Python 런타임은 실제 사용하는 구성요소의 requirements와 맞추고 가상 환경을 분리합니다.

```bash
python -m isort --check-only GUI Server Device tools shared tests
python -m black --check GUI Server Device tools shared tests
python -m mypy
MYPYPATH=Server/AIService python -m mypy -p app.session
QT_QPA_PLATFORM=offscreen python -m pytest
python -m unittest discover -s tools -p 'test_*.py'
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I . tests/firmware/test_connection_state.cpp -o /tmp/aifit-state-test
/tmp/aifit-state-test
```

새 브랜치에서 검증하고 Draft PR로 검토합니다. 통합은 기능 완성이나 전체 하드웨어 검증과 같지 않습니다. AI→백엔드 반복 결과 callback, 분산 상태 변경의 완전한 원자성, 실제 카메라·GPU·다중 PC 검증은 별도 항목입니다. 메인 반영 전에는 소유 담당자의 API/DB/하드웨어 검토가 필요합니다.


## 원본 브랜치별 변경 기록

[브랜치 통합·수정 기록](branch-change-records/README.md)에서 원본 8개 기능 브랜치와 main의 고정 커밋, 병합 방식, 통합 후 직접 수정, 적용 규칙 원문과 남은 불일치를 확인할 수 있습니다.


## 최신 자세 평가 작업 — 2026-10-25

`feature/exercise-posture-scoring`에는 기존 준비·복귀 미커밋 작업을 보존하고 `feature/team-integration-standards`의 최신 통합 소스와 규칙 적용 이력을 병합했습니다. 이전 브랜치별 변경 기록은 해당 시점의 고정 커밋 기준으로 유지합니다.

저장소 밖에 있던 코드·라벨·관절 특징·자체 학습 모델·검증 결과는 [최신 작업 스냅샷](checkpoints/workspace-20261025/README.md)에 원래 경로와 SHA-256으로 연결했습니다. [현재 위치 색인](LATEST_WORKSPACE.json)에서 공통 측정 포인터, 별도 4종 연구, 숄더 연구와 저장된 로컬 웹앱을 구분합니다. 과거 스쿼트 보존본·태그·동결 분할은 유지했습니다.

이 게시가 연구 모델의 운영 채택·독립 최종 Test 통과·새 라이브 배포를 뜻하지 않습니다. 원본 데이터셋·영상·검토 이미지·제공자 사전학습 가중치는 별도 보관 목록에 남겼으며, Confluence 공용 문서 본문과 개인 문서 사본은 Git에 넣지 않았습니다.

## 통합 후속 동기화 — 2026-10-25

자세 점수 브랜치의 `68d77539`까지 누락됐던 10개 커밋을 통합 브랜치에 fast-forward로 반영했습니다. 원본 커밋을 cherry-pick으로 재생성하지 않고 기존 이력을 유지했습니다. 원격 전체 브랜치를 확인했으며 다른 팀원의 추가 미반영 커밋은 없었습니다. 이후의 책임 분리·검사·기록 변경은 두 브랜치에 같은 이력으로 반영합니다.

서버 실행·관절 기록 구현을 GUI에서 분리하고 순수 실행 수명, 저장/실행 포트, 기존 교정/카메라 어댑터를 연결했습니다. 기존 GUI 명령과 API·JSON 형식, 스쿼트 동결 모델과 반복 계산은 유지합니다. 팀 설치·실행 절차는 [운동 모드 안내](GUI/FitHelper/MODES.md)를 따릅니다. [브랜치별 후속 기록](branch-change-records/branches/feature-exercise-posture-scoring.md#2026-10-25-후속-동기화와-규칙-적용)에서 원본 고정 커밋·적용 규칙·검증·남은 범위를 확인합니다. Confluence 기준서 원문과 개인 기록은 복사하지 않습니다.
