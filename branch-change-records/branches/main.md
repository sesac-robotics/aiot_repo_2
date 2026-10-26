# main 기준선과 공통 수정 기록

[전체 기록으로 돌아가기](../README.md) · [적용 규칙과 한계](../rules-and-evidence.md)

## 가져온 방식

새 worktree와 `feature/team-integration-standards`를 origin/main의 고정 커밋 C [`2c4636c`](https://github.com/sesac-robotics/aiot_repo_2/commit/2c4636cf811430f5aa0167ffee07ff36bd0466a1)에서 시작했습니다. main을 나중에 별도 merge한 것은 아닙니다. 따라서 이 기록에는 원본 기능 브랜치처럼 두 번째 부모를 가진 merge G가 없습니다. 이후 원본 기능 브랜치 8개를 순서대로 merge해 M [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)을 만들고, 직접 수정 R [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)로 진행했습니다.

main 스냅샷에는 15개 파일이 있었습니다. 빈 .ino·main.py·README 자리표시자가 포함되어 있었으므로 같은 이름의 기능이 이미 완성됐다는 근거로 사용하지 않습니다. Git의 유사도 판정은 빈 파일을 관계없는 빈 패키지 파일의 rename으로 표시할 수 있습니다. 이 기록의 파일 목록은 `--no-renames` 기준입니다.

## main에 있던 파일과 통합 결과

| main의 경로 | 통합 후 상태와 직접 수정 해석 |
|---|---|
| [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/.gitignore) | 여러 원본의 제외 규칙을 통합하고 로컬 자격 증명·DB·실행 산출물 제외와 테스트 추적을 정리했습니다. |
| [Device/CoachingController/CoachingController.ino](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Device/CoachingController/CoachingController.ino) | 빈 구현을 유지합니다. 제어 프로토콜·LED·부저를 새로 구현하지 않았습니다. |
| [Device/CoachingController/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Device/CoachingController/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [Device/ImageStreamer/ImageStreamer.ino](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Device/ImageStreamer/ImageStreamer.ino) | main에서는 빈 자리표시자였습니다. UI 브랜치가 CameraWebServer.ino를 제공했고 Codex가 실제 스케치 이름을 이 경로로 맞췄습니다. |
| [Device/ImageStreamer/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Device/ImageStreamer/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [GUI/AIFitAdmin/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/GUI/AIFitAdmin/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [GUI/AIFitAdmin/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/GUI/AIFitAdmin/main.py) | 원본 파일과 각 기능 브랜치의 변경을 인계받았습니다. 같은 파일 경로만으로 새 구현의 저자를 판단하지 않습니다. |
| [GUI/FitHelper/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/GUI/FitHelper/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [GUI/FitHelper/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/GUI/FitHelper/main.py) | 원본 기능 브랜치가 구현을 제공한 뒤 Codex가 진입점 조립과 import를 수정했습니다. 해당 기능 브랜치 기록에서 구분합니다. |
| [README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [Server/AIFitDB/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIFitDB/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [Server/AIFitService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIFitService/README.md) | 원본 브랜치의 설명을 먼저 반영하고 통합 후 해당 구성요소의 역할·실행·제한을 보강했습니다. |
| [Server/AIFitService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIFitService/main.py) | 원본 기능 브랜치가 구현을 제공한 뒤 Codex가 진입점 조립과 import를 수정했습니다. 해당 기능 브랜치 기록에서 구분합니다. |
| [Server/AIService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIService/README.md) | 원본 AI 서비스 브랜치의 패키지 구조 변화를 먼저 반영했습니다. 최종 README는 통합 후 설명으로 다시 추가됐으며 main.py는 app/main.py 경로를 사용합니다. |
| [Server/AIService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIService/main.py) | 원본 AI 서비스 브랜치의 패키지 구조 변화를 먼저 반영했습니다. 최종 README는 통합 후 설명으로 다시 추가됐으며 main.py는 app/main.py 경로를 사용합니다. |

## 통합 후 공통 변경과 규칙

| 수정 | 원본 통합 M | 직접 수정 R | 원문 근거 |
|---|---|---|---|
| 코드 서식 | 구성요소별 소스 서식이 혼재했습니다. | 루트 Black 80열·isort와 개발 도구 버전을 설정했습니다. | [PY-02/PY-03](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) |
| 책임 검사 | 공통 경계 검사 진입점이 없었습니다. | tools/check_source_contracts.py로 파싱·한국어 역할 설명·금지 import·순수 상태 의존을 검사합니다. | [ARC-01/ARC-02](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) |
| 타입 검사 | 원본 구성요소별 검사 설정이 분산되어 있었습니다. | 순수 도메인 4개와 AI 세션 5개 파일을 mypy로 검사합니다. 전체 저장소 엄격 타입 검사는 아닙니다. | [PY-04와 검사·리뷰 범위](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) |
| CI | 이번 통합용 공통 실행이 없었습니다. | Python 3.12의 서식·경계·타입·백엔드/전송/세션 23개 검사와 C++17 상태 시험을 구성했습니다. | [GIT-05](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580), [주석 규칙 §2](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834) |
| 설명 | 기존 브랜치별 README가 있었습니다. | 구성요소별 실행 경로·상태 소유자·모델 역할과 소스 목록을 연결했습니다. | [PY-09와 주석 규칙 §2/§3](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834) |
| 제외 규칙 | 브랜치별 DB·테스트·모델 제외 규칙이 달랐습니다. | 로컬 파일 제외를 유지하고 테스트 경로 전체 제외를 제거했습니다. 이미 원본 이력에 있던 동결 모델 파일을 삭제하지 않았습니다. | [GIT-07](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) |

## 적용 한계

.clang-format과 .editorconfig의 C/C++ 들여쓰기는 2칸이며 CPP-10 원문은 4칸입니다. 원문 일치를 완료했다고 기록하지 않습니다. 담당자 승인 또는 설정 정정이 필요한 상태로 남깁니다. 소스·서식 설명이 같은 기능 커밋에 포함된 경우도 있어 GIT-03의 완전한 분리로 평가하지 않습니다.

기존 원본 작성자의 커밋은 merge 이력에 남아 있습니다. 새 코드 변경은 M→R의 10개 직접 수정 커밋으로 추적합니다. 새 브랜치만 push하고 Draft PR #4를 만들었으며 main에는 병합하지 않았습니다. 기존 작업 디렉터리의 미커밋 변경·고정 스쿼트 태그·모델/라벨/분할은 보존했습니다.
