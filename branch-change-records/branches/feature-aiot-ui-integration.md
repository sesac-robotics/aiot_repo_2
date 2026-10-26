# feature/aiot-ui-integration 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/aiot-ui-integration` / `origin/feature/aiot-ui-integration`
- 가져온 원본 커밋 S: [`bbe98d3`](https://github.com/sesac-robotics/aiot_repo_2/commit/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84)
- 원본 스냅샷 전체 파일: 53개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`dd5e527`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd5e5274e23a6f9860b3d662bb77bb0cbc43158a) / 첫 번째 부모 [`489065c`](https://github.com/sesac-robotics/aiot_repo_2/commit/489065c176c6eced7a67882f53d223268bd7af83)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: Qt 사용자 화면과 ESP32 카메라 예제

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

GUI·장치 파일을 함께 제공하는 브랜치를 병합했습니다. 기존 .ui 화면과 대부분의 작업자를 유지하며 인증 상태·요청 수명·영상 프레이밍과 장치 연결 부분을 수정했습니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `dd5e527`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 41개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/489065c176c6eced7a67882f53d223268bd7af83...dd5e5274e23a6f9860b3d662bb77bb0cbc43158a)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

원본의 사진 업로드·운동·이력 QThread, Qt 화면 배치, ESP32 HTTP/카메라 예제는 인계받은 구현입니다. 새로 옮긴 비동기 요청은 로그인·회원가입·프로필 경로입니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 로그인 상태 소유 | login_client와 body_analysis_client에 복제 토큰이 있었고 화면이 private 전역 필드를 직접 읽거나 수정했습니다. | ApplicationContext가 AuthSession 하나를 소유합니다. 복사된 프로필을 제공하고 토큰/사용자를 함께 교체하며 revision으로 로그아웃 뒤 늦은 로그인 결과를 차단합니다. | [PY-07 · 변하는 전역 상태 축소](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 화면 요청 실행 | 로그인·회원가입·프로필 요청이 GUI 이벤트 처리 안에서 실행됐습니다. | RequestTask가 QRunnable/스레드 풀에 요청을 제출하고 QObject 슬롯이 GUI 스레드에서 결과를 반영합니다. 제출 중 버튼을 막고 닫힌 로그인 결과를 취소합니다. | [ARC-08 · UI와 Worker 분리](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| 인증 요청 어댑터 | 로그인 화면이 JWT 시험 모듈을 동적 로드해 요청하고 여러 모듈 토큰을 직접 맞췄습니다. | LoginClient가 로그인과 /api/me 검증을 담당합니다. 모두 성공해야 인증 상태를 게시합니다. 이전 JWT 시험 폴더는 삭제하지 않았습니다. | [ARC-05 · 명시적 API 경계](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| 패키지 경로와 주소 | 짧은 형제 import와 sys.path.insert가 섞였고 사설망 기본 주소가 분산되어 있었습니다. | GUI.FitHelper 경로로 통일하고 AIFIT_SERVER_URL 설정과 기본 loopback 주소를 사용합니다. 실행은 python -m GUI.FitHelper.main입니다. | [PY-03 · 안정된 패키지 경로](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 영상 프레임 처리 | GUI의 자체 JPEG 표식 조립은 미완성 프레임 버퍼의 상한이 없었습니다. | shared.JPEGFrameParser로 최대 크기·청크 경계·재동기화를 처리합니다. QImage 복사와 표시 신호는 유지하고 종료 실패를 상태 신호로 전달합니다. | [ARC-08 · 제한된 최신 영상 경로](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3866722) (참조 버전 3) |
| Arduino 진입점과 설정 | 폴더와 다른 CameraWebServer.ino 이름, 하드코딩된 연결 정보, 초기화하지 않은 설정 구조체와 무기한 Wi-Fi 대기 루프가 있었습니다. | ImageStreamer.ino로 이름을 맞추고 로컬 wifi_credentials.h를 분리했습니다. config{}·센서 널 검사·30초 제한 연결 상태를 추가합니다. | [CPP-01 · 초기화와 명시 상태](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3670121) (참조 버전 3) |
| 연결 상태 함수 | Wi-Fi 연결 완료를 setup의 반복 대기로 기다렸습니다. | uint32_t 기반 경과 시간과 enum class의 순수 advance 함수를 loop에서 호출합니다. 성공 후 서버를 한 번만 시작합니다. delay(1)은 SDK 작업에 실행을 양보하기 위한 것으로 남아 있습니다. | [CPP-08 · 제한 시간이 있는 비차단 갱신](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3670121) (참조 버전 3) |
| 펌웨어 상수·타입 | 프로젝트 연결 상태를 독립적으로 시험할 경계가 없었습니다. | connection_state.h에 고정 폭 타입·constexpr·헤더 가드·namespace를 적용하고 host C++17 시험을 추가했습니다. | [CPP-02 · 고정 폭 정수와 단위](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3670121) (참조 버전 3) |

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
| [Device/ImageStreamer/CameraWebServer.ino](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/CameraWebServer.ino) | A | A | 원본과 동일 | 통합 후 제거/이름 변경 |
| [Device/ImageStreamer/ESP32Cam.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/ESP32Cam.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Device/ImageStreamer/ESP32Cam.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/ESP32Cam.py) |
| [Device/ImageStreamer/ImageStreamer.ino](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Device/ImageStreamer/ImageStreamer.ino) | D | D | 해당 경로 없음 | 통합 후 새 파일 · [Device/ImageStreamer/ImageStreamer.ino](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/ImageStreamer.ino) |
| [Device/ImageStreamer/app_httpd.cpp](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/app_httpd.cpp) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/app_httpd.cpp](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/app_httpd.cpp) |
| [Device/ImageStreamer/board_config.h](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/board_config.h) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/board_config.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/board_config.h) |
| [Device/ImageStreamer/camera_index.h](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/camera_index.h) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/camera_index.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/camera_index.h) |
| [Device/ImageStreamer/camera_pins.h](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/camera_pins.h) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/camera_pins.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/camera_pins.h) |
| [Device/ImageStreamer/ci.yml](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/ci.yml) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/ci.yml](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/ci.yml) |
| [Device/ImageStreamer/partitions.csv](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/Device/ImageStreamer/partitions.csv) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Device/ImageStreamer/partitions.csv](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/partitions.csv) |
| [GUI/FitHelper/bodyAnalysisLoading.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/bodyAnalysisLoading.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [GUI/FitHelper/bodyAnalysisLoading.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/bodyAnalysisLoading.py) |
| [GUI/FitHelper/bodyAnalysisLoading.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/bodyAnalysisLoading.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/bodyAnalysisLoading.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/bodyAnalysisLoading.ui) |
| [GUI/FitHelper/bodyAnalysisResult.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/bodyAnalysisResult.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/bodyAnalysisResult.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/bodyAnalysisResult.ui) |
| [GUI/FitHelper/bodyUpload.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/bodyUpload.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/bodyUpload.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/bodyUpload.ui) |
| [GUI/FitHelper/body_analysis_cache.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/body_analysis_cache.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/body_analysis_cache.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/body_analysis_cache.py) |
| [GUI/FitHelper/body_analysis_client.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/body_analysis_client.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/body_analysis_client.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/body_analysis_client.py) |
| [GUI/FitHelper/body_analysis_report.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/body_analysis_report.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [GUI/FitHelper/body_analysis_report.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/body_analysis_report.py) |
| [GUI/FitHelper/body_upload_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/body_upload_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/body_upload_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/body_upload_window.py) |
| [GUI/FitHelper/deleteAccount.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/deleteAccount.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/deleteAccount.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/deleteAccount.ui) |
| [GUI/FitHelper/full_routine.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/full_routine.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/full_routine.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/full_routine.ui) |
| [GUI/FitHelper/history_detail.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/history_detail.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/history_detail.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/history_detail.ui) |
| [GUI/FitHelper/history_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/history_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/history_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/history_window.py) |
| [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/README.md) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/README.md) |
| [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/__init__.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/__init__.py) |
| [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/login.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/login.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/login.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/login.py) |
| [GUI/FitHelper/login_client.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/login_client.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/login_client.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/login_client.py) |
| [GUI/FitHelper/login_user.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/login_user.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/login_user.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/login_user.py) |
| [GUI/FitHelper/login_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/login_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/login_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/login_window.py) |
| [GUI/FitHelper/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/main.py) | M | M | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/main.py) |
| [GUI/FitHelper/mainLoginWindow.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/mainLoginWindow.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/mainLoginWindow.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/mainLoginWindow.ui) |
| [GUI/FitHelper/profileSettings.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/profileSettings.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/profileSettings.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/profileSettings.ui) |
| [GUI/FitHelper/profile_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/profile_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/profile_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/profile_window.py) |
| [GUI/FitHelper/registration.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/registration.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/registration.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/registration.ui) |
| [GUI/FitHelper/registrationComplete.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/registrationComplete.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/registrationComplete.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/registrationComplete.ui) |
| [GUI/FitHelper/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/requirements.txt) | A | A | 원본과 동일 | 설정·의존성 또는 펌웨어 변경 · [GUI/FitHelper/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/requirements.txt) |
| [GUI/FitHelper/routine_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/routine_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/routine_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/routine_window.py) |
| [GUI/FitHelper/video_gui.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/video_gui.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/video_gui.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/video_gui.py) |
| [GUI/FitHelper/video_receiver.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/video_receiver.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/video_receiver.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/video_receiver.py) |
| [GUI/FitHelper/welcomeBack.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/welcomeBack.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/welcomeBack.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/welcomeBack.ui) |
| [GUI/FitHelper/welcome_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/welcome_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/welcome_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/welcome_window.py) |
| [GUI/FitHelper/workout_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/workout_window.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [GUI/FitHelper/workout_window.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/workout_window.py) |
| [GUI/FitHelper/workout_window.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84/GUI/FitHelper/workout_window.ui) | A | A | 원본과 동일 | 통합 후 수정 없음 · [GUI/FitHelper/workout_window.ui](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/workout_window.ui) |

## 통합 후 새로 추가한 관련 경계와 검사

아래 파일은 원본 S에서 가져온 것으로 귀속하지 않습니다. 통합 완료 M 뒤에 추가한 구현·설명·시험입니다.

| 새 파일 | 변경 종류 | 직접 변경 커밋 |
|---|---|---|
| [Device/ImageStreamer/connection_state.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/connection_state.h) | 통합 후 새 파일 | [`08cd565`](https://github.com/sesac-robotics/aiot_repo_2/commit/08cd565b55d90d55f817d8250df8f0909c9172a5) |
| [Device/ImageStreamer/wifi_credentials.example.h](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Device/ImageStreamer/wifi_credentials.example.h) | 통합 후 새 파일 | [`08cd565`](https://github.com/sesac-robotics/aiot_repo_2/commit/08cd565b55d90d55f817d8250df8f0909c9172a5) |
| [GUI/FitHelper/application_context.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/application_context.py) | 통합 후 새 파일 | [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) |
| [GUI/FitHelper/domain/auth_session.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/domain/auth_session.py) | 통합 후 새 파일 | [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) |
| [GUI/FitHelper/request_worker.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/GUI/FitHelper/request_worker.py) | 통합 후 새 파일 | [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) |
| [shared/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/shared/README.md) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [shared/mjpeg.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/shared/mjpeg.py) | 통합 후 새 파일 | [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) |
| [tests/firmware/test_connection_state.cpp](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tests/firmware/test_connection_state.cpp) | 통합 후 새 파일 | [`08cd565`](https://github.com/sesac-robotics/aiot_repo_2/commit/08cd565b55d90d55f817d8250df8f0909c9172a5) |
| [tests/integration/test_gui.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tests/integration/test_gui.py) | 통합 후 새 파일 | [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) |

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) | refactor(server): isolate rules and restore backend contracts | 4 |
| [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) | refactor(gui): centralize auth state and isolate request workers | 23 |
| [`08cd565`](https://github.com/sesac-robotics/aiot_repo_2/commit/08cd565b55d90d55f817d8250df8f0909c9172a5) | fix(firmware): bound connection setup and separate credentials | 5 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

Qt offscreen의 인증 상태·스레드·늦은 응답 검사 4개와 공통 프레임 시험을 확인했습니다. C++17 호스트 상태 시험과 ESP32 코어 3.3.12/CLI 1.5.1 전체 스케치 컴파일이 통과했습니다. 실제 장치 업로드·Wi-Fi·카메라 동작 검증은 수행하지 않았습니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

ARC-03의 순수 생성자 주입이 전체 화면에 적용되지는 않았습니다. get_context()가 QApplication에서 상태를 조회하는 호환 경로가 남아 있습니다. .clang-format의 IndentWidth=2와 SDK 스타일은 CPP-10의 4칸 기준과 다릅니다. 기존 영문 함수 설명/Doxygen 일부, SDK 힙 사용, 전체 MISRA 검증도 남아 있습니다. 이 문서는 그 차이를 승인된 예외로 단정하지 않습니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.
