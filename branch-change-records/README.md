# 원본 브랜치별 통합·수정 기록

이 기록은 팀 저장소의 `feature/team-integration-standards`가 다른 작성자들의 원본 브랜치를 어떻게 가져왔고, 그 뒤 Codex가 무엇을 수정했는지 추적합니다. 문서 기준일은 2026-10-25이며 코드 기준은 고정 커밋 R [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)입니다. 이후 이 Markdown을 추가한 문서 커밋과 코드 수정 커밋을 구분합니다.

## 브랜치별 자식 문서

| 원본 브랜치 | 범위 | 기록 |
|---|---|---|
| main | 새 worktree의 시작 기준선·공통 검사 설정 | [main](branches/main.md) |
| feature/AIFitServer | 초기 백엔드 진입점과 독립 로그인·영상 GUI 시험 코드 | [feature/AIFitServer](branches/feature-AIFitServer.md) |
| feature/aifit-server | 백엔드 인증·사진 분석·루틴·운동 조정·영상 중계와 DB 스키마 | [feature/aifit-server](branches/feature-aifit-server.md) |
| feature/db | SQLite 기본 스키마와 두 데모 스크립트 | [feature/db](branches/feature-db.md) |
| feature/ai-service | AI API·세션·운동 전략·관절 추론·실시간 스트림 | [feature/ai-service](branches/feature-ai-service.md) |
| feature/wired-camera-link | 유선 네트워크 시험·파일 공유 도구 및 과거 AI 스냅샷 | [feature/wired-camera-link](branches/feature-wired-camera-link.md) |
| feature/aiot-ui-integration | Qt 사용자 화면과 ESP32 카메라 예제 | [feature/aiot-ui-integration](branches/feature-aiot-ui-integration.md) |
| feature/exercise-posture-scoring | 구간 채점·특징·학습/분할 감사·모델 번들·회귀 시험 | [feature/exercise-posture-scoring](branches/feature-exercise-posture-scoring.md) |
| feature/posture-error-classification | 연구 데이터 준비·CNN/GRU 추론·저장 실험·검토 UI | [feature/posture-error-classification](branches/feature-posture-error-classification.md) |

[규칙의 원문 위치·번호·참조 버전과 적용 한계](rules-and-evidence.md)는 모든 자식 문서의 공통 근거입니다. 대문자 AIFitServer와 소문자 aifit-server는 서로 다른 원격 참조이므로 별도 기록으로 유지합니다.

## 세 단계로 변경을 구분하는 방법

- **S — 원본 스냅샷:** 실제로 가져온 원격 브랜치의 고정 커밋입니다. 움직이는 브랜치 이름만으로 과거 원본을 설명하지 않습니다.
- **G/M — 병합 단계:** G는 해당 브랜치를 합친 개별 merge, M은 기능 브랜치 8개를 모두 합친 결과입니다. 이 단계에는 다른 작성자의 후속 변경·동일 파일의 병합 결과가 들어 있습니다.
- **R — 직접 수정:** M 이후 통합 브랜치에서 실행 문제·책임·서식·설명을 수정한 결과입니다. S→R 전체 차이를 Codex가 작성한 코드로 귀속하지 않습니다.

원본의 기능을 새로 만든 것, 기존 기능을 유지하며 경계를 정리한 것, 설명/서식만 바꾼 것, 데이터·모델을 그대로 보존한 것을 구분합니다. 같은 파일이 여러 원본에 있으면 자식 문서에도 중복될 수 있으며 파일 수를 더해 개인 기여율로 계산하지 않습니다. 원본 작성자의 실제 역할·책임자를 Git 메타데이터만으로 추정하지 않습니다.

## 가져온 순서와 고정 이력

원격은 git fetch origin --prune으로 갱신했습니다. origin/main에서 별도 worktree를 만든 뒤 원본 참조를 순서대로 merge했습니다. 이 작업은 서로 다른 브랜치에 반복 pull한 방식이 아닙니다. main은 시작점이며 별도 merge 1개로 세지 않습니다.

| 순서 | 가져온 원본 | 원본 S | 해당 merge G |
|---|---|---|---|
| 0 | main | [`2c4636c`](https://github.com/sesac-robotics/aiot_repo_2/commit/2c4636cf811430f5aa0167ffee07ff36bd0466a1) | 시작 기준선 |
| 1 | feature/AIFitServer | [`2d3274e`](https://github.com/sesac-robotics/aiot_repo_2/commit/2d3274e4b03b4b0222879a11016f8869dfae997d) | [`6e529b9`](https://github.com/sesac-robotics/aiot_repo_2/commit/6e529b9efebecec6b8f545631ce28e1bb2926e87) |
| 2 | feature/aifit-server | [`4a41665`](https://github.com/sesac-robotics/aiot_repo_2/commit/4a4166542aaf2ad5a04c64eebbd11fa53339c4fc) | [`fd3f979`](https://github.com/sesac-robotics/aiot_repo_2/commit/fd3f97901244d18149e0e055845230762088bc38) |
| 3 | feature/db | [`2856af6`](https://github.com/sesac-robotics/aiot_repo_2/commit/2856af67cb7791b332a204da0db65622f4e2bf75) | [`61dcdef`](https://github.com/sesac-robotics/aiot_repo_2/commit/61dcdefc0afcf72b28c3231bcb54fce59f2f8d08) |
| 4 | feature/ai-service | [`3bfa6a6`](https://github.com/sesac-robotics/aiot_repo_2/commit/3bfa6a64bc59bdcf69f27d2188ab72fa3d6288fa) | [`7f050a1`](https://github.com/sesac-robotics/aiot_repo_2/commit/7f050a17fe19922d1912773e2626dd91b1d312b1) |
| 5 | feature/wired-camera-link | [`68cee2e`](https://github.com/sesac-robotics/aiot_repo_2/commit/68cee2e35e42e941ac7807938b22d3640099ff50) | [`489065c`](https://github.com/sesac-robotics/aiot_repo_2/commit/489065c176c6eced7a67882f53d223268bd7af83) |
| 6 | feature/aiot-ui-integration | [`bbe98d3`](https://github.com/sesac-robotics/aiot_repo_2/commit/bbe98d37c6ebaa1b6dedaafeb6afa34e21e90d84) | [`dd5e527`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd5e5274e23a6f9860b3d662bb77bb0cbc43158a) |
| 7 | feature/exercise-posture-scoring | [`55fae69`](https://github.com/sesac-robotics/aiot_repo_2/commit/55fae6974bee80a71c50c8acbf2ce5fa782c58a6) | [`6271f0a`](https://github.com/sesac-robotics/aiot_repo_2/commit/6271f0ae3b4b9daeb263d167a25b6622e209b82c) |
| 8 | feature/posture-error-classification | [`ed04254`](https://github.com/sesac-robotics/aiot_repo_2/commit/ed042542b783b8969c17472e032e550463f08f2a) | [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd) |

모든 원본 S는 R의 조상입니다. 대문자 서버 브랜치는 소문자 서버 브랜치의 조상이지만 실제로 두 merge가 기록돼 있습니다. DB 스키마는 서버 브랜치에서 먼저 들어왔으며 DB merge는 같은 스키마를 새로 설계하지 않았습니다. 유선 브랜치의 과거 AI 파일과 AI 브랜치의 최신 파일도 공통 조상을 따라 병합했습니다.

충돌은 DB·AI·유선 브랜치의 .gitignore와 채점 브랜치의 app/__init__.py에서 발생했습니다. 전자는 로컬 산출물 제외와 테스트 추적 의도를 조정했고, 후자는 app 패키지 책임 설명으로 정리했습니다. 원본 브랜치 포인터를 이동하거나 다른 사람의 커밋을 재작성하지 않았습니다.

## 공통으로 추가한 변경

| 경로 | 통합 후 변경 | 원문 근거 |
|---|---|---|
| [pyproject.toml](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/pyproject.toml) | Black 80열·isort·pytest 경로와 순수 도메인 mypy 범위 | PY-02/03/04 |
| [requirements-dev.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/requirements-dev.txt) | 검사 도구의 확인된 버전 | PY 검사·리뷰 절 |
| [tools/check_source_contracts.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/check_source_contracts.py) | 파싱·설명·import·의존 방향 검사 | PY-03/09, ARC-01/02 |
| [.github/workflows/source-contracts.yml](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.github/workflows/source-contracts.yml) | 새 환경의 정적·백엔드·전송·세션·C++ 상태 검사 | GIT-05, PY 검사·리뷰 절 |
| [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.gitignore) | 로컬 자격 증명·실행 파일 제외, 테스트 경로 전체 제외 해제 | GIT-07 |
| [.editorconfig](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.editorconfig), [.clang-format](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.clang-format) | 문자·줄 끝·서식 설정 | PY-02, CPP-10 **부분 적용·불일치 있음** |
| [README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/README.md) 및 구성요소 README | 호출 흐름·경계·전체 소스 역할과 제한 | PY-09, 주석 §2/§3 |

번호별 원문 링크·버전과 불일치는 [규칙 근거](rules-and-evidence.md)에 있습니다. 규칙을 인용했다는 사실만으로 전체 준수 또는 담당자 승인 완료로 해석하지 않습니다.

## 직접 수정 커밋

| 커밋 | 제목 |
|---|---|
| [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) | refactor(server): isolate rules and restore backend contracts |
| [`f2bfc79`](https://github.com/sesac-robotics/aiot_repo_2/commit/f2bfc7998dbac154fb85395e6952397bcdbaa711) | refactor(gui): centralize auth state and isolate request workers |
| [`a3b1e8b`](https://github.com/sesac-robotics/aiot_repo_2/commit/a3b1e8b3abbb4abe190a5af57fe59778119671e4) | refactor(server): decouple session state from model adapters |
| [`08cd565`](https://github.com/sesac-robotics/aiot_repo_2/commit/08cd565b55d90d55f817d8250df8f0909c9172a5) | fix(firmware): bound connection setup and separate credentials |
| [`783c960`](https://github.com/sesac-robotics/aiot_repo_2/commit/783c960aa363e3f5b7bcabc5210a3b70e88870b1) | refactor(tooling): isolate research runtime resources |
| [`2eac398`](https://github.com/sesac-robotics/aiot_repo_2/commit/2eac398e73b2e0fc871a0154f2215faeae2e6b1b) | style(repo): format integrated sources and document module roles |
| [`8a6773a`](https://github.com/sesac-robotics/aiot_repo_2/commit/8a6773ad65bf7872baf5504eb97eb94f7afe40f8) | chore(tooling): enforce formatting and architecture contracts |
| [`dd68466`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd68466e59405aad95de603d3adfa4acd6a83b2d) | docs(repo): explain component flows and implementation boundaries |
| [`aa4c9c2`](https://github.com/sesac-robotics/aiot_repo_2/commit/aa4c9c20b719e303c56f31c31e7ffa117b4796c5) | fix(server): declare multipart upload dependency |
| [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c) | fix(tooling): close pose resources through process finalizers |

기능 커밋에도 관련 파일의 설명·서식이 함께 들어간 부분이 있습니다. 별도 style 커밋은 있지만 모든 포매팅이 독립 커밋으로 완전히 분리된 것은 아닙니다. 특히 154b890의 스케치 rename은 서버 변경과 함께 기록됐습니다. 문서는 제목만으로 분류하지 않고 실제 파일 diff를 확인합니다.

## 파일 차이를 재확인하는 절차

아래는 기록을 읽는 조회 예시입니다. 원본 브랜치를 체크아웃하거나 실행·재학습하지 않습니다.

```bash
# 원본 S가 통합된 수정 코드 R에 포함되는지 확인합니다.
git merge-base --is-ancestor 3bfa6a64bc59bdcf69f27d2188ab72fa3d6288fa b6be926ed8aafd0c5a4a7a14685034a460af9f1c
# AI 브랜치를 합친 바로 그 merge의 실제 전달 내용을 확인합니다.
git diff --no-renames --name-status 7f050a1^1 7f050a1
# 모든 원본 병합 이후 Codex가 바꾼 파일을 확인합니다.
git diff --no-renames --name-status 33bac7e b6be926
# 특정 수정 파일의 실제 동작 변경과 커밋을 확인합니다.
git diff 33bac7e b6be926 -- Server/AIFitService/database/connection.py
git log --oneline 33bac7e..b6be926 -- Server/AIFitService/database/connection.py
```

GitHub compare 링크는 커밋과 파일 diff를 읽기 위한 것입니다. 자식 문서의 파일 blob 비교와 원본→통합/통합→직접 수정 구분을 함께 사용합니다. 과거 빈 파일의 자동 rename 추정은 실제 의도적 파일 이동의 증거가 아닙니다.

## 보존·검증과 후속 범위

이전 구현 검사는 로컬 119개 통과·2개 skip, 별도 subtests 13개와 최종 원격 CI 통과로 기록돼 있습니다. 상세 범위는 규칙 근거 문서에 있습니다. 저장된 연구 Test 캐시 검사는 독립 최종 Test가 아닙니다. 새 모델 학습·전체 라벨 재검수·운영 배포·실제 장치 수락 시험은 이 변경 기록의 완료 항목이 아닙니다.

이번 작업은 이 브랜치에 Markdown 변경 기록을 추가하는 작업입니다. 스쿼트 고정 태그, 모델·라벨·분할·보고 파일, 실행 소스와 다른 작업 디렉터리의 미커밋 변경을 보존합니다. main 자동 병합은 수행하지 않습니다.
