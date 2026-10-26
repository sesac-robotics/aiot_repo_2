# feature/AIFitServer 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/AIFitServer` / `origin/feature/AIFitServer`
- 가져온 원본 커밋 S: [`2d3274e`](https://github.com/sesac-robotics/aiot_repo_2/commit/2d3274e4b03b4b0222879a11016f8869dfae997d)
- 원본 스냅샷 전체 파일: 18개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`6e529b9`](https://github.com/sesac-robotics/aiot_repo_2/commit/6e529b9efebecec6b8f545631ce28e1bb2926e87) / 첫 번째 부모 [`2c4636c`](https://github.com/sesac-robotics/aiot_repo_2/commit/2c4636cf811430f5aa0167ffee07ff36bd0466a1)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: 초기 백엔드 진입점과 독립 로그인·영상 GUI 시험 코드

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

이 브랜치는 대문자가 포함된 원본 참조입니다. `feature/aifit-server`와 이름이 다르며, 원본 커밋은 소문자 브랜치의 조상입니다. 먼저 별도 merge로 이력을 포함한 뒤 후속 소문자 브랜치의 서버 구조를 합쳤습니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `6e529b9`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 5개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/2c4636cf811430f5aa0167ffee07ff36bd0466a1...6e529b9efebecec6b8f545631ce28e1bb2926e87)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/2d3274e4b03b4b0222879a11016f8869dfae997d...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

`userLogin.py`, `video_gui.py` 제거는 `fd3f979`에서 가져온 후속 브랜치에 이미 포함된 변경입니다. Codex가 새 설계를 만들며 해당 두 파일을 삭제하거나 GUI로 이동한 것으로 귀속하지 않습니다. 같은 이름의 GUI 파일이 있다는 사실만으로 이동 또는 동일 저자를 추정하지 않습니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 초기 파일과 후속 서버의 구분 | 원본 main.py·README·requirements와 두 독립 GUI 파일이 있었습니다. | 후속 소문자 브랜치의 라우터·서비스 구조가 먼저 반영되었고, 그 통합 결과에 대해 설정·import·DB 수명을 수정했습니다. 구체적인 직접 수정은 [소문자 브랜치 문서](feature-aifit-server.md)에 연결합니다. | [GIT-06 · 기존 공유 이력 보존](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |
| 초기 이력 보존 | 대문자 원격 참조가 별도로 존재했습니다. | 대문자 브랜치를 이름 변경하거나 덮어쓰지 않고 6e529b9의 두 번째 부모로 보존했습니다. | [GIT-06 · 공유 이력 강제 재작성 금지](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |

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
| [Server/AIFitService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/2d3274e4b03b4b0222879a11016f8869dfae997d/Server/AIFitService/README.md) | M | M | 다른 원본 이력 반영 | README 설명 보강 · [Server/AIFitService/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/README.md) |
| [Server/AIFitService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2d3274e4b03b4b0222879a11016f8869dfae997d/Server/AIFitService/main.py) | M | M | 다른 원본 이력 반영 | 실행 코드 변경 포함 · [Server/AIFitService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/main.py) |
| [Server/AIFitService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/2d3274e4b03b4b0222879a11016f8869dfae997d/Server/AIFitService/requirements.txt) | A | A | 다른 원본 이력 반영 | 설정·의존성 또는 펌웨어 변경 · [Server/AIFitService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitService/requirements.txt) |
| [Server/AIFitService/userLogin.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2d3274e4b03b4b0222879a11016f8869dfae997d/Server/AIFitService/userLogin.py) | A | A | 통합에서 제거됨 | 통합 후 수정 없음 |
| [Server/AIFitService/video_gui.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2d3274e4b03b4b0222879a11016f8869dfae997d/Server/AIFitService/video_gui.py) | A | A | 통합에서 제거됨 | 통합 후 수정 없음 |

## 통합 후 새로 추가한 관련 경계와 검사

이 기록에 별도로 귀속하는 새 구현 파일은 없습니다. 공통 검사 설정·루트 README는 [부모 기록의 공통 변경](../README.md#공통으로-추가한-변경)에서 확인합니다.

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) | refactor(server): isolate rules and restore backend contracts | 2 |
| [`aa4c9c2`](https://github.com/sesac-robotics/aiot_repo_2/commit/aa4c9c20b719e303c56f31c31e7ffa117b4796c5) | fix(server): declare multipart upload dependency | 1 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

초기 userLogin.py/video_gui.py를 별도 실행하지 않았습니다. 현재 백엔드의 임시 DB·인증 검사는 후속 서버 구조에 대한 검사입니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

초기 프로토타입의 모든 기능이 현재 구조와 일대일 대응한다고 보장하지 않습니다. 현재 서버 계약의 변화는 [소문자 브랜치 문서](feature-aifit-server.md)와 함께 확인합니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.
