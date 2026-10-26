# feature/db 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/db` / `origin/feature/db`
- 가져온 원본 커밋 S: [`2856af6`](https://github.com/sesac-robotics/aiot_repo_2/commit/2856af67cb7791b332a204da0db65622f4e2bf75)
- 원본 스냅샷 전체 파일: 18개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`61dcdef`](https://github.com/sesac-robotics/aiot_repo_2/commit/61dcdefc0afcf72b28c3231bcb54fce59f2f8d08) / 첫 번째 부모 [`fd3f979`](https://github.com/sesac-robotics/aiot_repo_2/commit/fd3f97901244d18149e0e055845230762088bc38)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: SQLite 기본 스키마와 두 데모 스크립트

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

서버 브랜치 뒤에 병합했습니다. 같은 AIFit.sql은 앞서 서버 브랜치에서 이미 들어와 있었으므로 이 merge의 실제 추가는 두 데모 스크립트와 .gitignore 조정입니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `61dcdef`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 3개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/fd3f97901244d18149e0e055845230762088bc38...61dcdefc0afcf72b28c3231bcb54fce59f2f8d08)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/2856af67cb7791b332a204da0db65622f4e2bf75...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

테이블 관계·외래 키·UNIQUE 제약은 원본 스키마의 구현입니다. Codex가 새 DB 모델을 설계하거나 기존 저장 데이터를 마이그레이션해 성능을 높인 작업이 아닙니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 병합 충돌 | 서버 브랜치와 DB 브랜치가 .gitignore의 DB 제외 규칙을 함께 수정했습니다. | 양쪽 DB 제외 의도를 유지했습니다. merge의 첫 번째 부모와 결과를 비교하면 데모 2개 추가와 ignore 변경을 확인할 수 있습니다. | [Git 입문 절차 §4 · 복원·충돌 해결·검증·공유](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6947031) (참조 버전 4) |
| 데모 import 부작용 | db_test.py의 최상위 코드가 연결·스키마·샘플 INSERT/DELETE를 실행했습니다. | main()과 실행 가드 안으로 옮겼습니다. import만으로 DB를 수정하지 않습니다. 명시적으로 실행하면 여전히 데모 데이터 변경이 발생할 수 있습니다. | [PY-12 · 진입점 밖 자동 실행 방지](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 데모 가독성 | db_test_v2.py는 이미 실행 가드와 rollback/finally close가 있었습니다. | 기존 자원 처리와 SQL을 유지하면서 서식·import·설명을 정리했습니다. | [PY-02 · 공통 서식](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 스키마 보존 | AIFit.sql과 실제 로컬 DB는 서로 다른 자산입니다. | AIFit.sql은 바이트가 동일합니다. 통합 검증은 임시 DB를 사용했으며 로컬 사용자 DB를 초기화하지 않았습니다. | [GIT-07 · 데이터와 코드 변경 구분](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |

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
| [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/2856af67cb7791b332a204da0db65622f4e2bf75/.gitignore) | M | M | 다른 원본 이력 반영 | 설정·의존성 또는 펌웨어 변경 · [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.gitignore) |
| [Server/AIFitDB/AIFit.sql](https://github.com/sesac-robotics/aiot_repo_2/blob/2856af67cb7791b332a204da0db65622f4e2bf75/Server/AIFitDB/AIFit.sql) | A | — | 원본과 동일 | 통합 후 수정 없음 · [Server/AIFitDB/AIFit.sql](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitDB/AIFit.sql) |
| [Server/AIFitDB/db_test.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2856af67cb7791b332a204da0db65622f4e2bf75/Server/AIFitDB/db_test.py) | A | A | 원본과 동일 | 실행 코드 변경 포함 · [Server/AIFitDB/db_test.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitDB/db_test.py) |
| [Server/AIFitDB/db_test_v2.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2856af67cb7791b332a204da0db65622f4e2bf75/Server/AIFitDB/db_test_v2.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIFitDB/db_test_v2.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitDB/db_test_v2.py) |

## 통합 후 새로 추가한 관련 경계와 검사

아래 파일은 원본 S에서 가져온 것으로 귀속하지 않습니다. 통합 완료 M 뒤에 추가한 구현·설명·시험입니다.

| 새 파일 | 변경 종류 | 직접 변경 커밋 |
|---|---|---|
| [Server/AIFitDB/README.md](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIFitDB/README.md) | README 설명 보강 | [`dd68466`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd68466e59405aad95de603d3adfa4acd6a83b2d) |

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`154b890`](https://github.com/sesac-robotics/aiot_repo_2/commit/154b8904526e334686753a4b9f6ef96c436f7dfd) | refactor(server): isolate rules and restore backend contracts | 1 |
| [`2eac398`](https://github.com/sesac-robotics/aiot_repo_2/commit/2eac398e73b2e0fc871a0154f2215faeae2e6b1b) | style(repo): format integrated sources and document module roles | 1 |
| [`8a6773a`](https://github.com/sesac-robotics/aiot_repo_2/commit/8a6773ad65bf7872baf5504eb97eb94f7afe40f8) | chore(tooling): enforce formatting and architecture contracts | 1 |
| [`dd68466`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd68466e59405aad95de603d3adfa4acd6a83b2d) | docs(repo): explain component flows and implementation boundaries | 1 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

기본 스키마의 빈 DB 초기화·외래 키·트랜잭션·인증은 test_backend.py로 확인했습니다. db_test.py/db_test_v2.py 데모 자체의 명시적 실행 성공을 확인한 것은 아닙니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

db_test.py의 과거 데모 질의가 현행 v2 스키마에 모두 맞는지 수정·검증하지 않았습니다. 버전형 DB 마이그레이션 도구도 추가하지 않았습니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.
