# feature/wired-camera-link 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/wired-camera-link` / `origin/feature/wired-camera-link`
- 가져온 원본 커밋 S: [`68cee2e`](https://github.com/sesac-robotics/aiot_repo_2/commit/68cee2e35e42e941ac7807938b22d3640099ff50)
- 원본 스냅샷 전체 파일: 33개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`489065c`](https://github.com/sesac-robotics/aiot_repo_2/commit/489065c176c6eced7a67882f53d223268bd7af83) / 첫 번째 부모 [`7f050a1`](https://github.com/sesac-robotics/aiot_repo_2/commit/7f050a17fe19922d1912773e2626dd91b1d312b1)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: 유선 네트워크 시험·파일 공유 도구 및 과거 AI 스냅샷

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

AI 서비스 다음에 병합했습니다. 원본 브랜치에는 과거 AI 코드도 있지만 해당 merge가 새로 전달한 파일은 tools의 8개 파일과 .gitignore입니다. 앞서 들어온 최신 AI 코드를 과거 스냅샷으로 교체하지 않았습니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `489065c`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 9개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/7f050a17fe19922d1912773e2626dd91b1d312b1...489065c176c6eced7a67882f53d223268bd7af83)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/68cee2e35e42e941ac7807938b22d3640099ff50...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

유선 주소 검증·연결 원복과 파일 공유 서버/경로 제한 구현은 원본 작업입니다. 이 도구의 기능 개선을 Codex의 통합 후 성과로 귀속하지 않습니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 중복 AI 이력 | 브랜치에 AI main·추론 어댑터가 함께 들어 있었습니다. | 공통 조상 관계에 따라 이미 병합된 AI 변경을 유지했습니다. S→M 차이는 다른 AI 브랜치의 최신 코드가 반영된 결과입니다. | [GIT-06 · 공유 이력 보존](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |
| ignore 충돌 | 기존 ignore 규칙이 테스트·스크립트 경로 전체를 제외하는 경우가 있었습니다. | .gitignore 충돌을 해결하며 테스트를 추적 가능한 상태로 두고 DB·모델·환경·실행 산출물 제외 의도를 유지했습니다. 최종 app/test_*.py 전체 제외 제거는 후속 공통 설정 변경입니다. | [GIT-07 · 추적할 코드와 로컬 산출물 구분](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |
| 도구 코드와 테스트 | wired_camera_link와 t9_share 기능 및 회귀 시험이 이미 있었습니다. | 도구 구현은 그대로 유지했습니다. test_share.py의 import/서식을 정리하고 tools/README.md에 책임과 실행 조건을 보강했습니다. | [PY-02 · 도구 서식과 설명](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |

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
| [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/.gitignore) | M | M | 다른 원본 이력 반영 | 설정·의존성 또는 펌웨어 변경 · [.gitignore](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/.gitignore) |
| [Server/AIService/.env.example](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/.env.example) | A | — | 다른 원본 이력 반영 | 통합 후 수정 없음 · [Server/AIService/.env.example](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/.env.example) |
| [Server/AIService/app/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/__init__.py) | A | — | 다른 원본 이력 반영 | 통합 후 수정 없음 · [Server/AIService/app/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/__init__.py) |
| [Server/AIService/app/camera/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/camera/__init__.py) | A | — | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/camera/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/camera/__init__.py) |
| [Server/AIService/app/camera/stream.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/camera/stream.py) | A | — | 다른 원본 이력 반영 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/camera/stream.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/camera/stream.py) |
| [Server/AIService/app/inference/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/inference/__init__.py) | A | — | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/inference/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/inference/__init__.py) |
| [Server/AIService/app/inference/mediapipe_pose.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/inference/mediapipe_pose.py) | A | — | 다른 원본 이력 반영 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/inference/mediapipe_pose.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/inference/mediapipe_pose.py) |
| [Server/AIService/app/inference/yolo_pose.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/inference/yolo_pose.py) | A | — | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/inference/yolo_pose.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/inference/yolo_pose.py) |
| [Server/AIService/app/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/main.py) | A | — | 다른 원본 이력 반영 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/main.py) |
| [Server/AIService/app/streaming/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/streaming/__init__.py) | A | — | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/streaming/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/streaming/__init__.py) |
| [Server/AIService/app/streaming/output.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/app/streaming/output.py) | A | — | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/streaming/output.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/streaming/output.py) |
| [Server/AIService/main.py](https://github.com/sesac-robotics/aiot_repo_2/blob/2c4636cf811430f5aa0167ffee07ff36bd0466a1/Server/AIService/main.py) | D | — | 해당 경로 없음 | 통합 후 수정 없음 |
| [Server/AIService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/Server/AIService/requirements.txt) | A | — | 다른 원본 이력 반영 | 통합 후 수정 없음 · [Server/AIService/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/requirements.txt) |
| [tools/t9_share/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/__init__.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/t9_share/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/__init__.py) |
| [tools/t9_share/__main__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/__main__.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/t9_share/__main__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/__main__.py) |
| [tools/t9_share/filesystem.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/filesystem.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/t9_share/filesystem.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/filesystem.py) |
| [tools/t9_share/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/requirements.txt) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/t9_share/requirements.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/requirements.txt) |
| [tools/t9_share/server.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/server.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/t9_share/server.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/server.py) |
| [tools/t9_share/test_share.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/t9_share/test_share.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [tools/t9_share/test_share.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/t9_share/test_share.py) |
| [tools/test_wired_camera_link.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/test_wired_camera_link.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/test_wired_camera_link.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/test_wired_camera_link.py) |
| [tools/wired_camera_link.py](https://github.com/sesac-robotics/aiot_repo_2/blob/68cee2e35e42e941ac7807938b22d3640099ff50/tools/wired_camera_link.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [tools/wired_camera_link.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/tools/wired_camera_link.py) |

## 통합 후 새로 추가한 관련 경계와 검사

이 기록에 별도로 귀속하는 새 구현 파일은 없습니다. 공통 검사 설정·루트 README는 [부모 기록의 공통 변경](../README.md#공통으로-추가한-변경)에서 확인합니다.

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`a3b1e8b`](https://github.com/sesac-robotics/aiot_repo_2/commit/a3b1e8b3abbb4abe190a5af57fe59778119671e4) | refactor(server): decouple session state from model adapters | 1 |
| [`2eac398`](https://github.com/sesac-robotics/aiot_repo_2/commit/2eac398e73b2e0fc871a0154f2215faeae2e6b1b) | style(repo): format integrated sources and document module roles | 8 |
| [`8a6773a`](https://github.com/sesac-robotics/aiot_repo_2/commit/8a6773ad65bf7872baf5504eb97eb94f7afe40f8) | chore(tooling): enforce formatting and architecture contracts | 1 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

유선/공유 도구 검사 15개 중 14개 통과·1개 skip입니다. 실제 네트워크 설정을 변경하지 않았고 원격 팀 계정으로 파일을 전송하지 않았습니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

원본 도구의 기능을 새로 구현하거나 실제 두 PC 연결을 수락 시험한 작업은 아닙니다. 공통 check_source_contracts.py는 이 브랜치에서 가져온 파일이 아니라 통합 후 추가한 검사입니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.
