# feature/exercise-posture-scoring 통합·수정 기록

[전체 기록으로 돌아가기](../README.md) · [규칙 원문과 적용 한계](../rules-and-evidence.md)

## 기록 기준

- 원본 브랜치: `feature/exercise-posture-scoring` / `origin/feature/exercise-posture-scoring`
- 가져온 원본 커밋 S: [`55fae69`](https://github.com/sesac-robotics/aiot_repo_2/commit/55fae6974bee80a71c50c8acbf2ce5fa782c58a6)
- 원본 스냅샷 전체 파일: 58개. 이 수는 신규 기여 파일 수가 아닙니다.
- 이 브랜치의 merge G: [`6271f0a`](https://github.com/sesac-robotics/aiot_repo_2/commit/6271f0ae3b4b9daeb263d167a25b6622e209b82c) / 첫 번째 부모 [`dd5e527`](https://github.com/sesac-robotics/aiot_repo_2/commit/dd5e5274e23a6f9860b3d662bb77bb0cbc43158a)
- 모든 브랜치 통합 완료 M: [`33bac7e`](https://github.com/sesac-robotics/aiot_repo_2/commit/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- 직접 수정 완료 R: [`b6be926`](https://github.com/sesac-robotics/aiot_repo_2/commit/b6be926ed8aafd0c5a4a7a14685034a460af9f1c)
- 담당 범위: 구간 채점·특징·학습/분할 감사·모델 번들·회귀 시험

브랜치별 파일 목록과 고정 커밋을 기준으로 설명합니다. Git 작성자 정보만으로 실제 담당자·기여율을 추정하지 않습니다. 파일이 다른 원본 브랜치와 공유되는 경우 같은 변경이 여러 기록에 나타날 수 있습니다.

## 원본을 가져온 과정

AI 서비스 패키지와 겹치는 __init__.py add/add 충돌을 해결하고 채점 소스·기존 번들을 병합했습니다. 통합 후 채점 알고리즘을 재학습하거나 바꾸지 않았습니다.

원격 이력은 `git fetch origin --prune`으로 갱신한 뒤 새 worktree의 `feature/team-integration-standards`에서 순서대로 merge했습니다. 이 원본은 `6271f0a`의 두 번째 부모이며, 첫 번째 부모와 비교한 실제 전달 경로는 43개입니다. 단일 브랜치를 반복 pull한 것으로 기록하지 않습니다.

- [이 merge가 실제 추가·수정·제거한 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/dd5e5274e23a6f9860b3d662bb77bb0cbc43158a...6271f0ae3b4b9daeb263d167a25b6622e209b82c)
- [원본 S와 전체 통합 M의 차이](https://github.com/sesac-robotics/aiot_repo_2/compare/55fae6974bee80a71c50c8acbf2ce5fa782c58a6...33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd)
- [통합 후 Codex 수정 M→R](https://github.com/sesac-robotics/aiot_repo_2/compare/33bac7e1b5ecb9069fdd7eb96bae9fbf93af71fd...b6be926ed8aafd0c5a4a7a14685034a460af9f1c)

ScoringEngine, 관측 자격, 특징·규칙, 그룹 분할 감사, 후보 학습과 SHA-256 번들 검증은 원본 브랜치의 기능입니다. Codex가 이번 통합에서 새 채점 모델을 만든 것으로 기록하지 않습니다.

## 통합 후 직접 수정한 내용

| 항목 | 수정 전 | 수정 후·이유 | 적용 규칙 원문 |
|---|---|---|---|
| 패키지 충돌 | AI 서비스와 채점 브랜치가 app/__init__.py를 각각 추가했습니다. | 최종 app 패키지의 책임 설명으로 해결했습니다. 한쪽 패키지 전체를 덮어쓰지 않았습니다. | [PY-09 · 모듈 책임 설명](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| 채점 소스 서식 | 채점 패키지는 이미 타입·계약·분할 감사가 있는 코드였습니다. | Black/isort와 일부 모듈 설명을 정리했습니다. AST 비교에서 설명·import·타입을 제외한 채점 실행 구조를 보존했습니다. | [PY-02 · 일관된 서식](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3604597) (참조 버전 3) |
| C EGL 어댑터 | nvidia_egl_device.c는 기존 프로세스 전용 C 어댑터였습니다. | 서식만 정리하고 C로 컴파일했습니다. C++로 변환하거나 시스템 드라이버를 변경하지 않았습니다. | [COMMENT · §2/§4 파일 역할과 제약 설명](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/6979834) (참조 버전 2) |
| 번들과 분할 | 기존 모델·manifest·지원 정보·검증 보고가 포함됐습니다. | 원본 바이트를 유지했습니다. training.py에 후보 학습 코드가 있어도 이번 통합에서는 실행하지 않았습니다. | [GIT-07 · 동결 데이터·모델 보존](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3899580) (참조 버전 3) |

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
| [Server/AIService/app/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/__init__.py) | A | M | 다른 원본 이력 반영 | 통합 후 수정 없음 · [Server/AIService/app/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/__init__.py) |
| [Server/AIService/app/scoring/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/__init__.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/__init__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/__init__.py) |
| [Server/AIService/app/scoring/__main__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/__main__.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/__main__.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/__main__.py) |
| [Server/AIService/app/scoring/engine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/engine.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/engine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/engine.py) |
| [Server/AIService/app/scoring/features.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/features.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/scoring/features.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/features.py) |
| [Server/AIService/app/scoring/geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/geometry.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/geometry.py) |
| [Server/AIService/app/scoring/held_geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/held_geometry.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/held_geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/held_geometry.py) |
| [Server/AIService/app/scoring/media.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/media.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/media.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/media.py) |
| [Server/AIService/app/scoring/partition.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/partition.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/partition.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/partition.py) |
| [Server/AIService/app/scoring/plank_item_observation.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/plank_item_observation.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/scoring/plank_item_observation.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/plank_item_observation.py) |
| [Server/AIService/app/scoring/recovery_w_phase.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/recovery_w_phase.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/scoring/recovery_w_phase.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/recovery_w_phase.py) |
| [Server/AIService/app/scoring/report.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/report.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/report.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/report.py) |
| [Server/AIService/app/scoring/rule_score.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/rule_score.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/rule_score.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/rule_score.py) |
| [Server/AIService/app/scoring/rules.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/rules.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/rules.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/rules.py) |
| [Server/AIService/app/scoring/sequence.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/sequence.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/sequence.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/sequence.py) |
| [Server/AIService/app/scoring/three_items.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/three_items.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/three_items.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/three_items.py) |
| [Server/AIService/app/scoring/training.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/training.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/training.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/training.py) |
| [Server/AIService/app/scoring/w_morphology.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/w_morphology.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/w_morphology.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/w_morphology.py) |
| [Server/AIService/app/scoring/ytw_features.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/ytw_features.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/scoring/ytw_features.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/ytw_features.py) |
| [Server/AIService/app/scoring/ytw_partial_trial.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/ytw_partial_trial.py) | A | A | 원본과 동일 | 설명·서식·import·타입 변경; 비교한 실행 구조 동일 · [Server/AIService/app/scoring/ytw_partial_trial.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/ytw_partial_trial.py) |
| [Server/AIService/app/scoring/ytw_phases.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/app/scoring/ytw_phases.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/app/scoring/ytw_phases.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/app/scoring/ytw_phases.py) |
| [Server/AIService/artifacts/posture_score/development_round3.json](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/development_round3.json) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/development_round3.json](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/development_round3.json) |
| [Server/AIService/artifacts/posture_score/manifest.json](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/manifest.json) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/manifest.json](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/manifest.json) |
| [Server/AIService/artifacts/posture_score/plank.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/plank.joblib) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/plank.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/plank.joblib) |
| [Server/AIService/artifacts/posture_score/pushups.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/pushups.joblib) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/pushups.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/pushups.joblib) |
| [Server/AIService/artifacts/posture_score/shoulder.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/shoulder.joblib) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/shoulder.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/shoulder.joblib) |
| [Server/AIService/artifacts/posture_score/squats.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/squats.joblib) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/squats.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/squats.joblib) |
| [Server/AIService/artifacts/posture_score/training_support.json](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/training_support.json) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/training_support.json](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/training_support.json) |
| [Server/AIService/artifacts/posture_score/validation.json](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/validation.json) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/validation.json](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/validation.json) |
| [Server/AIService/artifacts/posture_score/ytw.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/artifacts/posture_score/ytw.joblib) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/artifacts/posture_score/ytw.joblib](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/artifacts/posture_score/ytw.joblib) |
| [Server/AIService/pyproject.toml](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/pyproject.toml) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/pyproject.toml](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/pyproject.toml) |
| [Server/AIService/requirements-posture-dev.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/requirements-posture-dev.txt) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/requirements-posture-dev.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/requirements-posture-dev.txt) |
| [Server/AIService/requirements-posture-score.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/requirements-posture-score.txt) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/requirements-posture-score.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/requirements-posture-score.txt) |
| [Server/AIService/requirements-posture-video.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/requirements-posture-video.txt) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/requirements-posture-video.txt](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/requirements-posture-video.txt) |
| [Server/AIService/tests/scoring/test_cli.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tests/scoring/test_cli.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tests/scoring/test_cli.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tests/scoring/test_cli.py) |
| [Server/AIService/tests/scoring/test_engine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tests/scoring/test_engine.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tests/scoring/test_engine.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tests/scoring/test_engine.py) |
| [Server/AIService/tests/scoring/test_geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tests/scoring/test_geometry.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tests/scoring/test_geometry.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tests/scoring/test_geometry.py) |
| [Server/AIService/tests/scoring/test_partition.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tests/scoring/test_partition.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tests/scoring/test_partition.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tests/scoring/test_partition.py) |
| [Server/AIService/tests/scoring/test_training.py](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tests/scoring/test_training.py) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tests/scoring/test_training.py](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tests/scoring/test_training.py) |
| [Server/AIService/tools/posture_score/examples/reviewed_clip.json](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tools/posture_score/examples/reviewed_clip.json) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tools/posture_score/examples/reviewed_clip.json](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tools/posture_score/examples/reviewed_clip.json) |
| [Server/AIService/tools/posture_score/install.sh](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tools/posture_score/install.sh) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tools/posture_score/install.sh](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tools/posture_score/install.sh) |
| [Server/AIService/tools/posture_score/nvidia_egl_device.c](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tools/posture_score/nvidia_egl_device.c) | A | A | 원본과 동일 | C 서식 변경 · [Server/AIService/tools/posture_score/nvidia_egl_device.c](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tools/posture_score/nvidia_egl_device.c) |
| [Server/AIService/tools/posture_score/run_nvidia.sh](https://github.com/sesac-robotics/aiot_repo_2/blob/55fae6974bee80a71c50c8acbf2ce5fa782c58a6/Server/AIService/tools/posture_score/run_nvidia.sh) | A | A | 원본과 동일 | 통합 후 수정 없음 · [Server/AIService/tools/posture_score/run_nvidia.sh](https://github.com/sesac-robotics/aiot_repo_2/blob/b6be926ed8aafd0c5a4a7a14685034a460af9f1c/Server/AIService/tools/posture_score/run_nvidia.sh) |

## 통합 후 새로 추가한 관련 경계와 검사

이 기록에 별도로 귀속하는 새 구현 파일은 없습니다. 공통 검사 설정·루트 README는 [부모 기록의 공통 변경](../README.md#공통으로-추가한-변경)에서 확인합니다.

## 관련 직접 수정 커밋

| 커밋 | 실제 제목 | 이 기록의 관련 경로 수 |
|---|---|---|
| [`2eac398`](https://github.com/sesac-robotics/aiot_repo_2/commit/2eac398e73b2e0fc871a0154f2215faeae2e6b1b) | style(repo): format integrated sources and document module roles | 6 |

커밋 수나 파일 수는 개인 기여율이 아닙니다. 일부 기능 변경 커밋에 해당 파일의 설명·서식 정리가 함께 포함되어 있습니다. 별도 style 커밋이 있다는 이유만으로 기능과 서식이 완전히 분리되었다고 표현하지 않습니다.

## 검증 근거와 해석

기존 채점 검사 40개와 별도 subtests 13개가 통과했습니다. 모델 파일 바이트 동일성, 채점 실행 구조, C 어댑터 컴파일을 확인했습니다. 모델 정확도나 새 독립 Test 결과를 측정한 것은 아닙니다.

최종 R의 [CI](https://github.com/sesac-robotics/aiot_repo_2/actions/runs/40164117985)와 [PR #4](https://github.com/sesac-robotics/aiot_repo_2/pull/4)에 전체 검사 범위를 기록했습니다. CI는 순수 경계와 백엔드·전송·세션 23개 검사 및 C++17 상태 시험을 확인하며, 로컬 119개 검사를 원격 CI 전체 수행 수로 바꾸어 쓰지 않습니다. 이번 문서 작성에서 기존 검사 119개를 다시 실행한 것으로 기록하지 않습니다.

## 적용 한계와 남은 차이

일반 formatter와 타입·계약 검사 통과는 새 채점 성능의 증거가 아닙니다. 연구/현재 운영 모델 간 채택 관계는 기존 기준서를 따라야 합니다. 원본 브랜치에 이미 있던 모델 파일을 이번에 새로 생성한 것으로 보지 않습니다.

이 문서는 규칙의 적용 근거를 추적하는 기록이며 팀 전체 승인·MISRA 준수 인증·모델 성능 평가가 아닙니다. 이번 추가 작업은 Markdown 문서화이며 실행 소스·동결 자료는 수정하지 않습니다.

## 2026-10-25 후속 동기화와 규칙 적용

### 가져온 이력과 원본 유지

최초 통합 기록 S·A·R은 위의 과거 고정 기준으로 유지합니다. 이번 입력 S2는 [`68d77539`](https://github.com/sesac-robotics/aiot_repo_2/commit/68d77539e8a350e733e687c4e8621ca660e206db)이며, 통합 브랜치의 이전 HEAD는 `2173ecb3be8a4f69a7dbba89a568486ee805c8fb`입니다. S2가 그 후손이므로 `git merge --ff-only feature/exercise-posture-scoring`으로 누락 10커밋을 가져왔습니다. 새로운 분기 병합 충돌이나 한쪽 파일의 강제 선택은 없었습니다.

원격의 main, feature/AIFitServer, feature/ai-service, feature/aifit-server, feature/aiot-ui-integration, feature/db, feature/posture-error-classification, feature/wired-camera-link를 확인했습니다. 각 원격 HEAD의 커밋이 자세 점수 브랜치와 통합 이력에 이미 포함되어 있어 팀원 브랜치에 새로 가져올 커밋은 없었습니다. 실제 원격 상태 확인 시점을 기록하는 것이며 향후 자동 동기화·지속 감시를 구성하지 않습니다.

### 통합 후 수정과 규칙 근거

후속 구현은 [`160f0898`](https://github.com/sesac-robotics/aiot_repo_2/commit/160f089801a8b54977739d7daf8bebd6adbc4537)입니다. 규칙의 최신 Confluence 버전을 직접 다시 읽었으며 공통 기준 v6, Python v3, 설계·모듈화·DDD v3, Git v3, 주석·docstring v2, Git 입문 v4로 기존 참조와 같았습니다. 원문 링크는 [규칙·증거](../rules-and-evidence.md)에 유지합니다.

| 기준 | 가져온 코드에서 수정한 부분 | 이유와 보존 범위 |
| --- | --- | --- |
| ARC-01·02 | `GUI/FitHelper/integrated_session.py`의 실제 제어를 `Server/ExerciseEvaluation/application/integrated_session.py`로 이동했습니다. 교정 저장/HTTP와 카메라 구현은 adapters로 옮겼습니다. | 서버가 GUI 구현을 import하지 않도록 하며, GUI의 기존 파일은 호환 facade로 유지합니다. |
| ARC-03·05 | `application/ports.py`의 교정·점수·저장 포트를 만들고 bootstrap에서 생성자로 주입했습니다. | application이 파일 저장·카메라 구현을 선택하지 않습니다. 새 공개 경계의 타입과 실패 의미를 명시했습니다. |
| State Machine·ARC-06·07 | `domain/lifecycle.py`가 실행 수명만 소유합니다. | 중복 pause/resume은 제어 이벤트를 증가시키지 않습니다. 기존 반복 추적기·점수 소유권을 바꾸거나 UNKNOWN을 오류로 채우지 않습니다. |
| Repository·PY-05·06·08 | `adapters/session_repository.py`에서 UUID·요약 ID를 검증하고 기존 JSON 저장/조회·구간 경로 탐색을 담당합니다. | 파일 실패를 성공으로 숨기지 않습니다. 실패 후 동일 세션 재시도를 확인했습니다. 기존 DB 스키마와 개인 기록은 이동하지 않습니다. |
| PY-03·09·12 | 런타임 import 환경을 서버의 `runtime_environment.py`로 옮기고 모듈 실행·한국어 책임/실패 설명을 유지했습니다. | import만으로 카메라·운동을 시작하지 않으며 실행 중 sys.path 변경을 추가하지 않았습니다. |
| PY-02·04·ARC-01·02 | Black/isort, 새 domain·port mypy, 계층별 import 금지 검사와 합성 위반 시험을 추가했습니다. | 형식 검사만으로 설계 준수를 주장하지 않도록 상태·저장 실패·의존 방향을 실제로 검사합니다. |
| GIT-01·03·05·06 | 기존 이력을 보존하고 구현과 설명을 분리하여 두 작업 브랜치에 반영합니다. | main 직접 push·강제 push·자가 리뷰 승인을 하지 않습니다. |

### 검증과 적용 한계

전체 자동 검사 119개 및 subtests 13개, 제어 9개를 통과했습니다. 도메인/포트 포함 지정 mypy, 소스 계약 230파일과 Black/isort를 확인했습니다. 보존 관절 690표본·7회에서 횟수·항목 점수·AI/규칙 총점이 같았습니다. 팀원 원본과 스쿼트 보존 파일 1,183개의 SHA-256도 작업 전과 같았습니다.

새 환경의 실제 Qt 가입·로그인·체형분석 없는 통합 실행·기록 저장/재조회에서 관절33행을 확인했습니다. 기존 영상을 재생했으며 새 물리 웹캠 촬영은 하지 않았습니다. pause 중 완료2회가 유지되고 resume 후 검출3회가 됐으며 전체채점2회·재확인1회였습니다. 이는 전 영상 채점 성공률이나 독립 정확도 검증이 아닙니다.

DDD 폴더 구조를 이유로 전체 운영 서비스를 단일 Aggregate로 바꿨다고 주장하지 않습니다. 기존 모델·연구 파이프라인의 동적 타입과 호환 전역 주입은 보존 경계에 남아 있습니다. 새 도메인·포트는 엄격한 타입·의존 검사를 적용하지만 모든 레거시 파일의 타입·상태 소유권까지 재작성하지 않았습니다. 기존 회원 DB·API·장치·팀원 원본·스쿼트 보존본은 변경하지 않습니다. Confluence 전문이나 개인 운동 기록은 이 기록에 포함하지 않습니다.
