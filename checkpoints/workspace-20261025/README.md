# 최신 자세 평가 작업 스냅샷 — 2026-10-25

이 폴더는 팀 저장소 밖에 남아 있던 연구·라이브 앱 소스, 검토 라벨, 분할, 관절 특징 캐시, 자체 학습 모델 및 검증 결과를 경로·크기·SHA-256과 함께 보관합니다. 기존 스쿼트 보존본은 `../squat-20261025`에 그대로 유지합니다. 새로운 학습·모델 채택·운영 배포를 수행한 커밋이 아닙니다.

## 저장 위치와 버전 구분

- 현재 서비스 소스는 저장소의 `Server/AIService`입니다. 기존 미커밋 준비·복귀 실험 API와 테스트를 먼저 보존한 후 `feature/team-integration-standards`의 소스 통합·규칙 적용 이력을 병합했습니다.
- `payload/tools`는 실제 연구 스크립트이며 `payload/five-exercise-comparison`, `payload/mediapipe-pilot` 등은 선행 실험과 의존 코드입니다.
- `payload/output/live-webcam-20261023`은 별도로 실행하던 로컬 웹앱의 당시 스냅샷입니다. 팀 FastAPI 서버와 같은 실행 프로그램으로 간주하지 않습니다. 내부 `runtime` 사본은 해당 세션 시점의 코드·모델 버전입니다.
- `payload/output/four-exercise-source-phase-v2-20261025/clock-posture-corrected-v4`는 별도 4종 연구의 최종 기록입니다. 공통 `CURRENT_FIVE_EXERCISE_RUN.json`이 가리키는 후보와 구분합니다.
- `payload/output/shoulder-expanded-train-v3-20261025` 및 `payload/output/shoulder-work-20261024`는 숄더의 검토·연구·관측 계약을 포함합니다. 연구 결과를 검증된 다등급 운영 모델로 바꾸어 표현하지 않습니다.
- `payload/data`에는 원본 역할·동결 분할·구간 라벨·출처 목록 등 메타데이터가 있습니다. 제공자 조건 라벨, 초기 준비 상태 0/1, 운동 심각도 0/1/2를 혼용하지 않습니다.
- `payload/project-documents`에는 기술 검증 영수증·시험 코드와 기계 판독용 자료 사용 원장이 있습니다. Confluence 기준서·학습 가이드의 Markdown 본문은 포함하지 않습니다.

과거 보고서의 ‘현재’, ‘미커밋’, ‘미게시’는 해당 보고서 작성 시점의 상태입니다. 과거 기록을 이번 Git 게시에 맞춰 소급 수정하지 않습니다. 같은 모델·원본·동기화 구도·겹친 구간을 여러 버전에서 독립 사례로 합산하지 않습니다.

## 해시 검사와 안전한 복원

저장소 루트에서 다음 명령을 실행합니다. 표준 Python만 필요합니다.

```bash
python3 checkpoints/workspace-20261025/verify_restore.py
python3 checkpoints/workspace-20261025/verify_restore.py --restore /tmp/posture-workspace-restored
python3 checkpoints/workspace-20261025/verify_restore.py --verify-workspace /tmp/posture-workspace-restored
```

복원은 원본 바이트를 그대로 복사하며 기존 내용이 다른 파일은 덮어쓰지 않습니다. 모델 로드·학습·카메라·서버를 시작하지 않습니다. 파일 복원 성공과 실행 환경·원본 데이터 확보·자세 정확도는 별개입니다.

기존 연구 코드는 `/home/jayden/SeSACPAI-Mid` 절대 경로를 사용하는 부분이 있습니다. 다른 위치에서 파일이 복원되어도 그대로 실행 가능하다고 가정하지 않습니다. 동결 JSON과 라벨의 과거 경로를 일괄 치환하지 않습니다. 실제 재현 시 원본 작업공간 구조, `team-posture-scoring` Git 체크아웃과 해당 코드·모델 버전을 먼저 일치시켜야 합니다. 실행 환경의 확인된 패키지는 `ENVIRONMENT.json`에 기록합니다.

## 별도 보관 대상과 빠짐 확인

`MANIFEST.json`의 `files`는 실제 포함한 파일 전체이며 `excluded_files`는 조사 범위에서 제외한 파일과 이유입니다. 가상환경·설치 라이브러리·빌드 결과·캐시·일반 로그, 원본 영상·이미지 및 검토 프레임, 제공자 사전학습 가중치·대용량 원본 아카이브는 재배포하지 않습니다. 원본 파일은 이동하거나 삭제하지 않았습니다. 입력·출처 원장의 원본 해시와 경로를 통해 원본 저장소에서 다시 확보해야 합니다. 이 Git 스냅샷만으로 전체 원본 데이터셋까지 백업되었다고 해석하지 않습니다.

Confluence 문서 본문, 개인 문서 저장소 사본 및 게시용 첨부 ZIP은 제외했습니다. 팀 공용 문서는 [프로젝트 기준서](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/13631497)에서 확인합니다. 소스 구성요소 README와 이 복원 안내는 실행·버전 보존을 위한 저장소 문서입니다.

최종 검사와 포함 수량은 `VERIFICATION.json`에서 확인합니다. 검사는 파일 보존·기능 회귀의 근거이며 독립 최종 Test 정확도 시험이 아닙니다.
