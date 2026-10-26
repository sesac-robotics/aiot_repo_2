# 자세 점수 실행 모듈

전체 설치와 기본 GUI 실행은 [운동 모드 안내](../../GUI/FitHelper/MODES.md)를 따릅니다. 이 폴더의 서버는 점수 전용입니다. 원본 교정과 점수의 동시 실행은 별도 `GUI.FitHelper.run_integrated` 서버가 담당합니다. 이전 `run_combined`, `run_selectable`, 브라우저 UI는 이 브랜치에서 추가했던 이전 경로이며 새 GUI 경로로 대체했습니다. 팀원 원본 진입점은 유지합니다.

```bash
.venv-score/bin/python -m Server.ExerciseEvaluation.run_score
```

기본 주소는 `http://127.0.0.1:8790`입니다. 서버 시작만으로 촬영하지 않습니다. `--score-python`을 생략하면 현재 가상환경의 Python을 사용합니다. 기본 Heavy VIDEO·운동 분절·점수 계산은 `checkpoints/squat-20261025/payload/runtime` 보존본을 사용합니다. Full IMAGE 모델은 같은 실제 프레임의 관절 재검출에만 사용하고 모델 출처·해시는 `models/pose_landmarker_full.source.json`에 기록합니다.

기본 저장 경로는 Git 제외 `.aifit-local/score-mode-sessions/{session_id}/epoch-*`입니다. 환경 변수 `AIFIT_DATA_DIR`로 상위 경로를 지정합니다. `GET /api/recordings`와 `/api/recordings/{recording_id}?offset=0&verify=1`로 기록·결과·무결성을 재조회합니다. 저장 슬롯·관측률·검출 횟수·전체 항목 채점 확정 횟수는 구분하며 미관측 점수를 임의로 채우지 않습니다.

## 검사

```bash
.venv-score/bin/python -m unittest discover -s Server/ExerciseEvaluation/tests -v
QT_QPA_PLATFORM=offscreen .venv-score/bin/python -m pytest tests/integration/test_evaluation_gui.py tests/integration/test_integrated_session.py tests/integration/test_correction_recording.py tests/integration/test_score_observation.py tests/integration/test_score_preview_privacy.py tests/integration/test_evaluation_portability.py tests/integration/test_evaluation_architecture.py
PYTHONPATH=".:checkpoints/squat-20261025/payload/runtime/output/live-webcam-20261023:checkpoints/squat-20261025/payload/runtime/team-posture-scoring/Server/AIService" .venv-score/bin/python -m Server.ExerciseEvaluation.verify_score
```

pytest는 개발 검사 패키지로 별도 설치합니다. 저장 관절 재생은 기존 계산의 보존 검사이며 전문가 정답 대비 정확도 검사가 아닙니다. `VERIFICATION.json`은 과거 검사 당시의 동결 기록이며 현재 절차와 구분합니다. 학습·라벨·분할·스쿼트 보존본은 변경하지 않습니다.

## 책임 경계와 패턴

| 계층·소스 | 소유 책임 | 의존·실패 계약 |
| --- | --- | --- |
| `domain/lifecycle.py` | 시작·일시정지·재개·종료 상태 | 표준 라이브러리만 사용합니다. 중복 전이는 이벤트를 늘리지 않습니다. |
| `application/integrated_session.py` | 두 분석의 실행 순서와 공통 세션 기록 연결 | 저장소와 점수 실행 포트를 생성자로 주입합니다. 파일·카메라 구현이나 GUI를 import하지 않습니다. |
| `application/ports.py` | 교정·점수·저장 인터페이스 | 기존 결과 스냅샷과 원본 판정 의미를 유지합니다. 저장 실패는 호출자에게 전달합니다. |
| `adapters/session_repository.py` | 기존 JSON 통합 요약의 저장·목록·구간 경로 조회 | UUID와 요약 ID가 일치해야 합니다. 원자적 교체 실패를 저장 완료로 표시하지 않습니다. |
| `adapters/correction_recording.py`, `correction_server.py` | 원본 교정에 기록 호출·HTTP 조회 추가 | 원본 루프 AST가 다르면 시작을 거절합니다. |
| `adapters/integrated_camera.py` | 단일 카메라·방향·표시 모자이크·점수 Worker 조립 | 원본 교정이 카메라를 소유합니다. 교정·점수의 관절을 혼합하지 않습니다. |
| `runtime_environment.py` | 레거시 import 환경의 자식 프로세스 설정 | 실행 중 sys.path를 변경하지 않습니다. |
| GUI 호환 파일 | 이전 import·실행 명령 유지 | 실제 구현을 위 경계로 위임하며 카운터를 추가하지 않습니다. |

Repository는 통합 기록의 두 사용처인 종료 저장과 HTTP 목록 조회를 분리하기 위해 사용합니다. 상태 전이는 경량 State Machine으로 표현하며 상태마다 클래스를 만들지 않습니다. Adapter는 기존 카메라·모델 계약을 유지하고, DI는 테스트에서 실제 장치·저장소를 가짜 포트로 바꾸는 데 사용합니다. Qt Signal/Slot·비동기 HTTP는 기존 화면 갱신 경로를 유지합니다.

`SessionLifecycle`는 실행 수명의 소유자이며 새로운 반복 Aggregate나 정상/오류 판정자는 아닙니다. 기존 MovementTracker/ShoulderTracker가 반복 분절을 소유합니다. 교정 시간창 확률을 Verdict나 점수로 환산하지 않습니다. 기존 연구·운영 세션과 단일 Aggregate로 통합했다거나 모든 레거시 타입·전역 상태를 정리했다고 표현하지 않습니다. 담당자 소유 DB·API·펌웨어·동결 모델은 변경하지 않았습니다.

새 도메인과 포트는 mypy 검사 대상입니다. `tools/check_source_contracts.py`는 도메인의 외부 라이브러리 의존, 서버의 GUI 의존, application의 adapter 구현 의존을 거절합니다. 모델·Qt의 동적 스냅샷 전체가 엄격한 타입 검사 대상인 것은 아닙니다. 형식·타입 통과는 모델 정확도의 증거가 아닙니다.
