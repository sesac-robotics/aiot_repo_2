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
QT_QPA_PLATFORM=offscreen .venv-score/bin/python -m pytest tests/integration/test_evaluation_gui.py tests/integration/test_integrated_session.py tests/integration/test_correction_recording.py tests/integration/test_score_observation.py tests/integration/test_score_preview_privacy.py tests/integration/test_evaluation_portability.py
PYTHONPATH=".:checkpoints/squat-20261025/payload/runtime/output/live-webcam-20261023:checkpoints/squat-20261025/payload/runtime/team-posture-scoring/Server/AIService" .venv-score/bin/python -m Server.ExerciseEvaluation.verify_score
```

pytest는 개발 검사 패키지로 별도 설치합니다. 저장 관절 재생은 기존 계산의 보존 검사이며 전문가 정답 대비 정확도 검사가 아닙니다. `VERIFICATION.json`은 과거 검사 당시의 동결 기록이며 현재 절차와 구분합니다. 학습·라벨·분할·스쿼트 보존본은 변경하지 않습니다.
