# 체형분석 없는 운동 교정

[설치·실행 안내](MODES.md)의 두 가상환경을 먼저 준비합니다. 저장소 최상위에서 실행합니다.

```bash
.venv-score/bin/python -m GUI.FitHelper.run_direct_correction
```

운동·웹캠 번호를 선택하고 **선택한 운동 시작**을 누릅니다. 기본 웹캠은 0번입니다. ESP32 스트림 주소를 사용하는 경우 화면에서 해당 입력을 선택합니다. **결과 자동저장**을 확인하고 **중지·저장**으로 마감합니다. **저장 목록 새로고침**과 **기록 열기**에서 결과를 조회합니다. 회원가입·체형분석·추천 루틴은 이 독립 교정 화면의 시작 조건이 아닙니다.

`--correction-python`으로 교정 Python을, `--correction-root`로 원본 교정 폴더를, `--sessions`로 저장 위치를 지정합니다. 기본 저장 위치는 Git에서 제외한 `.aifit-local/correction-sessions/`입니다. 개인 기록·DB·미리보기를 커밋하지 않습니다.

원본 LiveEngine·모델·특징·임계값을 유지하고 새 파일에서 기록 호출을 추가합니다. 실행 시 원본과 기록 사본의 루프를 AST로 비교하며 달라지면 실행을 중단합니다. 관절은 실제 처리한 프레임의 33개 x·y·z·visibility 및 world 관절을 저장합니다. `metadata.json`, `frames.jsonl`, `summary.json`과 해시를 기록합니다. 이 원본 교정 모드의 오류 시간창 수는 운동 반복 횟수가 아닙니다. 반복 횟수와 자세 점수는 별도 점수 또는 통합 모드를 사용합니다.

API는 기존 `/api/start`, `/api/stop`, `/api/state`, `/api/frame.jpg`에 새 기록 조회 `/api/sessions`, `/api/sessions/{id}`를 추가합니다. 원본 회원 서버·GUI·스쿼트 보존본은 수정하지 않습니다. ESP32 실장비와 전문가 정답 대비 정확도는 별도 검증 대상입니다.

기존 `GUI.FitHelper.direct_correction_server` 명령은 호환 진입점입니다. 실제 HTTP·교정 루프 사본·파일 기록 구현은 `Server/ExerciseEvaluation/adapters/`에 있으며 GUI에 모델·저장 구현을 두지 않습니다.
