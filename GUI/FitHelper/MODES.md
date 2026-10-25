# 운동 교정·자세 점수 실행

팀원 원본 GUI·교정·회원 서버를 수정하지 않고 별도 실행기에서 연결합니다. 저장소 최상위에서 아래 명령을 실행합니다. Confluence 공용 문서의 사본이 아니라 이 코드의 설치·실행 안내입니다.

## 설치

Ubuntu의 Python 3.12와 venv를 사용합니다. 시스템 Qt 라이브러리가 없으면 `sudo apt-get install libegl1 libgl1 libopengl0 libxkbcommon0 libxcb-cursor0`로 설치합니다. 가상환경은 서로 섞지 않습니다. 기존 팀 가상환경을 사용하는 경우 아래 패키지와 호환되는지 먼저 점검합니다.

```bash
python3.12 -m venv .venv-score
.venv-score/bin/python -m pip install -r Server/ExerciseEvaluation/requirements-score.txt -r GUI/FitHelper/requirements-desktop.txt -r Server/AIFitService/requirements.txt
python3.12 -m venv .venv-correction
.venv-correction/bin/python -m pip install -r Server/ExerciseEvaluation/requirements-correction.txt
.venv-correction/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python3 tools/check_evaluation_runtime.py
```

점수는 MediaPipe Tasks 1.1.1, 교정은 팀원 원본이 사용하는 MediaPipe Solutions 0.10.21입니다. PyQt6은 기존 팀 문서에 고정 버전이 없어 사용자가 승인한 6.11.0을 사용합니다. 상세 버전은 각 requirements에 있습니다. 설치된 torch CPU 2.14.1과 기존 requirements 환경으로 실행을 검증했으며 이를 팀 전체 합의 버전으로 표현하지 않습니다. Python 가상환경 실행 파일의 심볼릭 링크를 시스템 Python 경로로 치환하지 않습니다.

모델은 저장소의 교정 모델, `checkpoints/squat-20261025` 보존본 및 `Server/ExerciseEvaluation/models/pose_landmarker_full.task`를 사용합니다. Full 관절 모델의 [공식 출처](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)와 SHA-256은 같은 폴더의 `.source.json`에 있습니다. 추가 개인 영상이나 외부 `output` 폴더를 복사할 필요가 없습니다. 모델 로딩은 수행하지만 이 실행 절차가 모델을 새로 학습하지는 않습니다.

## 실행

```bash
.venv-score/bin/python -m GUI.FitHelper.run_mode_gui
```

1. 기존 로그인 GUI에서 가입·로그인합니다. 새 로컬 실행은 빈 개인 DB를 만들며 다른 팀원의 계정이나 개인 DB를 배포하지 않습니다.
2. 대시보드에서 **운동 모드 선택 · 교정 / 점수 / 통합**을 선택합니다. 체형분석·추천 루틴 없이도 세 모드를 열 수 있습니다. 기존 체형분석 기능은 유지되지만 원본 서버의 AI 서비스·외부 API 설정이 별도로 필요합니다.
3. 운동과 웹캠 번호를 지정하고 **운동 교정**, **자세 점수**, **운동 교정 + 자세 점수** 중 하나를 엽니다. 기본 장치는 웹캠 0번입니다. 서버 준비만으로 촬영하지 않으며 화면에서 시작을 눌러야 합니다.
4. 정면에서 필요한 관절을 화면 안에 두고 운동합니다. 통합 화면은 기존 640×480 영상·오버레이를 유지하고 오른쪽에 항목별 AI·규칙 점수를 추가합니다. 추가 상단 배너는 표시하지 않습니다.
5. **중지·저장**으로 결과를 확정하고 **관절·결과 기록 열기**에서 저장 기록을 조회합니다. 모드 전환은 이전 측정을 중지한 후 진행합니다.

통합·점수 GUI의 기본 촬영 구도는 정면입니다. 교정 모드의 ESP32 스트림/웹캠 선택은 기존 입력을 사용합니다. 통합 API는 `camera=webcam` 또는 `camera=esp32`를 지원하지만 현재 모드 선택 GUI는 웹캠을 기본 사용합니다. ESP32 실장비의 동작은 별도 검증이 필요합니다.

기존 환경을 지정할 때는 다음과 같이 실행합니다.

```bash
/path/to/score-venv/bin/python -m GUI.FitHelper.run_mode_gui --correction-python /path/to/correction-venv/bin/python
```

포트 충돌 시 `--auth-port`, `--correction-port`, `--score-port`, `--integrated-port`를 지정합니다. 기본값은 8000·8782·8790·8793입니다. 별도 관리 중인 서버는 모드 선택 시 중지 요청의 대상이 될 수 있으므로 독립된 포트를 사용합니다. 실행기 종료 시에는 이 실행기가 시작한 서버만 정리합니다.

## 개인 데이터와 환경 설정

기본 저장 위치는 Git에서 제외한 `.aifit-local/`입니다. `AIFIT_DATA_DIR=/absolute/private/path` 환경 변수로 다른 폴더를 지정할 수 있습니다. 점수·교정·통합 요약은 각각 `score-mode-sessions/`, `correction-sessions/`, `integrated-sessions/`에 저장합니다. 기존 노트북의 이전 기록을 이동하거나 삭제하지 않습니다. 새 기본 폴더에서는 이전 기록이 자동 조회되지 않습니다.

로컬 회원 DB는 `AIFit.db`, 로컬 인증 키는 `local-jwt.key`입니다. 기존 서버 설정을 사용하는 경우 `AIFIT_DB_PATH`, `AIFIT_JWT_SECRET` 또는 `JWT_SECRET`을 공급합니다. 이 파일과 개인 운동 기록·미리보기·DB·인증정보를 커밋하지 않습니다. 외부 폴더를 데이터 경로로 지정하면 해당 폴더의 Git 제외 설정도 별도로 확인합니다.

교정은 실제 분석한 33×4 관절 및 world 관절과 오류 출력을, 점수는 실제 10Hz 입력 슬롯의 원시 VIDEO·재검출·선택 관절 33×5, 채점 좌표·점수와 모자이크 미리보기를 기록합니다. 관절 추출 결과를 서로 섞지 않습니다. 통합 요약은 공통 세션 ID와 촬영 시각으로 두 기록을 연결합니다. 원본 교정의 선택적 로그도 원본의 Git 제외 `live_sessions/` 경로를 사용할 수 있습니다.

## 원본 보존과 검증 범위

`mode_gui.py`는 원본 WelcomeWindow를 상속하여 실행 프로세스 안에서 대시보드 연결만 교체합니다. `direct_correction_server.py`는 저장 호출을 제외하면 원본 LiveEngine의 판정 루프와 AST가 같은지 검사하고 다르면 실행을 중단합니다. 저장소의 원본 UI·회원·교정 파일과 스쿼트 체크포인트를 수정하지 않습니다. 팀원 변경으로 계약이 바뀌면 새 연결 파일을 수정합니다. 원본을 이전 버전으로 되돌리지 않습니다.

점수 모델·임계값은 보존하고 관측이 약한 실제 프레임만 독립 Full IMAGE 검출로 재확인합니다. 관절 보간·누락 복제·미평가 점수의 임의 채움은 하지 않습니다. 교정 확률을 점수로 환산하지 않습니다. 일시정지는 완료 결과를 유지하며 재개 때 진행 중 동작을 초기화합니다.

카메라 없는 회귀 검사, 원본 교정 일치, 저장 관절 점수 보존 및 실제 GUI·저장·재조회를 확인합니다. 영상 재생에서 채점한 사례는 모든 사람·운동의 정확도나 독립 최종 Test 통과를 의미하지 않습니다. 통합 기록의 팀 DB·날짜별 리포트 연결과 새 사람의 실제 운동 평가는 남은 작업입니다.
