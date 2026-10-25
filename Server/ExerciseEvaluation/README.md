# 운동 교정·자세 점수 선택 실행기

`feature/exercise-posture-scoring`에서 팀원의 오류 분류와 기존 자세 점수를 함께 실행하는 추가 프로그램입니다. 원래 프로그램을 수정하지 않고 `Server/ExerciseEvaluation`의 새 파일로 연결합니다. 동시 실행과 개별 실행 모두 지원합니다.

## 설치

Ubuntu x86-64와 Python 3.12를 기준으로 합니다. 저장소 최상위에서 실행합니다. 두 환경은 저장소 밖의 상위 폴더에 만들어 Git에 포함하지 않습니다.

```bash
python3.12 -m venv ../.correction-runtime
../.correction-runtime/bin/python -m pip install -r Server/ExerciseEvaluation/requirements-correction.txt
../.correction-runtime/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python3.12 -m venv ../.score-integration-runtime
../.score-integration-runtime/bin/python -m pip install -r Server/ExerciseEvaluation/requirements-score.txt
```

`venv`는 프로젝트별 Python 패키지를 분리하는 환경입니다. 교정은 MediaPipe 0.10.21의 `solutions.pose`, 점수는 MediaPipe 1.1.1의 Tasks Heavy를 사용합니다. 같은 환경에 설치하면 한쪽 API가 사라지거나 NumPy 버전이 충돌합니다. 두 프로세스로 분리하여 각 모델의 기존 관절 추출 방식을 유지합니다. CPU용 PyTorch는 교정 환경에만 설치합니다. 점수만 실행할 때는 교정 환경과 PyTorch가 필요하지 않습니다.

설치 환경에 `venv` 또는 `libGL.so.1`이 없으면 Ubuntu의 `python3.12-venv`, `libgl1` 패키지가 필요합니다. CUDA는 필요하지 않습니다. 위 패키지 버전은 이번 연결 검증에서 사용한 버전이며 기존 팀원 학습 환경을 재작성하지 않습니다.

## 실행

교정·점수를 선택하는 화면은 다음과 같이 실행합니다.

```bash
PARENT_DIR="$(cd .. && pwd)"
../.score-integration-runtime/bin/python -m Server.ExerciseEvaluation.run_selectable \
  --correction-python "$PARENT_DIR/.correction-runtime/bin/python" \
  --score-python "$PARENT_DIR/.score-integration-runtime/bin/python"
```

동시 실행 전용은 위 명령에서 `run_selectable`을 `run_combined`로 바꿉니다. Python 실행 파일의 심볼릭 링크 대상이 아니라 가상환경 안의 실행 경로를 전달해야 합니다. 기본 주소는 `http://127.0.0.1:8790`입니다. 기존 앱과 충돌하면 `--port 8791`처럼 지정합니다. 서버를 실행하는 것만으로 카메라를 열지 않습니다.

1. 브라우저에서 위 주소를 엽니다.
2. 운동, 평가 방식, 카메라를 선택합니다. `run_combined`는 교정과 점수를 함께 실행하도록 고정합니다. `run_selectable`은 교정만, 점수만, 동시 실행 중에서 선택합니다.
3. ESP32는 실제 장치의 스트림 주소를 입력합니다. 기본 예시는 `http://192.168.15.208:81/stream`입니다. 웹캠은 보통 `0`이며 외장 카메라는 장치 구성에 따라 `1` 이상일 수 있습니다.
4. **선택한 설정으로 시작**을 누릅니다. 자동 모드는 ESP32 연결이 10초 동안 성립하지 않으면 웹캠으로 전환합니다. 종료 대기 때문에 전환에 추가 시간이 걸릴 수 있습니다. 웹캠으로 전환한 뒤 ESP32로 자동 복귀하지 않습니다.
5. 정면에서 운동별 준비 자세를 유지한 뒤 수행합니다. 스쿼트는 기립, 숄더는 양팔 하단 준비 상태를 확인합니다. 필요한 관절이 보이지 않으면 평가를 보류합니다.
6. **중지**로 장치와 평가 프로세스를 종료합니다. 이미 완료된 반복의 대기 점수는 실제 확보된 관측만으로 마감하며, 미완료 동작을 완료로 만들지 않습니다.

카메라·운동·모드를 바꾸고 다시 시작하면 새로운 세션입니다. 연결 재시도, 자동 카메라 전환, 영상 크기 변경 때도 진행 구간과 반복 목록을 초기화합니다. 영상과 관절을 파일로 저장하지 않습니다. 미리보기는 로컬 메모리의 JPEG이며 모델에는 압축하지 않은 원시 BGR 프레임을 전달합니다.

## 팀 PyQt6 환경에서 실행

2026-10-25 확인한 Confluence 팀 공통 기준과 카메라 가이드에는 PyQt6의 정확한 패치 버전이 없습니다. GUI의 기존 `requirements.txt`도 버전 번호를 고정하지 않습니다. 사용자 승인에 따라 최신 정식 PyQt6 6.11.0과 Qt 런타임 6.11.2를 선택했으며, 이를 과거 팀 전체 합의 버전으로 간주하지 않습니다. 새 설치 목록은 `GUI/FitHelper/requirements-desktop.txt`입니다.

현재 노트북에서는 기존 팀 가상환경 `../aiot_repo_2/R2D2`를 사용합니다. 이 환경의 점수·관절 패키지 버전은 점수 실행기와 일치합니다. 다른 체크아웃에서는 해당 환경의 실제 위치로 경로를 바꿉니다.

```bash
source ../aiot_repo_2/R2D2/bin/activate
python -m pip install -r GUI/FitHelper/requirements-desktop.txt
python -m GUI.FitHelper.exercise_evaluation_window
```

공통 평가 엔진 `http://127.0.0.1:8790`이 먼저 실행되어 있어야 합니다. 위의 `run_selectable` 실행에서 점수 Python을 기존 R2D2 환경의 절대 경로로 지정하면 됩니다. 교정 Python은 호환되는 별도 환경을 유지합니다. 새 PyQt 화면은 카메라를 따로 열지 않고 같은 엔진의 영상과 결과를 표시합니다. 기본 카메라는 웹캠이며 ESP32가 없어도 실행할 수 있습니다. 화면에서 카메라·운동·동시/개별 평가를 선택하고 시작합니다. 창을 닫으면 측정 중지를 요청합니다.

이 파일은 기존 팀 PyQt6 구성 안의 추가 평가 창입니다. 기존 `GUI/FitHelper/main.py`의 로그인·회원가입 흐름이나 `workout_window.py`, `Device/ImageStreamer/ESP32Cam.py` 원본을 교체하지 않습니다. 원래 팀 GUI와 서버·인증·DB까지의 전체 서비스 시험은 이 추가 창의 실행과 구분합니다.

버전 확인 근거: [팀 공통 개발 기준](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/3932327), [팀 카메라 연결 가이드](https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/10158237), [공식 PyQt6 6.11.0 릴리스](https://riverbankcomputing.com/news/PyQt_v6.11.0_Released). 설치 전 환경 목록과 실제 설치 결과는 로컬 `output/team-pyqt-environment-20261025`에 기록합니다. 기존 NumPy·MediaPipe·scikit-learn·모델 파일을 덮어 교체하지 않습니다.

## 결과 해석

교정은 최근 32개 연속 관측을 약 6Hz로 처리하는 선택 오류 분류입니다. 약 5.2초 이상의 문맥이 필요합니다. 6Hz 시간 슬롯마다 실제 프레임을 하나만 선택합니다. 카메라가 느리거나 추론이 지연되어 슬롯이 누락되면 다시 수집합니다. 4fps 입력으로 6Hz의 연속 32표본을 채우지 않습니다. 프레임을 복제해 관측 수를 채우지 않습니다. 선택 오류 미검출은 전신 자세가 정상이라는 판정이 아닙니다.

점수는 기존 10Hz 슬롯 기준 반복 분절과 점수 계산을 사용합니다. 실제 표본이 없는 슬롯은 NaN으로 유지합니다. 스쿼트의 복귀 평가는 완료 직후 최대 0.3초의 실제 관측을 추가로 사용합니다. 진행 중 항목은 잠정 평가이고 완료 반복의 점수와 구분합니다. 플랭크처럼 유지 시간이 중요한 운동은 상태의 `hold_seconds`와 진행 중 항목을 확인합니다. 미평가는 0점 또는 만점으로 채우거나 100점으로 환산하지 않습니다.

두 결과는 같은 세션과 카메라에서 나오지만 분석 시간 범위가 다릅니다. 최근 교정 창의 오류를 특정 완료 반복의 오류 라벨로 저장하지 않습니다. 두 모델을 하나로 재학습하거나 오류 확률을 100점으로 바꾸지 않습니다. 한쪽 오류는 다른 쪽 점수를 0점으로 만드는 이유가 되지 않습니다. 5종 선택을 지원한다는 사실은 5종의 정확도가 검증되었다는 뜻이 아닙니다.

## 소스 연결과 보존 범위

| 새 파일 | 역할과 원본 |
| --- | --- |
| `camera_sources.py` | 팀원의 `Server/PostureAnalysis/posture-error-training/live_app/engine.py`에서 `CameraReader`를 복사했습니다. ESP32·웹캠·자동 선택과 종료 대기 검사를 추가했습니다. 원본의 HTTP 스트림, `/capture` 대체 수신, 재접속을 유지합니다. |
| `correction_adapter.py` | 팀원의 `suite_inference.FiveExerciseClassifier`와 `TemporalGate`를 직접 사용합니다. 원래 입력 크기·정규화·32표본·평활화 방식을 유지하고, 느린 입력의 관측 간격 검사와 실제 창 시간을 추가합니다. |
| `legacy_score.py` | 고정 스쿼트 런타임 `server.py`의 `safe`, `score_samples`, `finish_observation_window`를 복사했습니다. 계산식은 유지하고 서버 전역 초기화와 촬영을 제외했습니다. |
| `score_adapter.py` | 고정 런타임의 `MovementTracker`, `ShoulderTracker`, Heavy 추출기와 점수 계산을 사용합니다. 완료 반복과 진행 중 잠정 항목을 별도로 전달합니다. |
| `worker.py` | 한 종류의 평가만 해당 Python 환경에서 불러옵니다. 프레임·시간·세션 정보를 받고 결과만 반환합니다. |
| `coordinator.py` | 카메라를 한 번 열고 선택한 평가에 전달합니다. 입력 큐 크기는 1이며 느린 처리로 버린 실제 프레임 수를 표시합니다. 세션·연결 구간이 다른 늦은 결과를 버립니다. |
| `contracts.py`, `paths.py` | 운동 별칭, 지원 옵션, 보존 소스 경로, 프로세스 환경을 정의합니다. |
| `server.py`, `web/` | localhost 전용 API와 선택 화면입니다. 시작 전 촬영하지 않으며 외부 사이트의 시작 요청을 거부합니다. |

점수 모델은 `checkpoints/squat-20261025/payload/runtime`의 고정 코드·모델·보조 모델을 사용합니다. `squat-verified-20261025` 태그와 체크포인트는 수정하지 않습니다. 새로운 학습 후보를 자동 채택하지 않습니다. 교정 모델은 팀원의 `outputs/best.pt` 및 `multi/<운동>/outputs/best.pt`를 사용합니다. 기존 15개 PyTorch 체크포인트와 원래 프로그램은 변경하지 않습니다.

팀 AIService의 별도 V2 분류기는 저장소에 없는 `model_registry.json`, quality 모델 등을 요구하므로 이 실행기에서 대체로 연결하지 않습니다. 이 실행기는 저장소에 실제 포함된 팀원의 연구 분류기를 사용합니다. 기존 DB·FastAPI·팀 GUI의 실행 경로도 변경하지 않습니다.

## 검증

카메라 없는 제어 검사는 다음과 같습니다.

```bash
../.score-integration-runtime/bin/python -m unittest discover -s Server/ExerciseEvaluation/tests -v
```

원래 점수 모듈을 찾는 경로를 지정한 뒤 저장된 관절 회귀 검사를 실행합니다.

```bash
export PYTHONPATH=".:checkpoints/squat-20261025/payload/runtime/output/live-webcam-20261023:checkpoints/squat-20261025/payload/runtime/team-posture-scoring/Server/AIService"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
../.score-integration-runtime/bin/python -m Server.ExerciseEvaluation.verify_score
```

실제 두 프로세스의 연결·초기화·종료 검사에도 카메라를 사용하지 않습니다.

```bash
PARENT_DIR="$(cd .. && pwd)"
../.score-integration-runtime/bin/python -m Server.ExerciseEvaluation.verify_workers \
  --correction-python "$PARENT_DIR/.correction-runtime/bin/python" \
  --score-python "$PARENT_DIR/.score-integration-runtime/bin/python"
```

저장 스쿼트 세션 `1792858741575615610`의 690개 관절 표본을 재생합니다. 기대 완료 수는 7회이며 항목 점수와 AI·규칙 총점을 저장값과 비교합니다. AI 총점은 순서대로 73.6, 72.3, 75.7, 75.6, 76.0, 55.7, 84.8입니다. 내부 부동소수점 확률에는 실행 환경에 따른 극소 차이가 생길 수 있지만 표시 항목 점수와 총점은 일치해야 합니다.

회귀 일치는 이전 계산을 보존했다는 증거입니다. 전문가 정답과 비교한 정확도나 ESP32 실제 통신 검증은 아닙니다. 실제 장치가 연결된 상태의 주소, 웹캠 번호, 수신 속도와 동시 추론 지연은 사용자가 시작한 장치 시험에서 별도로 확인합니다.
