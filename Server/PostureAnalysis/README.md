# 자세 오류 분류 · QEVD 라벨링 · ESP32-CAM 실시간 테스트

유선혁 담당 작업: 공개 운동 데이터 정리, 정상 후보·오류 클래스 정의, MediaPipe 전처리, CNN/GRU 학습·평가, Rule proxy 비교 및 AI Server용 추론 모듈을 정리한 실험 구현입니다. 기존 API 서버 및 장치 코드를 변경하지 않고 별도 폴더에 추가했습니다.

## 포함한 작업

- `shoulder-qevd/`: 어깨 영상 100개에 대한 QEVD 호환 라벨 초안, 출처·검수 상태, 영어 교정 문구, 검증·재생성 도구.
- `qevd-selected/`: QEVD 메타데이터와 스쿼트·푸시업·엘보 플랭크·YTW의 오류 이름 및 영어 교정 문구 매핑. 원본 라벨과 작성한 피드백을 구분합니다.
- `posture-error-training/`: 5종목의 MediaPipe 특징 전처리, CNN/GRU 학습, 사람 단위 분할·중복 검사, 학습된 체크포인트, Accuracy/Precision/Recall/Macro F1 평가.
- `posture-error-training/live_app/`: ESP32-CAM 영상 → PC 관절 추출 → 선택 종목 모델 → 오류 이름·영어 교정 문구. 영상 위 판정 표시와 중지 전 마지막 결과 확인을 지원합니다.
- `posture-error-training/api_server.py`: AI Server 연동용 로컬 `/predict` API. 종목·image landmarks·aspect를 입력합니다.
- `shoulder-qevd/test_app/`: 라벨과 저장 영상, 학습 결과를 확인하는 로컬 UI.

원본 운동 영상, 대용량 특징 배열·전체 캐시, 실시간 촬영 이미지와 로그는 포함하지 않습니다. 테스트 재현을 위해 공개 데이터에서 얻은 수치 관절 캐시를 종목별 1개씩 포함했습니다. `test-fixtures.json`에 목록이 있습니다. 원본 QEVD 라벨의 권리·사용 조건은 해당 데이터셋을 따릅니다. 이 브랜치가 원본 자료의 새 라이선스를 부여하는 것은 아닙니다.

## 설치와 VS Code 실행

Python 3.11 또는 3.12 환경을 권장합니다. 현재 구현은 MediaPipe 0.10.21의 `solutions.pose` API를 사용합니다.

```bash
cd Server/PostureAnalysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd posture-error-training
python run_esp32cam_live.py
```

VS Code에서 `shoulder-qevd/shoulder-qevd.code-workspace`를 열고 **Python: Select Interpreter**로 위 가상환경을 선택하세요. `run_esp32cam_live.py` 오른쪽 위 ▶ 또는 F5의 **ESP32-CAM 실시간 자세 오류 테스트**로 실행합니다. 기존 개발 PC의 `/home/yoo/yolo/bin/python`은 `POSTURE_PYTHON` 환경변수로 지정할 수 있습니다. 개인 PC 경로를 필수로 사용하지 않습니다.

브라우저 `http://127.0.0.1:8780/`에서 본인 카메라 주소와 종목을 선택하고 시작합니다. 카메라 기본 예시는 `http://192.168.15.208:81/stream`이며 실제 보드 IP에 맞게 수정해야 합니다. 스트림이 응답하지 않으면 같은 보드의 `/capture` 사진 연속 수신으로 전환합니다. 카메라는 영상만 전송하고 모델은 PC에서 실행합니다. 처음 32개 관절 표본이 필요하며, 수신 속도에 따라 5~10초 이상 걸릴 수 있습니다.

## AI Server 연동

```bash
cd posture-error-training
python api_server.py --port 8770
```

`GET /health`, `POST /predict`를 제공합니다. 요청 본문은 다음 형식입니다. 실제 `landmarks`에는 8~300개 프레임의 MediaPipe **image landmarks** `(T, 33, 4)`가 필요합니다. 각 관절은 `[x, y, z, visibility]`이며 world landmarks가 아닙니다.

```json
{"exercise":"squats","aspect":1.3333333333,"landmarks":[]}
```

위 빈 배열은 구조 설명용이며 유효한 요청이 아닙니다. 서버 코드는 `suite_inference.FiveExerciseClassifier`를 한 번 생성해 사용하면 됩니다. 입력 종목은 `squats`, `pushups`, `elbow_plank`, `ytw`, `shoulder_press`입니다. 이 코드가 종목을 자동 인식하지는 않습니다. API는 기본적으로 `127.0.0.1`에서만 열립니다.

## 학습·검증 결과와 한계

| 종목 | 선택 모델 | 오류별 Macro F1 | 평가 해석 |
|---|---|---:|---|
| 스쿼트 | CNN | 0.7264 | shallow / back not straight 2개 오류 |
| 푸시업 | CNN | 0.4176 | 선택 오류 6개 |
| 엘보 플랭크 | CNN | 0.5869 | 선택 오류 8개 |
| YTW | CNN | 0.7059 | 기존 영상의 오류 4개, 동작 변형 재검수 미완료 |
| 숄더 프레스 | GRU | 전체 클래스 산출 불가 | 미검증 소량 초안, test 오류 영상 1개 |

모델 선택은 검증 세트 기준입니다. 오류 미검출은 전체 자세의 정상을 뜻하지 않으며, 모델 점수는 보정된 정확도가 아닙니다. 어깨는 100개 라벨 중 관절 품질을 통과한 74개로 실험했고, 별도 전문가 검수는 없습니다. 실제 ESP32-CAM의 사람을 대상으로 한 정답 대비 성능 평가는 완료하지 않았습니다. 스쿼트의 정상 후보/선택 오류 이진 정확도는 0.8667이며 오류별 Macro F1과 다른 지표입니다. Rule proxy는 기울기·깊이를 사용한 비교 실험으로, 실제 등 굽음을 직접 측정하지 않습니다.

자세한 근거는 `posture-error-training/outputs/REPORT.md`, `posture-error-training/multi/REPORT.md` 및 각 종목의 `outputs/metrics.json`을 확인하세요. YTW는 기존 데이터와 모델을 유지합니다.

## 테스트와 데이터 복원

```bash
cd posture-error-training
python -m unittest test_pipeline test_suite test_live test_api test_suite_api
```

위 테스트는 포함된 모델·최소 캐시·메타데이터를 사용합니다. 23개 중 전체 특징 배열로 정규화를 재계산하는 1개는 원본 데이터 전처리 전까지 skip하며, 나머지 22개는 원본 영상 없이 실행됩니다. 로컬 HTTP 소켓 사용 권한이 필요합니다. 원본 영상 없이 실행할 수 있습니다.

재학습이나 저장 영상 재생은 별도 원본 영상이 필요합니다. 기존 로컬 `qevd-selected/raw/fit/`와 `shoulder-qevd/raw/fit/`을 이 폴더 아래 같은 상대 경로로 복원하세요. 어깨 미확정·보류 영상은 `pending/videos/`, `quarantine/videos/`에 별도 보관합니다. 상세 라벨 및 경로는 각 데이터셋의 manifests에 있습니다. 원본 확보 및 사용 조건은 각 자료 제공처를 확인해야 합니다.

학습은 `run_training.py`(스쿼트), `run_remaining_exercises.py`(다른 4종목)로 실행합니다. 원본 영상 복원 전에는 재학습을 실행하지 마세요. 원본 경로를 재생성할 때는 `prepare.py` 및 `suite_prepare.py`가 현재 폴더 기준의 메타데이터를 만듭니다.
