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

<!-- integration-standards-20261025 -->

## 통합 후 연구 코드 해석

이 디렉터리는 운영 AI API와 별개인 데이터 준비·연구 학습·추론·시연 코드입니다. `posture-error-training`은 학습과 저장 모델 검증, `shoulder-qevd`와 `qevd-selected`는 제공자 데이터의 원본·선택 목록·검토 UI를 다룹니다. 디렉터리명에 하이픈이 있어 기존 연구 명령은 해당 디렉터리에서 스크립트로 실행합니다. 운영 패키지로 가장해 경로를 일괄 이동하지 않습니다.

처리 순서는 원본 메타데이터 선택 → 사람/출처 분할 감사 → 관절 캐시 → 공간 정규화와 시간 재표본화 → Train만으로 정규화 통계 계산 → 모델 학습 → 검증 기반 선택 → 고정된 시험 평가입니다. source Train/Test를 프로젝트 편의로 뒤섞지 않습니다. 같은 사람·겹치는 원본 구간·동기화된 시점은 독립 표본으로 흩어지면 누수가 생깁니다.

`models.py`의 CNN/GRU와 suite 모델은 여러 오류의 logit을 출력합니다. sigmoid는 각 오류의 독립 확률이며 softmax처럼 합이 1일 필요가 없습니다. 오류 라벨이 없는 항목은 학습 활성/평가 지원 여부를 확인해야 합니다. 원본 파일이 많아도 특정 관절 오류·복귀 실패·심각도 0/1/2의 확정 라벨 수는 별도 집계입니다. 기존 정상 라벨을 사람 전체 영상에 일괄 확장해서는 안 됩니다.

추출기는 카메라 대신 저장 영상 파일을 읽습니다. `suite_prepare`의 MediaPipe 객체는 프로세스별 추출 어댑터가 소유하며 영상마다 추적을 reset합니다. 프로세스 종료 시 close를 등록합니다. 저장 캐시의 형식·특징 수·정규화·설정 버전은 모델 입력 계약입니다. 이 통합에서 학습 명령을 다시 실행하거나 동결 산출물을 덮어쓰지 않습니다.

저장 결과 UI는 영상 전체의 모델 출력 또는 제공자 라벨 예시를 보여 줍니다. 영상 재생 위치마다 새로운 실시간 정답 판별을 수행하는 것으로 설명하지 않습니다. 연구용 live_app은 시간 단절·미관측·예측 안정화를 시험하는 별도 경로이며 운영 GUI의 세션 횟수 권위가 아닙니다.

`posture-error-training`에서 `python -m unittest test_pipeline test_suite test_live test_api test_suite_api`로 저장 모델·추론 계약을 검사할 수 있습니다. 전체 추출 배열처럼 Git에 없는 자료가 필요한 검사는 skip으로 명시합니다. skip을 통과한 정확도 검사로 계산하지 않습니다. 일부 테스트는 진단에 이미 노출된 저장 Test 캐시를 읽으므로 새 독립 최종 Test가 아닙니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [posture-error-training/api_server.py](posture-error-training/api_server.py) | 저장된 관절 시퀀스를 받는 연구용 HTTP 추론 진입점입니다. 실제 카메라 연속성이나 운영 세션 정확도를 대신 검증하지 않습니다. | `make_server`, `main` |
| [posture-error-training/audit_duplicates.py](posture-error-training/audit_duplicates.py) | 원본 ID·해시를 대조해 데이터 중복과 분할 누수를 검사합니다. 중복 제거 결과와 라벨의 자세 정답 여부는 별도로 검토합니다. | `main` |
| [posture-error-training/features.py](posture-error-training/features.py) | 관절 위치를 신체 중심·크기에 정규화하고 기하 특징과 시간 재표본화를 수행합니다. 저장된 학습 정규화 통계와 현재 입력의 공간 정규화를 구분합니다. | `angle`, `features`, `resample` |
| [posture-error-training/finish_remaining.py](posture-error-training/finish_remaining.py) | 기존 중간 학습 산출물에서 남은 보고 단계 실행을 조정하는 연구 진입점입니다. 실행 시 산출물을 변경할 수 있으므로 보존본에 자동 실행하지 않습니다. | `main` |
| [posture-error-training/inference.py](posture-error-training/inference.py) | 기존 단일 종목 저장 모델과 Train 정규화 통계를 사용해 오류 확률을 계산합니다. 입력 관절의 형식과 가시성 조건을 먼저 검사합니다. | `PostureErrorClassifier` |
| [posture-error-training/live_app/__init__.py](posture-error-training/live_app/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [posture-error-training/live_app/engine.py](posture-error-training/live_app/engine.py) | 연속 관절 관측 게이트·안정화·카메라 읽기와 연구 추론을 연결합니다. 시간 단절·저가시성 입력은 연속 운동 구간을 끊습니다. | `TemporalGate`, `CameraReader`, `LiveEngine` |
| [posture-error-training/live_app/server.py](posture-error-training/live_app/server.py) | 연구용 라이브 엔진을 로컬 HTTP UI에 연결합니다. 외부 Origin과 잘못된 운동 입력을 차단하며 운영 AIFit API와 별도 프로세스입니다. | `LiveHTTPServer`, `make_server`, `main` |
| [posture-error-training/make_report.py](posture-error-training/make_report.py) | 저장된 단일 종목 실험 지표를 보고서와 그림으로 변환합니다. 보고서의 과거 수치를 현재 운영 성능으로 간주하지 않습니다. | `main` |
| [posture-error-training/models.py](posture-error-training/models.py) | 시계열 CNN·GRU와 통계 특징 결합 분류기의 구조를 정의합니다. 출력 logit은 학습 손실 입력이며 sigmoid 확률·등급과 구분합니다. | `CNN`, `GRU` |
| [posture-error-training/predict.py](posture-error-training/predict.py) | 저장된 모델과 정규화 통계로 단일 입력의 추론 결과를 출력하는 CLI입니다. 학습이나 정답 라벨 수정은 수행하지 않습니다. | `main` |
| [posture-error-training/predict_exercise.py](posture-error-training/predict_exercise.py) | 종목을 선택해 저장 모델의 오류 확률을 조회하는 CLI입니다. 비지원 오류 항목의 결과와 정상 판정을 구분합니다. | `main` |
| [posture-error-training/prepare.py](posture-error-training/prepare.py) | 기존 스쿼트 원본 라벨과 분할을 보존하며 관절을 추출하던 연구 도구입니다. 이번 통합에서는 실행·재학습하지 않으며 자원 정리와 서식만 검토합니다. | `extract`, `main` |
| [posture-error-training/run_all_training_results.py](posture-error-training/run_all_training_results.py) | 저장된 다종목 보고서를 구성하고 읽기 전용 결과 화면을 여는 실행 도구입니다. 전체 영상을 새로 학습하는 진입점이 아닙니다. | `main` |
| [posture-error-training/run_esp32cam_live.py](posture-error-training/run_esp32cam_live.py) | ESP32 카메라 기반 연구용 라이브 UI를 명시적으로 실행합니다. 자동 문서 검사나 import에서는 카메라 캡처를 시작하지 않습니다. | `main` |
| [posture-error-training/run_remaining_exercises.py](posture-error-training/run_remaining_exercises.py) | 나머지 네 종목의 추출·분할 감사·학습·보고 순서를 조정합니다. 명시적으로 실행하면 학습 산출물이 변경되므로 동결 보존본과 분리해야 합니다. | `main` |
| [posture-error-training/run_training.py](posture-error-training/run_training.py) | 단일 종목 연구 파이프라인의 추출·학습·보고 명령을 순서대로 실행합니다. 재현 시 Python 환경과 원본·분할 버전을 함께 고정해야 합니다. | `main` |
| [posture-error-training/suite_config.py](posture-error-training/suite_config.py) | 다종목 실험의 경로·대상 오류·관측 조건·설정 버전을 정의합니다. 제공자 과제와 프로젝트의 다등급 채점 과제는 같은 라벨 체계가 아닙니다. | 패키지 경계/선언 |
| [posture-error-training/suite_data_audit.py](posture-error-training/suite_data_audit.py) | 다종목 선택 원본의 사람·분할·라벨과 관절 캐시 연결을 검사합니다. 데이터 양뿐 아니라 사용 가능 범위와 누수 여부를 확인합니다. | `main` |
| [posture-error-training/suite_features.py](posture-error-training/suite_features.py) | 33개 관절 시계열에서 공통 기하·시간 특징과 관측 가능 비율을 계산합니다. 학습과 추론이 동일한 전처리 순서를 사용해야 합니다. | `make_features`, `prepare_landmarks` |
| [posture-error-training/suite_finalize.py](posture-error-training/suite_finalize.py) | 연구 실험의 저장 산출물과 평가 보고를 마무리합니다. 기존 동결 결과를 운영 성능의 현재 측정값으로 취급하지 않습니다. | `finalize` |
| [posture-error-training/suite_inference.py](posture-error-training/suite_inference.py) | 다종목의 저장 모델·정규화·활성 항목을 불러와 오류 확률을 계산합니다. 미지원 항목은 확률 없음으로 구분하고 관절 미관측을 정상으로 보고하지 않습니다. | `ExerciseClassifier`, `FiveExerciseClassifier` |
| [posture-error-training/suite_prepare.py](posture-error-training/suite_prepare.py) | 기존 제공자 Train/Test 구분을 유지하며 선택 영상의 관절을 추출합니다. 프로세스별 MediaPipe 어댑터를 재사용하고 추적 상태는 영상 사이에서 초기화합니다. | `init_worker`, `extract`, `collect`, `main` |
| [posture-error-training/suite_report.py](posture-error-training/suite_report.py) | 저장된 다종목 실험 지표와 지원 범위를 표·화면 자료로 정리합니다. 보고서 생성이 새로운 독립 평가나 학습을 의미하지 않습니다. | `fmt`, `main` |
| [posture-error-training/suite_train.py](posture-error-training/suite_train.py) | 선택된 다종목 Train과 검증 분할로 후보 모델을 학습하고 정규화·모델·지표를 저장합니다. 시험 정답으로 파라미터를 학습하지 않아야 합니다. | `dump`, `duplicate_audit`, `scores`, `binary_metrics`, `choose_thresholds`, `fit`, `main` |
| [posture-error-training/suite_video_smoke.py](posture-error-training/suite_video_smoke.py) | 일부 저장 영상에서 관절 추출·추론 경로가 실행되는지 검사합니다. 표본 실행 성공을 전체 데이터의 정확도로 해석하지 않습니다. | `main` |
| [posture-error-training/test_api.py](posture-error-training/test_api.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `APITests` |
| [posture-error-training/test_live.py](posture-error-training/test_live.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `GateTests`, `HTTPTests` |
| [posture-error-training/test_pipeline.py](posture-error-training/test_pipeline.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `PipelineTests` |
| [posture-error-training/test_suite.py](posture-error-training/test_suite.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `SuiteTests` |
| [posture-error-training/test_suite_api.py](posture-error-training/test_suite_api.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `FiveAPITests` |
| [posture-error-training/train.py](posture-error-training/train.py) | 기존 단일 종목 연구 모델의 사람 분할·정규화·후보 학습을 수행합니다. 보존된 스쿼트 산출물에 자동 재실행하지 않습니다. | `save`, `metrics`, `tune`, `main` |
| [qevd-selected/derived/build_feedback_mapping.py](qevd-selected/derived/build_feedback_mapping.py) | 제공자 오류 라벨을 피드백 문구에 연결합니다. 안내 문구의 연결과 실제 프레임별 정답 라벨링은 별개입니다. | `sha`, `write`, `main` |
| [shoulder-qevd/audit_training_compatibility.py](shoulder-qevd/audit_training_compatibility.py) | 제공자 데이터의 라벨·관측·사람·분할이 프로젝트 과제에 맞는지 검사합니다. 파일 수가 많다는 이유만으로 특정 오류 등급의 정답이 충분하다고 판단하지 않습니다. | `read`, `main` |
| [shoulder-qevd/build_dataset.py](shoulder-qevd/build_dataset.py) | 제공자 숄더 프레스 원본과 메타데이터를 정리합니다. 영상 경로·라벨·분할의 연결을 유지하는 데이터 어댑터입니다. | `read`, `write`, `sha`, `main` |
| [shoulder-qevd/build_five_exercises.py](shoulder-qevd/build_five_exercises.py) | 제공자 메타데이터에서 다섯 종목의 선택 목록을 구성합니다. 원본 라벨을 프로젝트 다등급 정답으로 자동 치환하지 않습니다. | `read`, `save`, `main` |
| [shoulder-qevd/dataset_io.py](shoulder-qevd/dataset_io.py) | 제공자 메타데이터의 경로와 JSON·CSV 읽기를 모읍니다. 원본 ID와 분할을 유지하며 네트워크 다운로드와 라벨 판정을 구분합니다. | `load_records`, `main` |
| [shoulder-qevd/run_demo.py](shoulder-qevd/run_demo.py) | 로컬 저장 데이터와 라벨을 읽는 검토 UI를 실행합니다. 표시된 라벨 예시와 실제 모델 예측을 구분해 안내합니다. | `main` |
| [shoulder-qevd/run_five_exercises.py](shoulder-qevd/run_five_exercises.py) | 다섯 종목의 저장 자료 검토 화면을 여는 진입점입니다. 화면 시연은 신규 학습·독립 시험 완료를 의미하지 않습니다. | 패키지 경계/선언 |
| [shoulder-qevd/test_app/server.py](shoulder-qevd/test_app/server.py) | 원본 영상·메타데이터·저장 모델 결과를 로컬 검토 화면에 제공합니다. 파일 경로를 허용된 데이터 루트로 제한하며 라벨과 모델 출력의 출처를 구분합니다. | `read`, `safe_path`, `media_url`, `Catalog`, `parse_range`, `Handler`, `make_server`, `main` |
| [shoulder-qevd/test_app/test_server.py](shoulder-qevd/test_app/test_server.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `RangeTests`, `AppTests` |
| [shoulder-qevd/validate_dataset.py](shoulder-qevd/validate_dataset.py) | 원본 파일·라벨·구간·메타데이터의 연결과 기본 형식을 검증합니다. 형식 검증이 자세 라벨의 전문가 정답 검수를 대신하지 않습니다. | `read`, `sha`, `LocalLinks`, `main` |
