# 나머지 4종목 학습 및 5종목 추론

기존 스쿼트 모델을 유지하고 푸시업·엘보 플랭크·YTW·숄더 프레스의 CNN과 GRU를 따로 학습합니다. 운동 종목 자동 인식 모델이 아니므로 추론 시 종목을 명시해야 합니다.

## VS Code 실행

- `run_remaining_exercises.py`에서 ▶: 나머지 4종목 전체 추출·학습·보고서 생성. 정상 완료 캐시는 재사용합니다.
- F5 → `종목 선택해서 학습`: 종목 하나를 선택해 재학습.
- `run_all_training_results.py`에서 ▶: 재학습 없이 저장된 실제 예측과 결과 화면을 표시.
- F5 → `학습 모델로 영상 또는 ESP32-CAM 추론`: 종목과 영상 경로/카메라 URL을 선택.

```bash
/home/yoo/yolo/bin/python run_remaining_exercises.py
/home/yoo/yolo/bin/python run_remaining_exercises.py --exercise pushups
/home/yoo/yolo/bin/python run_remaining_exercises.py --skip-prepare
/home/yoo/yolo/bin/python predict_exercise.py --exercise shoulder_press --source /absolute/video.mp4 --no-window
```

좌표 추출은 CPU를 사용해 오래 걸릴 수 있습니다. 다른 작업을 함께 할 때 `--workers 4`처럼 낮추세요. 학습에는 기존 `/home/yoo/yolo/bin/python` 환경을 사용하며 원본 영상과 기존 스쿼트 모델을 덮어쓰지 않습니다.

## 대상 라벨

| 종목 | 대상 오류 | 추가 주의 |
|---|---|---|
| 푸시업 | shallow, butt sinking, elevated hips, head down, moving head, side to side | 다른 변형/오류 라벨이 음성 대조에 포함되어 있으며 완전 정상 대조군은 아님 |
| 엘보 플랭크 | arms facing inward, arms too wide, butt sinking, head down, head up, high hips, legs and hips on the floor, shoulders shrugged up toward the ears | 복합 원본 라벨은 명시된 하위 오류를 함께 부여함. average는 제외 |
| YTW | arms too low, wrong order, too fast, too slow | upright/bent torso 변형 미검수. not low enough 제외. 정석 upright YTW 판별기로 사용하면 안 됨 |
| 숄더 프레스 | 기존 초안의 6개 오류 | 전문가 검수 전이며 서로 독립적인 오류 피험자가 극소수 |

모호함·가시성 문제 등의 제외 기준은 `suite_config.py`, 선별 수는 `multi/selection_report.json`, 영상별 원본 라벨과 target은 `multi/<exercise>/selected.json`에 있습니다. 데이터 수를 늘리기 위해 미확정 어깨 40개와 보류 10개를 억지로 포함하지 않았습니다.

**선택 오류 미검출 = 정상 판정이 아닙니다.** 엘보 플랭크와 YTW는 명확하게 검수된 정상 대조군이 부족합니다. `normal_candidate`를 제공하지 않고 `normal_assessment=unavailable`로 반환합니다. 어깨에 한해서 원본 good와 오류 초안 기반의 `normal_candidate`를 제공하지만 역시 미검수 후보입니다. 모든 모델은 원본 라벨 재현 실험이고, 미기재 대상 라벨을 음성으로 보는 closed-world 가정을 사용합니다.

## 특징 추출과 평가

- 기존 설치 MediaPipe Pose full을 사용하고 약 6fps, 긴 변 최대 480px로 처리합니다.
- 모든 종목의 109개 특징: 25개 관절의 골반 중심·몸통 길이 정규화 좌표와 가시성, 관절 각도 9개.
- 바닥 운동에서 발목이 가려질 수 있어 푸시업/플랭크의 필수 가시성 관절은 어깨·팔꿈치·골반입니다. 어깨와 YTW는 손목도 필수입니다.
- 필수 관절 visibility≥0.3인 프레임이 60% 이상이고 유효 프레임 8개 이상이어야 합니다. visibility가 낮은 관절 좌표·해당 각도는 마스킹합니다. 누락 프레임은 보간합니다.
- 영상 전체를 32시점으로 재표본화합니다. 이 과정으로 실제 시간 구간별 정답이 생기지는 않습니다.
- 원본 test 피험자는 유지합니다. source train 안에서 seed42의 후보 분할 중 학습 가능한 원본 target을 train에 남기고 validation 보조 이진 대상에 양성·음성이 있는 첫 분할을 선택합니다. test 라벨로 분할을 조정하지 않습니다.
- SHA256 완전 중복은 test 우선으로 하나만 남기고, target이 충돌하면 해당 중복 전체를 제외합니다.
- 동일 분할에서 CNN·GRU, 가중 BCE, AdamW, 검증 기반 조기 종료와 임곗값 선택을 사용합니다.
- `any_selected_error`는 선택된 오류의 OR를 학습하는 별도 보조 출력입니다. 세부 출력과 다를 수 있습니다. 그 음성도 전체 정상 클래스를 뜻하지 않습니다.
- 양성/음성 test 지지가 없거나 train에서 볼 수 없던 오류의 F1은 null입니다. 모든 오류가 평가 가능한 경우에만 `macro_f1_all_targets`를 제공합니다. `macro_f1_supported`는 평가 가능한 일부 오류만의 평균입니다.
- 어깨는 세부 클래스 검증 지지가 부족해 보조 이진 검증 점수로 모델을 선택합니다. 검증 양성이 없는 세부 target은 고정 임곗값 0.5를 사용합니다. train 양성이 없는 target은 비활성화해 추론 시 null로 반환합니다.

종목별 문제 난이도·오류 수·가시성 탈락·대조군이 다르므로 정확도를 종목 사이의 우열로 단순 비교할 수 없습니다. 이번 4종목에는 검증된 Rule-based 비교가 없습니다. 기존 스쿼트의 기하학 proxy 비교는 별도 실험입니다.

## 어깨 100개로 충분한가?

현재 어깨 라벨 초안은 정상 후보 70개와 오류 후보 30개입니다. 100개가 모두 독립적인 사람은 아니며 총 25명의 여러 카메라 뷰가 섞여 있습니다.

| 오류 코드 | 영상 수 | 오류가 있는 피험자 수 |
|---|---:|---:|
| TOP_WIDE | 6 | 3 |
| TOP_INCOMPLETE | 3 | 2 |
| HANDS_BEHIND_HEAD | 4 | 3 |
| FOREARM_INWARD | 10 | 6 |
| ELBOW_PATH_NARROW | 6 | 3 |
| ARM_ASYMMETRY | 1 | 1 |

한 명에게만 있는 오류는 사람을 분리한 독립 평가가 불가능합니다. 두 명에게만 있는 오류도 train/validation/test 세 곳에 각각 양성 피험자를 둘 수 없습니다. 이번 가시성 선별을 통과한 74개에서는 원본 test가 정상 후보 8개·오류 후보 1개뿐이므로, 모두 정상 후보라고 해도 정확도 88.9%입니다. 이런 조건에서 높은 정확도는 충분한 성능 증거가 아닙니다. 최종 수치는 `multi/shoulder_press/outputs/splits.json`에서 확인하세요.

다른 세 종목의 영상을 늘려도 어깨 오류 정답이 늘어나는 것은 아닙니다. 영상 증강·프레임 분할도 새로운 독립 피험자를 만들지 않습니다. 우선 미확정 영상을 검수하고, 각 오류에 대해 여러 새로운 피험자·뷰·환경의 사례와 명확한 정석 기준을 확보해야 합니다. 단순히 총 영상 수 목표만 채우지 말고 오류별 피험자 수, 오탐/미탐, 독립 카메라 평가를 확인해야 합니다.

참고: [scikit-learn 피험자 분리 평가와 소표본 교차 검증](https://scikit-learn.org/stable/modules/cross_validation.html), [소표본 분류기의 평가 불확실성 연구](https://arxiv.org/abs/1211.1323).

## 결과 및 AI Server 사용

- `multi/REPORT.md`, `multi/summary.json`: 전체 종목 요약.
- `multi/<exercise>/outputs/best.pt`: 해당 종목 선택 모델.
- 같은 폴더 `metrics.json`, `splits.json`, `history.json`, `*_predictions.csv`: 실제 평가·분할·학습 기록.
- `suite_inference.py`: `FiveExerciseClassifier`로 5종목을 명시적으로 분기하거나 `ExerciseClassifier`로 종목 하나를 유지.

```python
from suite_inference import FiveExerciseClassifier
classifier = FiveExerciseClassifier()
result = classifier.predict_landmarks('pushups', image_landmark_window, aspect=width/height)
```

실시간 BGR 프레임은 `classifier.model('pushups').push_bgr_frame(frame, timestamp_seconds)`에 전달합니다. 카메라/사용자별 인스턴스가 필요하며 약 5.3초 버퍼를 채운 후 예측합니다. ESP32-CAM은 영상을 전송하고 MediaPipe·분류기는 PC/AI Server에서 실행합니다. 실제 ESP32-CAM 환경의 지연과 정확도는 별도 검증이 필요합니다. sigmoid 점수는 검수된 신뢰도나 보정된 확률이 아닙니다.

기존 `api_server.py`의 POST `/predict`도 `exercise` 필드로 5종목을 분기합니다. 예: `{"exercise":"pushups", "landmarks": [...], "aspect": 1.7778}`. `/health`는 종목별 모델 준비 상태를 반환합니다.
