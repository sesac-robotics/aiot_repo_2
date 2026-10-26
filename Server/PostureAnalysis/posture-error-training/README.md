# 5종목 자세 오류 분류 모델

나머지 4종목 학습은 `run_remaining_exercises.py`, 5종목 결과 화면은 `run_all_training_results.py`에서 ▶를 누릅니다. 종목별 대상·평가 제한·어깨 데이터 부족 분석은 [MULTI_EXERCISE_README.md](MULTI_EXERCISE_README.md)에 있습니다.

아래는 기존 스쿼트 1차 모델의 설명입니다. 나머지 종목의 결과는 `multi/REPORT.md`에 저장됩니다.

# 스쿼트 자세 오류 분류 모델

VS Code에서 `run_training.py`를 열고 ▶를 누르거나 F5의 **스쿼트 CNN + GRU 학습**을 선택합니다. 기존 `/home/yoo/yolo/bin/python` 환경을 사용합니다. 원본 데이터와 기존 Server 모델은 덮어쓰지 않습니다.

## 실행 파일

- `run_training.py`: MediaPipe 추출 → 원본 중복 검사 → CNN/GRU 학습·평가. 추출한 영상은 cache에서 재사용합니다.
- `prepare.py`: QEVD 스쿼트 4,308개에서 명확한 대상 1,602개를 선별합니다. 가시성 부족 영상은 추가로 제외합니다.
- `train.py`: 피험자 분리, 학습셋 통계로 표준화, 검증셋에서 조기 종료·임곗값·모델 선택 후 원본 test 평가.
- `predict.py`: 저장 영상, 웹캠, 명시한 ESP32-CAM URL에서 추론합니다.
- `inference.py`: AI Server가 import해서 쓰는 `PostureErrorClassifier`.
- `api_server.py`: 선택적으로 localhost:8770에서 JSON API를 실행합니다.
- `outputs/REPORT.md`, `metrics.json`, `splits.json`: 실제 평가 결과와 데이터 분할.
- `outputs/best.pt`: 검증 Macro F1로 선택한 모델. 원본 source test 점수로 고르지 않습니다.

## 클래스 정의

스쿼트의 `shallow`(깊이 부족), `back not straight`(원본의 등 정렬 오류) 두 오류를 독립적인 sigmoid 출력으로 분류합니다. 둘 다 있는 영상도 허용합니다. 정상 후보는 원본에 `no obvious issue`가 있고, 명확한 기준 특성 외 다른 오류 라벨이 없는 영상으로 제한합니다. `insufficient`, `not visible`, 설명이 충돌하는 `knees over toes`, 운동 변형 `plie`는 제외합니다. 다른 오류가 붙은 영상은 대상 두 오류 중 하나가 명시된 경우에만 학습에 포함합니다. 두 오류 모두 미검출은 **선택한 두 오류가 미검출**이라는 뜻이며 모든 오류에 대한 정상 판정은 아닙니다.

원본 라벨이 명시하지 않은 대상 오류는 음성으로 보는 closed-world 가정입니다. 이 가정은 모든 프레임의 정상 여부를 보장하지 않습니다. 문구는 생성된 교정 초안이며 학습 입력은 영상에서 추출한 좌표입니다. 문구나 라벨 텍스트를 입력 특징에 섞지 않습니다.

## 전처리 및 평가

MediaPipe Pose full(기존 설치 0.10.21), 긴 변 최대 480px, 약 6fps. 주요 8관절 visibility≥0.3인 프레임이 60% 이상이며 유효 프레임 8개 이상인 영상을 사용합니다. 누락 프레임은 시간 보간하고 각 영상을 32시점으로 재표본화합니다. 16개 관절의 상대좌표/가시성과 5개 각도, 총 69개 특징을 사용합니다. 이미지 가로세로 비율을 좌표에 반영하고 골반 중심·몸통 길이로 정규화합니다. 단안 z는 실제 3D 측정값이 아닙니다.

원본 train/test를 유지하고 train 피험자 중 20%를 seed42로 validation에 분리합니다. 동일 worker의 영상은 분할 사이에 섞이지 않습니다. SHA256가 같은 타 피험자 영상이 있으면 학습을 중단합니다. CNN과 GRU 모두 같은 분할을 사용하며 BCEWithLogitsLoss, train 양성 비율 기반 가중치, early stopping을 사용합니다.

Accuracy는 `normal_vs_selected_error.accuracy`(정상 후보 대 선택 오류의 이진 정확도), 오류별 Precision/Recall/Macro F1은 두 대상의 평균입니다. 다중 라벨 완전 일치율도 따로 저장합니다. 정상 다수 클래스만 맞춰 얻은 정확도와 혼동하지 않도록 majority baseline을 함께 제공합니다.

Rule proxy는 무릎 각도의 하위 10%와 몸통 기울기의 상위 10%를 사용하며 임곗값은 validation에서만 고릅니다. **몸통 기울기는 척추 굽음과 동일하지 않습니다.** 이 비교는 두 기하학 휴리스틱 기준이며 기존 프로젝트 Rule-based 전체 시스템을 검증한 결과가 아닙니다.

## AI Server 연동

```python
from inference import PostureErrorClassifier
classifier = PostureErrorClassifier()
# 이미 디코딩한 BGR 프레임마다 호출. 카메라별로 별도 인스턴스 사용.
result = classifier.push_bgr_frame(frame_bgr, timestamp_seconds)
if result and result['status'] == 'ok':
    print(result['scores'], result['detected_errors'], result['feedbacks'])
```

ESP32-CAM은 영상 전송을 담당하고 MediaPipe·분류 추론은 PC/AI Server에서 수행합니다. 호출자는 운동 종목을 스쿼트로 지정해야 하며 이 모델이 운동 종목을 자동 구분하지는 않습니다. 연속 32샘플(약 5.3초)을 모은 뒤 추론하며, 장시간 연결 끊김·시간 역전은 버퍼를 비웁니다. 짧은 저장 영상은 파일 종료 시 8샘플 이상이면 partial_window_at_file_end를 표시해 보간 후 평가합니다. 실시간에서는 32샘플 전까지 warming_up을 유지합니다. 실제 영상 전체와 실시간 윈도 길이가 달라 생기는 차이는 별도 검증이 필요합니다. 영상 예제의 뒤쪽 동작을 앞 프레임에서 맞히는 프레임별 ground truth는 없습니다.

```bash
/home/yoo/yolo/bin/python predict.py --source /absolute/video.mp4 --no-window
/home/yoo/yolo/bin/python predict.py --source http://CAMERA_IP:81/stream
/home/yoo/yolo/bin/python api_server.py
```

POST `/predict` body: `{"landmarks": [[[x,y,z,visibility], ...]], "aspect": 1.7778}`. MediaPipe **image** 좌표의 33관절, 8~300프레임을 받습니다. `normal_candidate=true`도 다른 오류의 부재를 보장하지 않습니다.

## 범위와 한계

이번 학습은 스쿼트만 대상으로 합니다. 나머지 4종목, 특히 어깨 초안과 YTW 몸통 변형은 검수·좌표 추출 후 별도 학습이 필요합니다. 원본 clip 수준 라벨 재현 실험이며 전문가의 프레임별 오류 판단이나 ESP32-CAM 실사용 정확도를 보증하지 않습니다. 명확하지 않은 클래스는 억지로 정상·오류로 확정하지 않았습니다.

기술 참고: [MediaPipe Pose](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md), [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), [QEVD 논문](https://arxiv.org/abs/2407.08101).

## ESP32-CAM 실시간 테스트

`run_esp32cam_live.py`를 열어 ▶로 실행하세요. [실행 방법](LIVE_ESP32CAM_README.md)에 주소·종목 선택·결과 해석을 정리했습니다.
