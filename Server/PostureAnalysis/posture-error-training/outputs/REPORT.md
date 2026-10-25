# 스쿼트 자세 오류 학습 결과

선택 모델: cnn (검증 Macro F1로 선택)

| 모델 | 정확도(정상/선택 오류) | Precision macro(오류별) | Recall macro(오류별) | Macro F1(오류별) |
|---|---:|---:|---:|---:|
| cnn | 0.8667 | 0.6571 | 0.8371 | 0.7264 |
| gru | 0.8500 | 0.6468 | 0.7895 | 0.6915 |
| rule_proxy | 0.6833 | 0.2619 | 0.4400 | 0.3284 |
| majority_train | 0.4167 | 0.0000 | 0.0000 | 0.0000 |

정확도는 정상 후보 대 선택 오류의 이진 정확도입니다. 오류별 수치는 shallow/back not straight 다중 라벨 평균입니다. 모든 오류 조합을 맞추는 exact_match_accuracy는 metrics.json에 별도로 기록합니다.

Rule proxy는 무릎 굽힘 깊이와 몸통 기울기를 사용하는 비교용 휴리스틱입니다. 몸통 기울기는 등 굽음과 동일하지 않아 기존 Rule-based 시스템과의 완전한 비교를 뜻하지 않습니다.

- Source clip-level labels, not expert reannotation or frame-level truth.
- Unlisted target labels are treated as negative under a closed-world annotation assumption.
- 2D image landmarks and monocular z are view-dependent; back curvature is incompletely observed.
- Original source test workers preserved; validation workers held out from source train.
- Each video resampled to 32 steps; live windows may differ from full video duration.
- No accuracy guarantee on ESP32-CAM or on the other four exercises.
