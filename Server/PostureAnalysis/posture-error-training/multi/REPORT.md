# 5종목 실제 학습 결과

| 종목 | 선택 모델 | 선택 오류/기타 정확도 | 전체 오류 Macro F1 | 오류 평가 범위 |
|---|---|---:|---:|---:|
| squats | cnn | 0.867 | 0.726 | 2/2 |
| pushups | cnn | 0.737 | 0.418 | 6/6 |
| elbow_plank | cnn | 0.837 | 0.587 | 8/8 |
| ytw | cnn | 0.836 | 0.706 | 4/4 |
| shoulder_press | gru | 0.333 | 평가 불가 | 1/6 |

종목별 대상 오류와 평가 분포가 다르므로 숫자를 종목 간 우열로 비교하면 안 됩니다. 스쿼트 외 선택 오류 미검출은 정상 판정이 아닙니다. 어깨는 미검수 초안이며 테스트 오류 양성이 극소수입니다. YTW는 몸통 변형 미검수 상태의 탐색 실험입니다.

## 푸시업

# 푸시업 학습 결과

Six selected QEVD error annotations; absent labels do not certify normal form.

선택 모델: cnn. 피험자 분리: {'train': 2577, 'validation': 639, 'test': 243}

| 모델 | 선택 오류/기타 정확도 | 평가 가능한 오류 Macro F1 | 평가 오류 수 |
|---|---:|---:|---:|
| cnn | 0.7366 | 0.4176466790179972 | 6/6 |
| gru | 0.7613 | 0.4008497170742409 | 6/6 |

Macro F1 전체값은 모든 대상에 양성·음성 테스트 지지가 있을 때만 제공합니다. 지지가 없는 클래스는 null이며 0점이나 100점으로 채우지 않습니다.

- Six selected QEVD error annotations; absent labels do not certify normal form.
- Source clip labels; not frame-level or expert-verified error truth.
- Unlisted selected labels treated as negative (closed-world assumption).
- All models use same source test and group-disjoint validation; no test tuning.
- Invisible joints are masked, but occlusion, camera view and annotation bias remain.
- Selected-error auxiliary negative is not a comprehensive normal class.
- No validated rule baseline for these four tasks; squat proxy comparison remains separate.

## 엘보 플랭크

# 엘보 플랭크 학습 결과

Eight selected QEVD errors; other annotations include supported variants and timing, not verified normal controls.

선택 모델: cnn. 피험자 분리: {'train': 4064, 'validation': 1030, 'test': 203}

| 모델 | 선택 오류/기타 정확도 | 평가 가능한 오류 Macro F1 | 평가 오류 수 |
|---|---:|---:|---:|
| cnn | 0.8374 | 0.5869345319712966 | 8/8 |
| gru | 0.8374 | 0.6047274042413624 | 8/8 |

Macro F1 전체값은 모든 대상에 양성·음성 테스트 지지가 있을 때만 제공합니다. 지지가 없는 클래스는 null이며 0점이나 100점으로 채우지 않습니다.

- Eight selected QEVD errors; other annotations include supported variants and timing, not verified normal controls.
- Source clip labels; not frame-level or expert-verified error truth.
- Unlisted selected labels treated as negative (closed-world assumption).
- All models use same source test and group-disjoint validation; no test tuning.
- Invisible joints are masked, but occlusion, camera view and annotation bias remain.
- Selected-error auxiliary negative is not a comprehensive normal class.
- No validated rule baseline for these four tasks; squat proxy comparison remains separate.

## YTW

# YTW 학습 결과

Exploratory source-label reproduction. Upright/bent torso variants not reviewed; no verified upright normal controls.

선택 모델: cnn. 피험자 분리: {'train': 2398, 'validation': 730, 'test': 214}

| 모델 | 선택 오류/기타 정확도 | 평가 가능한 오류 Macro F1 | 평가 오류 수 |
|---|---:|---:|---:|
| cnn | 0.8364 | 0.7059121621621622 | 4/4 |
| gru | 0.7290 | 0.6771768029245596 | 4/4 |

Macro F1 전체값은 모든 대상에 양성·음성 테스트 지지가 있을 때만 제공합니다. 지지가 없는 클래스는 null이며 0점이나 100점으로 채우지 않습니다.

- Exploratory source-label reproduction. Upright/bent torso variants not reviewed; no verified upright normal controls.
- Source clip labels; not frame-level or expert-verified error truth.
- Unlisted selected labels treated as negative (closed-world assumption).
- All models use same source test and group-disjoint validation; no test tuning.
- Invisible joints are masked, but occlusion, camera view and annotation bias remain.
- Selected-error auxiliary negative is not a comprehensive normal class.
- No validated rule baseline for these four tasks; squat proxy comparison remains separate.
- No verified upright YTW subset. This model is exploratory and must not be used as a validated upright-form assessor.

## 숄더 프레스

# 숄더 프레스 학습 결과

Unverified visual/source-binary drafts; small pilot only. Source test has one positive video.

선택 모델: gru. 피험자 분리: {'train': 53, 'validation': 12, 'test': 9}

| 모델 | 선택 오류/기타 정확도 | 평가 가능한 오류 Macro F1 | 평가 오류 수 |
|---|---:|---:|---:|
| cnn | 0.4444 | 0.0 | 1/6 |
| gru | 0.3333 | 0.0 | 1/6 |

Macro F1 전체값은 모든 대상에 양성·음성 테스트 지지가 있을 때만 제공합니다. 지지가 없는 클래스는 null이며 0점이나 100점으로 채우지 않습니다.

- Unverified visual/source-binary drafts; small pilot only. Source test has one positive video.
- Source clip labels; not frame-level or expert-verified error truth.
- Unlisted selected labels treated as negative (closed-world assumption).
- All models use same source test and group-disjoint validation; no test tuning.
- Invisible joints are masked, but occlusion, camera view and annotation bias remain.
- Selected-error auxiliary negative is not a comprehensive normal class.
- No validated rule baseline for these four tasks; squat proxy comparison remains separate.
- Only 25 source subjects and 30 draft error videos; multiple views are correlated.
- Source test contains only one positive error clip, so headline accuracy can be misleading.
- Some errors occur in only one or two subjects; independent train/validation/test error evaluation is impossible.
