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
