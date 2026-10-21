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
