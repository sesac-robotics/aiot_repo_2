# 어깨 운동 데이터셋 — QEVD 호환 초안

어깨 운동 원본 영상과 기존 시각 검토 결과를 **로컬 `qevd-selected`와 같은 JSON 키·타입 및 CSV 컬럼**으로 정리한 버전입니다. 원본 QEVD와 다운로드 폴더는 변경하지 않았습니다.

현재 버전은 `0.1-draft`입니다. **형식 변환과 후보 라벨 정리는 완료했지만, 검증된 학습 정답 데이터셋은 아닙니다.**

## 구성

| 구분 | 영상 수 | 처리 |
|---|---:|---|
| 원본 `good` 기반 정상 후보 | 70 | QEVD 형식 라벨 포함. 원본 이진 라벨에서 만든 약한 라벨 |
| 시각 비교로 발견한 오류 후보 | 30 | QEVD 형식 라벨 포함. 상세 근거가 있는 10개 사례 + 샘플 선별 후보 20개 |
| 오류 미확정 | 40 | `pending/`에 보관, 라벨 JSON·학습 목록에 포함하지 않음 |
| 라벨 충돌 중복 | 6 | `quarantine/`에 보관, 양쪽 라벨 모두 보류 |
| 운동 종류 확인 필요 | 4 | `quarantine/`에 보관 |
| **전체 원본 보존** | **150** | 영상은 재인코딩하지 않고 복사하고 SHA-256 일치 확인 |

`good` 영상의 실제 정석 여부를 독립적으로 검증한 것은 아닙니다. `bad` 전체에 하나의 오류를 일괄 부여하지 않았으며, 같은 참여자의 다른 시점으로 라벨을 자동 전파하지 않았습니다. 근거는 샘플 프레임 검토이며, 전 구간 프레임별 라벨 검토가 아닙니다.

## 폴더 구조

```text
shoulder-qevd/
├── raw/
│   ├── fit/part1/90000000.mp4 ...       # 라벨 초안이 있는 원본 영상 100개
│   ├── coach/
│   │   ├── fine_grained_labels.json
│   │   └── feedbacks_short_clips.json
│   └── annotations/
│       └── fine_grained_labels_with_worker_ids.json
├── manifests/
│   ├── short_clips.csv
│   ├── annotation_review.json           # 초안 상태·라벨 출처·한국어 피드백·근거
│   ├── source_video_mapping.json        # 전체 150개 원본과 새 ID 연결
│   ├── label_definitions.json           # 오류 6개 + 정상 후보 1개
│   ├── exercise_catalog.json            # 향후 5종목 통합용 운동명 매핑
│   ├── official_test_workers.json       # 파일명만 QEVD 호환. 어깨 데이터 자체 분할
│   └── split_policy.json
├── pending/
│   ├── videos/                         # 미확정 40개
│   └── annotations_pending.json
├── quarantine/
│   ├── videos/                         # 보류 10개
│   └── annotations_quarantine.json
└── checks/
    ├── sha256.csv
    ├── build_report.json
    └── validation_report.json
```

## QEVD와 동일한 필드

`raw/coach/fine_grained_labels.json`은 객체들의 배열이며, 각 객체에는 다음 네 필드만 있습니다. 검수 상태 같은 추가 정보는 **별도 sidecar 파일**에 저장해 기존 필드 구성을 유지했습니다.

```json
{
  "video_path": "./90000000.mp4",
  "labels": ["shoulder press - arms too wide at the top"],
  "labels_descriptive": [
    "shoulder press - Arms spread farther outward at the top than in the reference good video"
  ],
  "split": "train"
}
```

위 ID는 형식 설명용입니다. 실제 ID와 라벨 연결은 `short_clips.csv`에서 확인하세요.

`feedbacks_short_clips.json`은 QEVD와 동일하게 `video_path`, `feedbacks`, `split`을 사용합니다. **이 패키지의 영어·한국어 피드백은 기존 검토 결과에 맞춰 작성한 템플릿이며, 원본 데이터 제공자나 운동 전문가가 제공한 정답 문장이 아닙니다.** 한국어 문구는 `annotation_review.json`과 `label_definitions.json`에 있습니다.

참여자 파일에는 `video_path`, `worker_id`만 있으며 `worker_id`는 정수입니다. CSV 컬럼은 아래 순서로 QEVD와 같습니다. 두 라벨 컬럼은 JSON 배열을 직렬화한 문자열입니다.

```text
video_id,video_path,split,worker_id,labels,labels_descriptive
```

JSON의 `./ID.mp4`를 실제 파일 경로로 바로 열지 마세요. QEVD와 같이 manifest를 사용해 `raw/fit/part1/ID.mp4`로 연결합니다.

## 라벨 종류

| 코드 | QEVD 스타일 라벨 접미사 | 한국어 의미 |
|---|---|---|
| NO_OBVIOUS_ISSUE | no obvious issue | 원본 good 기반 정상 후보 |
| TOP_WIDE | arms too wide at the top | 상단에서 양팔이 바깥으로 벌어짐 |
| TOP_INCOMPLETE | incomplete upward range | 기준보다 상단 도달 범위가 작음 |
| HANDS_BEHIND_HEAD | hands move behind the head | 손이 머리 뒤로 내려가는 패턴 |
| FOREARM_INWARD | forearms tilt inward | 손목이 팔꿈치보다 안쪽으로 모임 |
| ELBOW_PATH_NARROW | elbow path narrower than reference | 하단 팔꿈치 경로가 기준보다 좁음 |
| ARM_ASYMMETRY | uneven arm height and timing | 좌우 팔 높이·타이밍 불균형 |

모든 라벨은 `shoulder press - `로 시작합니다. 팔꿈치 간격 차이는 그립·운동 변형 차이일 수도 있고, 머리 뒤 손 이동은 다른 운동 혼입일 수도 있습니다. 보편적인 자세 오류로 확정하지 않았습니다.

`annotation_review.json`에는 출처가 구분되어 있습니다.

- `source_binary_good_weak_label`: 원본 good에서 파생한 정상 후보 70개
- `visual_comparison_example`: 상세 비교 근거가 있는 오류 초안 10개
- `sampled_visual_screening_candidate`: 샘플 선별에서 발견한 오류 후보 20개

모든 객체는 `annotation_status: draft`, `expert_verified: false`, `training_ready: false`입니다. 미확정 40개를 정상으로 바꾸지 않았습니다. 동작이 관찰되지 않았다는 것과 오류가 없다는 것은 다릅니다.

## 시간 라벨과 MediaPipe

이 버전의 라벨은 **클립 수준의 후보**입니다. 오류 시작·종료 시각은 미작성으로 `segments: null`입니다. 빈 배열을 넣어 오류가 없다고 표현하지 않았습니다. MediaPipe 좌표·각도·판정 임계값도 없습니다.

10개 상세 사례에는 원본 근거 시각과 검토용 6초 구간이 연결되어 있습니다. 이 6초 전체가 오류라는 뜻이 아닙니다. 비교 이미지·검토용 영상도 `docs/evidence/`에 복사해 패키지 안에서 볼 수 있습니다. `preview.html`에서 한국어 문구 검색, 원본 영상, 상세 비교 자료를 열 수 있습니다.

## 나중에 구성할 5종목

| 표시 종목 | 기존 라벨 prefix | 자료 |
|---|---|---|
| 스쿼트 | squats | qevd-selected |
| 푸시업 | pushups | qevd-selected |
| 엘보 플랭크 | elbow plank | qevd-selected |
| YTW | standing ytw 및 순서 오류 변형 5종 | qevd-selected |
| 덤벨 숄더 프레스 | shoulder press | shoulder-qevd |

`exercise_catalog.json`에 매핑을 저장했습니다. YTW의 순서가 잘못된 변형은 원래 라벨을 보존하고 표시 종목만 YTW로 묶습니다.

어깨 `video_id`는 90000000번대, `worker_id`는 900000번대를 사용해 현재 QEVD의 ID와 충돌하지 않게 했습니다. ID 범위가 다르다는 것이 두 자료에 같은 사람이 전혀 없다는 의미는 아닙니다.

어깨 자료는 참여자 20명 train / 5명 test로 구분했습니다. 같은 참여자의 good/bad와 모든 시점은 같은 split에 있습니다. `official_test_workers.json`은 호환 파일명이며, **공식 QEVD 테스트 분할은 아닙니다.** 기존 QEVD split은 수정하지 않았습니다.

소수 클래스는 평가에 충분하지 않습니다. 특히 좌우 불균형 후보는 1개이며 test에 없는 클래스도 있습니다. 이 분할은 통합 준비용 기본 분할입니다.

## 읽기 및 검증

```bash
# 명시적으로 초안을 포함해 데이터셋 구조 확인
python3 /home/yoo/R2D2/shoulder-qevd/dataset_io.py --include-draft

# 원본과 사본 해시·QEVD 필드·ID·분할·파일 연결 확인
python3 /home/yoo/R2D2/shoulder-qevd/validate_dataset.py
```

기본 로더는 미검수 초안을 학습 정답으로 자동 읽지 않습니다. `--include-draft`로 100개 초안을 읽을 수 있습니다. 향후 검수된 레코드는 sidecar의 상태와 `training_ready`를 업데이트하면 기본 로더에 포함할 수 있습니다.


## English posture labels

자세 이름·오류 설명·교정 문구는 영어로 통일했습니다. `docs/ENGLISH_LABEL_GUIDE.md`에서 전체 이름을 확인할 수 있습니다. 기존 라벨 문자열과 오류 코드는 유지했으며, 검수 상태는 계속 초안입니다. 데모에서도 영어 자세 이름·설명·교정 문구가 표시됩니다.
