# 5종 Rule-based 자세 오류 평가

Reps/Hold와 분리된 실험용 오류 관측기입니다. 새 모델이나 확률/점수를 만들지 않습니다.
기존 운동 Analyzer는 1회만 실행하고 Bridge.last_result에 담긴 **같은 프레임의**
AnalysisResult(운동 이름, phase, monotonic timestamp)를 읽습니다. 분석 결과가 없거나
다른 운동/시각이면 단계 관련 평가는 보류합니다. 오류로 Reps를 취소하거나 추가하지 않습니다.

## 변경 파일

추가: rule_based/__init__.py, config.py, geometry.py, base.py, exercises.py, registry.py,
본 문서, tests/test_rule_posture.py.
수정: app/analysis/exercise.py(읽기용 last_result), app/streaming/pose_stream.py(결과 전달),
app/analysis/posture/service.py(평가기 선택), app/main.py(명시 모드 우선순위), .env.example,
상위 자세 평가 README.

5종 기존 Reps/Hold Analyzer, SessionManager, posture_state, Pydantic Schema/REST Router,
카메라, MediaPipe, CNN/GRU predictor/vendor/가중치는 이번 작업에서 변경하지 않았습니다.
새 라이브러리를 설치하지 않습니다.

## 실행 모드와 기존 설정 호환성

```ini
POSTURE_EVALUATION_MODE=rule_based
POSTURE_CAMERA_VIEW=unknown
POSTURE_CAMERA_VIEWS='{"squat":"side","push_up":"side","plank":"side","ytw_raise":"front","dumbbell_shoulder_press":"front"}'
```

- 명시적 POSTURE_EVALUATION_MODE가 기존 POSTURE_MODEL_ENABLED보다 우선합니다.
- `rule_based`: 새 평가기만 실행합니다. CNN/GRU를 로딩하지 않습니다.
- `model`: 기존 CNN/GRU만 실행합니다. ARCHITECTURE/DEVICE/MODEL_DIR 설정을 유지합니다.
- `off`: 자세 워커와 추가 Live 필드가 없습니다. 기존 Reps/Hold는 계속 실행합니다.
- MODE 키 자체가 없으면 기존 POSTURE_MODEL_ENABLED=true/1 → model,
  false/0/미설정 → off 정책을 유지합니다. 기존 .env는 자동 변경하지 않았습니다.
- 잘못된 명시 모드/설정은 unavailable로 격리하고 기존 운동 측정을 중단하지 않습니다.
- 모든 설정은 시작 시 읽습니다. 변경 후 서버를 재시작합니다.

촬영 방향은 2D 좌표만으로 확정하지 않습니다. **실제 촬영 위치에 맞춰** front/side를
설정해야 합니다. 공통 POSTURE_CAMERA_VIEW보다 운동별 POSTURE_CAMERA_VIEWS가 우선합니다.
위 맵은 배치 예시이며 실제 영상을 확인했다는 뜻이 아닙니다. 미설정 방향은 unknown이고
방향이 필요한 항목은 정상으로 간주하지 않습니다. 한 세션 안에서 운동이 바뀌어도
운동별 방향 맵을 사용하므로 운동 전환마다 서버를 재시작할 필요는 없습니다.

## 좌표 및 관절 유효성

기존 squat.knee_angle의 2D 각도 계산 함수를 재사용합니다. 위치 벡터는 x에 width/height를
반영하고 y를 유지하므로 종횡비를 고려합니다. Reps 임계값은 가져오지 않습니다.
필수 관절은 중복 ID, 화면 밖 x/y, NaN/Infinity, visibility <0.6, 제공된 presence <0.5이면
거부합니다. presence가 없는 경우는 기존 Analyzer 정책처럼 추가 값을 생성하지 않습니다.
z는 기하 특징에 사용하지 않으나 제공됐을 때 비유한 값은 거부합니다.

측면 body/squat은 유효 관절의 visibility/presence 최솟값이 높은 한쪽을 선택합니다.
관측이 유효한 동안 같은 측면을 유지하여 좌우 전환으로 동작 범위를 조립하지 않습니다.
선택한 측면이 무효이면 해당 평가/범위/타이머를 초기화하고 반대쪽 좌표로 미완료 범위를
완성하지 않습니다. 다음 유효 관측에서 다시 선택할 수 있습니다. 반대쪽 낮은 신뢰도는
필수 조건이 아닙니다. 누락된 관절을 채우거나 정상이라고 추정하지 않습니다.

## 오류별 기준: 모두 실영상 검증 전 초기 실험값

| 코드 | 관절/관측 | 최초 오류 진입 / 해제 기준 |
|---|---|---|
| SQUAT_INSUFFICIENT_DEPTH | 선택 측 Shoulder/Hip/Knee/Ankle | 관측 최저 무릎 >115° **및** 최저 고관절 >120°; 무릎 이동폭 ≥25°. 실제 bottom/완료 범위에서만 판정 |
| SQUAT_EXCESSIVE_FORWARD_LEAN | 선택 측 Shoulder/Hip | 수직 대비 기울기 >50° / <40°. 일반 힙힌지를 전부 오류로 보지 않음 |
| SQUAT_KNEE_VALGUS | 양 Hip/Knee/Ankle | 무릎 가로폭/발목 가로폭 <0.65 / >0.75. 하강·하단·상승 구간, 정면에서만 |
| PUSHUP_HIP_SAG | 선택 측 Shoulder/Hip/Ankle | 골반 signed 이탈비 <-0.12 / >-0.08 |
| PUSHUP_HIP_RAISED | 동일 | 골반 signed 이탈비 >0.12 / <0.08 |
| PUSHUP_INSUFFICIENT_DEPTH | 선택 측 Shoulder/Elbow/Wrist | 관측 최저 팔꿈치 >120°, 이동폭 ≥25°. 실제 bottom/완료 범위에서만 |
| YTW_ELBOW_BENT | 양 Shoulder/Elbow/Wrist 및 Hip 기준축 | Y/T 중 팔꿈치 최소 각도 <150° / >162° |
| YTW_ARM_ASYMMETRY | 동일 | 정규화 손목 높이 차 >0.45 또는 팔꿈치/상완 방향 각도 차 >25° / 각각 <0.30 및 <15° |
| YTW_WRIST_POSITION_ERROR | W의 양 Shoulder/Elbow/Wrist | (Elbow_y-Wrist_y)/상완길이 <0.10 또는 팔꿈치가 40~145° 밖 / 높이 >0.20 및 각도 55~130° |
| PLANK_HIP_SAG | 선택 측 Shoulder/Hip/Ankle | 골반 signed 이탈비 <-0.12 / >-0.08 |
| PLANK_HIP_RAISED | 동일 | 골반 signed 이탈비 >0.12 / <0.08 |
| PLANK_BODY_MISALIGNMENT | 동일 | Shoulder-Hip-Ankle <145° / >155°. 골반 방향 오류가 확정되면 이 일반 오류는 숨김 |
| PRESS_INCOMPLETE_EXTENSION | 양 Shoulder/Elbow/Wrist | TOP에서 덜 펴진 팔 <155°, 또는 완료된 상향 범위의 최댓값 <155°(이동폭 ≥25°) |
| PRESS_ARM_ASYMMETRY | 양 Shoulder/Elbow/Wrist 및 Hip 기준축 | 정규화 높이 차 >0.45 또는 팔꿈치/상완 방향 차 >25° / 각각 <0.30 및 <15° |
| PRESS_TORSO_LEAN | 양 Shoulder/Hip 중심 | 정면 수직축 대비 좌우 기울기 >15° / <10° |

무릎/팔꿈치 각도는 Shoulder/Hip/Knee 등 세 점의 내각입니다.
골반 signed 이탈비는 Shoulder→Ankle 기준선의 위쪽 법선에 대한 거리/몸 길이입니다.
위쪽(골반 상승)이 양수이고 좌우 반전에도 방향이 유지됩니다. 손목 높이의 양팔 차이는
어깨-골반 중심 길이로 정규화합니다. 팔꿈치 굽힘 차이도 함께 사용하므로 단순한 높이
차이만으로 모든 비대칭을 설명하지 않습니다.

보조 범위 제한도 Config에 분리했습니다. body는 기준선 기울기 ≤40°를 요구합니다.
plank은 Shoulder/Elbow/Wrist도 관측하고 팔꿈치 45~140°, Elbow가 Shoulder보다 몸 길이의
0.03 이상 아래에 있어야 팔꿈치 지지로 봅니다. 서기/앉기/엎드려 쉬기를 단순 몸통선만으로
정상 플랭크라 확정하지 않습니다. 기본 phase도 stabilizing/holding/interrupted여야 합니다.

YTW는 기존 y_confirmed/t_confirmed/wait_next_y(기존 W 완료 후) 단계만 사용합니다.
Y/T는 확인된 상완 방향이 각각 25~85° 위/수평 ±25° 안에 있을 때만 그 단계의 기준을
적용합니다. W는 상완 방향 -65~25°, 팔꿈치 20~165°의 넓은 주변 범위를 요구합니다.
따라서 단계 사이 이동과 W 이후 새 Y에서 오래된 단계 기준을 적용하지 않습니다.
이 조건은 새 YTW 상태 머신이 아니며 Y→T→W 카운트/상태 전이를 수정하지 않습니다.
원래 Analyzer가 단계를 전혀 확인하지 못한 경우 임의로 오류를 생성하지 않습니다.

정면 valgus의 보조 제한은 골반 가로폭 ≥영상 높이의 0.08, 발목 폭 ≥골반 폭의 0.5입니다.
가림/비스듬한 투영/극단적으로 좁은 지지면은 평가를 보류합니다. 2D 무릎 모임을 3D
진단으로 해석하지 않습니다. front에서 허리 전후 과신전을 평가하지 않습니다.

## 시간 안정화와 동작 범위

- 일반 오류는 연속 0.4초 이상 + 서로 다른 시각 2표본 이상에서 확정합니다.
- 정상 복귀는 더 좁은 해제 임계값을 0.25초 만족해야 오류를 지웁니다.
- 정상 표시 자체도 해당 phase의 모든 항목이 충분히 관측된 후 0.4초 확인합니다.
- 중복 frame_id는 기존 워커에서, 중복 timestamp는 평가기에서 무시합니다.
- 시간 역행, 0.75초 초과 간격, 필수 관절 누락은 오류 타이머/미완료 범위를 끊습니다.
  누락 구간을 정상/오류 지속 시간으로 더하지 않습니다.
- 스쿼트/푸시업 깊이와 완료된 프레스 신전은 실제 관측 범위에 근거하므로 일률적으로
  0.4초 대기를 적용하지 않습니다. 하강/들어 올리는 도중에는 깊이/신전 부족을 출력하지
  않습니다. 최소 관측 이동폭 25°로 단순 미세 팔꿈치 움직임을 구분합니다.
- 완료 범위는 다음 움직임 시작/무효 관절/세대 변경 또는 최대 12초 후 폐기합니다.
  `last_cycle_evaluated_at`으로 마지막 완료 범위의 관측 시각을 구분합니다.
- 기존 generation 검증, Pose 결과의 프레임 ID, 운동 이름/시각 검증을 재사용합니다.
  Pause/Resume/운동·세션 전환에서 평가 인스턴스를 초기화하고 지연 결과는 거부합니다.
  SessionManager 잠금 중 자세 평가를 실행하지 않습니다.

## Live 응답 및 평가 범위

기존 Schema에 맞춰 classification/display_feedback은 **dict**를 유지합니다.
예시처럼 문자열로 바꾸지 않습니다. confidence와 form_score는 항상 null입니다.
다음은 합성 테스트의 플랭크 오류 구조 예시이며 실제 사용자의 측정 결과가 아닙니다.

```json
{
  "posture_analysis": {
    "status": "ready",
    "reason": null,
    "is_correct": false,
    "form_score": null,
    "posture_errors": [{
      "code": "PLANK_HIP_SAG",
      "label": "엉덩이 처짐",
      "confidence": null,
      "feedback": "골반을 조금 들어 몸통을 일직선으로 유지하세요."
    }],
    "predicted_at": 100.6,
    "classification": {
      "method": "rule_based", "exercise": "plank", "model_name": "rule_based:plank",
      "status": "ERROR", "phase": "holding", "camera_view": "side", "thresholds_validated": false
    },
    "display_feedback": {
      "text": "골반을 조금 들어 몸통을 일직선으로 유지하세요.",
      "corrective": true, "error_type": "PLANK_HIP_SAG", "provisional": true,
      "reason": "rule_based_experimental"
    }
  }
}
```

추가 진단 필드는 evaluated_checks, unevaluated_checks(코드별 보류 사유), pending_checks,
measured_values, assessment_scope, last_cycle_evaluated_at입니다. 기존 session/exercise,
Reps/Duration 필드는 그대로입니다. 프레임 ID/observed_at은 기존 자세 워커가 붙입니다.

필수 관절 부족이면 insufficient_pose/is_correct=null입니다. 스쿼트는 현재 촬영 방향의
적용 항목만 집계합니다. 측면은 깊이/상체, 정면은 무릎 모임이고 반대 방향 전용 항목은
not_applicable로 제외합니다. 적용 항목에서 단계/관측 부족이 있으면 정상으로 확정하지
않습니다. 확정된 오류는 다른 적용 항목이 보류/불가여도 is_correct=false로 유지합니다.
정상 true는 현재 촬영 방향의 적용 항목이 관측되고 안정화됐다는 의미이며, 반대 방향에서만
평가 가능한 오류까지 없다는 뜻은 아닙니다. Y/T에 W 손목 기준, W에 Y/T 팔꿈치 신전
기준은 적용하지 않습니다.

오류 목록 순서가 피드백 우선순위입니다. 골반 처짐/상승은 플랭크 일반 정렬 오류보다
우선합니다. 스쿼트는 무릎 모임→기울기→깊이, 프레스는 몸통 기울기→비대칭→신전입니다.
교정 문구는 이번 통합에서 작성한 임시 안내이며 전문가 검증을 받았다는 뜻이 아닙니다.
기존 모델 결과와 혼합하지 않고 Backend POST/GUI/DB를 새로 만들지 않습니다.

## 임계값 조정 및 실제 검증

```ini
POSTURE_RULE_CONFIG='{"confirm_sec":0.5,"clear_sec":0.3,"squat_depth_knee_deg":115,"press_extension_deg":155}'
```

기본값과 모든 필드명은 config.RuleConfig에 있습니다. JSON에는 바꿀 필드만 넣으면 됩니다.
부적절한 값/알 수 없는 필드는 unavailable로 처리합니다. 테스트에서는 dataclasses.replace로
Config를 주입할 수도 있습니다. Reps/Hold Config나 학습 모델 임계값은 공유하지 않습니다.

실장비에서 운동별 front/side 배치, 정상 가동범위, 카메라 기울기, 의복/가림, 신체 비율,
좌우 반전, 낮은 FPS와 관측 간격을 정답 영상과 비교해야 합니다. 정상 힙힌지·개인별
스쿼트 깊이·앉은/선 프레스·W 팔 위치는 특히 추가 검증이 필요합니다. 실제 영상을
검증하지 않았으며 합성 테스트는 기하/타이머/연동 계약만 확인합니다.

```bash
python -m unittest discover -s tests -p 'test_rule_posture.py' -v
POSTURE_MODEL_ENABLED=false CAMERA_SOURCE_TYPE=http ESP32_STREAM_URL=http://camera/stream \
python -m unittest discover -s tests -v
```

새 테스트 파일은 저장소의 기존 tests/ ignore 규칙에 포함되므로 커밋 시 명시적으로
추가해야 합니다. 원래 CNN/GRU 경로 테스트는 기존 테스트를 그대로 재실행합니다.


## 최초 Rule-based 통합 테스트 결과 (아래 스쿼트 집계 수정 전)

신규 테스트 38개 모두 통과했습니다. 전체 429개 중 421개 통과, 기존 실패 8개입니다.
작업 시작 시 복사한 기존 코드에서도 전체 391개 중 같은 8개 실패를 재현했습니다.
로거 1개, PushUp 설정 관련 2개, YTW 진단 로그 관련 5개이며 신규 회귀 실패는 없습니다.
실제 인계 CNN/GRU CPU 로딩/추론·전처리·경로 테스트도 기존 전체 테스트에 포함됐습니다.
별도 프로세스에서 rule_based 선택 시 torch 자체가 import되지 않음을 확인했습니다.
보호 대상 18개 파일의 바이트 동일성 및 상세 실패 이름은 [테스트 요약](test_summary.json)에 있습니다.
실제 카메라의 정상/오류 영상 검증과 임계값 정확도 평가는 미실시입니다.


## 스쿼트 촬영 방향 집계 수정 및 Reps 진단

이전 구현은 반대 촬영 방향 전용 검사도 unavailable에 넣어 complete를 막았으므로
스쿼트가 정면/측면 모두 정상 true에 도달할 수 없었습니다. 현재는 다음을 구분합니다.

| check_statuses 상태 | 의미 | 정상 집계 |
|---|---|---|
| not_applicable | 반대 촬영 방향 전용 | 제외 |
| pending | 하단/완료 범위·운동 단계 대기 또는 오류 안정화 중 | 정상 확정 보류 |
| unavailable | 필수 관절/투영이 유효하지 않음 | 정상 확정 보류 |
| evaluated | 유효한 관측 완료 | 포함 |
| error | 확정된 오류 | false 우선 |

unevaluated_checks에는 현재 방향의 pending/unavailable만 남깁니다.
not_applicable_checks와 unavailable_checks, pending_checks, check_statuses에 상세 사유를
나눕니다. 깊이는 하강 중 매 프레임 평가하지 않으며 실제 하단/완료 관측을 기다립니다.
어깨/골반이 유효한 상체 기울기 관측은 발목 누락에 따른 깊이 평가 실패와 격리했습니다.
이 변경은 자세 집계에만 적용하고 Reps 입력 조건은 변경하지 않았습니다.

POSTURE_CAMERA_VIEWS의 squat 값이 POSTURE_CAMERA_VIEW보다 우선합니다. 예를 들어
공통 VIEW=side여도 VIEWS 안의 squat=front이면 정면으로 평가합니다. 변경 후 재시작해야
합니다. 실제 .env는 이번 작업에서 변경하지 않았습니다.

Reps는 MediaPipe → keypoints → 기존 SquatAnalyzer → RepCompleted → Bridge →
record_analysis_rep/record_rep → Live exercise.rep_count 경로입니다. 자세 평가의
is_correct/status로 Reps를 필터링하지 않습니다. 기존 SquatAnalyzer는 양쪽 Hip/Knee/Ankle
6개를 모두 요구하고 visibility≥0.6, 제공된 presence≥0.5, 좌우 각도 차≤25°, 단계별
0.15초/2표본, 프레임 간격≤0.75초, 반복 0.8~10초를 요구합니다. 시각화는 visibility 0.35
기준이므로 스켈레톤이 보이는 것만으로 이 조건을 만족했다고 할 수 없습니다.
측면에서는 반대 다리 가림, 정면에서는 2D 투영에 따른 작은 무릎 각도 변화도 점검해야
하지만 당시 실영상/로그가 없어 이번 실패 원인으로 확정하지 않았습니다.

```ini
SQUAT_DEBUG=1
```

재시작 후 [SquatDebug] 로그에서 다음을 비교합니다.

- 입력: frame_id/frame_timestamp, 관절 23~28의 x/y/visibility/presence 및 invalid_reasons
- 호출/단계: analyzer_called, analyzer_name, previous_phase/current_phase,
  좌우/평균 무릎 각도, 좌우 차이, candidate_phase/samples/duration
- 시간: frame_gap_sec, cycle_duration_sec/valid, 실제 적용 thresholds
- 이벤트: rep_completed, event_id, manager_called, event_accepted, event_rejected_reason
- 누적: 같은 session_id/exercise_id/revision의 rep_count

rep_completed=false이면 이벤트 생성 전 입력/단계/시간을 확인합니다.
true이고 manager_called=false이면 세대 변경 등 Bridge의 반영 전 거부를 확인합니다.
manager_called=true/event_accepted=false이면 중복/상태 변경 등 관리자 거부를 구분합니다.
일반 로그는 초당 최대 1회이고 phase/무효 사유 변경과 이벤트는 즉시 출력합니다.
카메라 프레임이 오지 않으면 별도 pipeline 로그에서 analyzer_called=false를 출력합니다.
DEBUG OFF에서는 추가 관절 계산/로그를 실행하지 않습니다.

### 정면 스쿼트 자세 평가 진단

`SQUAT_DEBUG=1`로 서버를 재시작하면 `logs/debug.jsonl`의
`debug_type=SquatDebug`, `data.kind=posture`에서 자세 평가 경로를 확인할 수 있습니다.
Reps 경로의 `kind=reps`와 별도이며 판정 임계값은 변경하지 않습니다.
일반 기록은 약 1초 간격, 단계/평가 사유 변경은 즉시 기록합니다.

- `effective_camera_view`, `camera_view_source`: 운동별 JSON 설정이 공통 설정보다 우선합니다.
- `check_enabled`, `check_evaluated`, `check_pending_reason`: 정면 무릎 모임 검사 활성화/실행 및 보류·불가 사유입니다.
- `knee_ankle_width_ratio`, `projection`: 실제 검사에 사용한 무릎/발목 폭 비율과 투영 검증값입니다.
- `joints`: Hip/Knee/Ankle(23~28)의 원본 좌표·Visibility/Presence 및 실제 유효성 검사 결과입니다.
- `phase`, `analyzer_phase`, `analyzer_timestamp`, `timestamp`: 동일 운동·동일 시각의 분석 단계인지 확인합니다.
- `temporal_pending_reason`, 오류/정상 후보 표본 수·지속시간, `frame_gap_sec`: 평가 실행과 시간 안정화 대기를 구분합니다.

정면에서는 골반 폭이 이미지 높이 기준 `squat_front_min_hip_width=0.04` 이상,
골반~발목 높이의 `squat_front_min_hip_to_leg_ratio=0.10` 이상이어야 합니다.
골반~발목 높이는 `squat_front_min_leg_height=0.15` 이상이어야 하며 양쪽 모두
Hip/Knee/Ankle의 위아래 순서와 좌표/신뢰도를 확인합니다. 발목 폭은 골반 폭의
`squat_front_min_ankle_to_hip_ratio=0.5` 이상이어야 합니다.
기존 고정 0.08은 영상 높이에 대한 실험용 절대 폭으로 신체 크기 대비 폭이 아니며,
실영상으로 보정된 근거는 없었습니다. 새 설정도 검증 전 실험값이며 `POSTURE_RULE_CONFIG`로
조정할 수 있습니다. `projection`에는 모든 사용값과 조건별 결과를 기록합니다.
투영 조건 실패는 `front_projection_or_stance_unreliable`로,
필수 관절 실패는 `joint_숫자_사유`로 기록됩니다. 서 있는 단계는
`active_squat_phase_required`로 보류하며, 같은 시각의 운동 단계가 없으면
`phase_unconfirmed`입니다. 정상/오류 확정에는 각각 기본 0.4초와 2개 이상 표본이
필요합니다. 측면 전용 두 검사는 `not_applicable_checks`에 들어가며 정면 판정을 막지 않습니다.

### Squat 관측 실패 유예와 비교 검증

Reps의 각도/좌우 차이 기준은 유지합니다. `SquatConfig.occlusion_grace_sec=0.35`
(0보다 크고 `max_frame_gap_sec` 이하여야 함) 동안의 부분 관절 누락, 신뢰도 저하,
좌우 각도 불일치는 phase를 보존하되 결과는 `invalid`로 반환합니다.
무효 표본은 각도 평균이나 안정화에 사용하지 않고 후보 시간/표본 수를 지웁니다.
복구 후 실제 유효 표본으로 각 단계를 다시 안정화해야 하며, 보지 못한 Bottom을 추정하지 않습니다.
유예 만료 시점에 다시 관측되더라도 사이클을 폐기합니다. NaN/화면 이탈/중복 관절/
퇴화 벡터/하체 전체 누락, 큰 프레임 간격, 시간 역행, 세션 세대 변경은 계속 초기화합니다.
`SquatDebug`의 `invalid_elapsed_sec`, `cycle_preserved_during_invalid`,
`last_valid_timestamp`로 보류 상태를 확인합니다. 지속적인 잘못된 추정을 복구하는 정책은 아닙니다.

`5564e83`, 수정 전 HEAD, 수정 후에 같은 합성 5회 시퀀스를 입력한 결과,
정상 입력은 모두 5회였습니다. 반복마다 짧은 좌우 각도 불일치 1표본을 넣으면
이전 두 버전은 0회, 수정 후는 5회였습니다. OFF/rule_based에서도 동일 입력의
단계와 이벤트 수락 결과가 같았습니다.
저장된 실제 정면 관측 6건은 골반 폭 때문에 모두 평가 불가였으나 변경 후 모두
평가됐습니다. 이 비교는 로그의 보정된 골반 폭/원본 x 차이로 w/h를 복원해 수행했습니다.
로그는 전 프레임 시퀀스가 아니므로 실제 영상 Reps를 재현했다고 볼 수 없습니다.
관측 비율 약 0.68~1.175에는 정상/오류 정답 라벨이 없어 `valgus_enter_ratio=0.65`와
해제 0.75는 유지했습니다. 비율 0.65~0.75의 경계 구간은 평가되더라도 정상으로 확정되지 않을 수 있습니다.

추가 테스트는 tests/test_squat_assessment_scope.py이며 아래처럼 실행합니다.

```bash
python -m unittest discover -s tests -p 'test_squat_assessment_scope.py' -v
```

실패 당시의 영상/연속 관절 로그가 없으므로 실제 Reps=0 원인이 해결됐다고 주장하지
않습니다. 합성 입력으로 OFF/정면 rule_based/측면 rule_based 모두 정상 5회가 누적되는지,
가려진 발목과 좌우 각도 불일치에서는 두 모드 모두 이벤트가 생성되지 않는지, 생성 후
중복/세대 거부와 Live 누적을 구분하는지 검증했습니다. SquatAnalyzer 임계값/상태 머신은
수정하지 않았습니다.


### 스쿼트 집계 수정 최종 결과

신규 20개 모두 통과했습니다. 최초 최종 실행에서는 전체 449개 중 441개 통과,
기존 실패 8개였습니다. 수정 전 복사본(429개)에서도 같은 기존 실패 8개를 확인했습니다.
이후 작업 중 별도로 camera/stream.py의 CameraMetrics 로그가 주석 처리된 변경을 감지했고
해당 변경은 덮어쓰지 않았습니다. 그 상태로 재실행한 현재 결과는 449개 중 440개 통과,
기존 8개 및 카메라 진단 로그 테스트 1개 실패입니다. 이번 스쿼트 수정에 따른 회귀는 없습니다.
기존 Rule 테스트의 정면 정상 기대값을 새 방향별 집계에 맞추고, 기존 모델 테스트 2개는
로컬 .env의 명시 MODE가 legacy 설정 테스트에 섞이지 않도록 환경변수를 격리했습니다.
SquatAnalyzer/다른 운동 Analyzer/MediaPipe/CNN·GRU/SessionManager/API 스키마 및 Router를
포함한 보호 대상 17개 파일은 시작 시점과 바이트 단위로 동일합니다.
[이번 테스트 요약](squat_scope_test_summary.json)에 상세 결과가 있습니다.

이번 변경 파일: rule_based/base.py, exercises.py(스쿼트 관측만), registry.py, 본 README,
squat_scope_test_summary.json; 신규 app/analysis/squat_debug.py, tests/test_squat_assessment_scope.py;
연결부 exercise.py, streaming/pose_stream.py, .env.example; 관련 기존 테스트
테스트 기대값/환경 격리만 수정했습니다. 실제 .env는 수정하지 않았습니다.
# 정면 스쿼트의 관측 연속성

정면에서는 Reps 단계가 없거나 invalid/unknown이어도 자세 전용 움직임 관측을 사용합니다.
유효한 양쪽 골반·무릎·발목으로 무릎 각도 155도 이상, 수직 신장 비율 0.85 이상인
입위를 0.3초/3샘플 이상 관측해 기준을 잡습니다. 다리 길이로 정규화한 골반~발목
수직 거리 비율이 기준보다 0.08 이상 감소하면 운동 중으로 보고, 감소가 0.035 이하로
돌아오면 입위로 봅니다. 이는 정면 무릎 정렬 검사를 위한 실험 기준이며 깊이 판정이 아닙니다.
기존의 유효한 Reps 단계도 사용할 수 있지만 자세 전용 관측이 운동 중이면 우선합니다.
관절 신뢰도 기준은 완화하지 않으며 긴 누락/시간 역행/세션 변경에는 기준을 다시 잡습니다.
`debug.jsonl`의 posture 기록에 `phase_source=front_pose_motion`, `front_motion`의
extension_ratio/baseline/hip_drop_ratio를 남깁니다. 기준 확보 대기는
`front_motion_baseline_required`로 구분합니다. 횟수/Rep 이벤트는 생성하거나 변경하지 않습니다.

정면 스쿼트 자세 평가만 짧은 누락을 `squat_pose_grace_sec`(기본 0.35초) 동안
허용합니다. 결측/단계 미확정 중에는 UNKNOWN을 반환하며, 결측 시간은 확인 시간에
누적하지 않습니다. 유예를 넘기거나 문맥/시간이 초기화되면 후보와 저장 결과를 지웁니다.
운동 중 확정한 정상/오류는 standing에서 최대 `squat_result_hold_sec`(기본 2초) 동안
표시합니다. `assessment_held=true`, `assessment_observed_at`, `assessment_age_sec`,
`assessment_scope=last_confirmed_active_phase`로 과거 관측임을 구분합니다.
새 동작을 시작하면 이전 결과는 해제합니다. 시작부터 standing이면 정상으로 간주하지 않습니다.
깊이·상체 숙임의 정면 제외 정책, 관절 신뢰도·오류 임계값, Reps/rep_count는 바꾸지 않습니다.
