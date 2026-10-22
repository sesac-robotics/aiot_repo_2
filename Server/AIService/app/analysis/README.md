# 공통 운동 분석 구조

지원 이름·유형·분석기 생성자는 `registry.py`의 등록표 한 곳에서 관리합니다.
`list_exercises()`로 메타데이터를 조회하고 `get_analyzer(name)`으로 새 인스턴스를 만듭니다.
미지원 이름은 `InvalidSessionInput`이며 기존 내부 API의 예외 변환으로 HTTP 400이 됩니다.

| 식별자 | 유형 | 구현 파일 |
|---|---|---|
| squat | repetition | exercises/squat.py |
| dumbbell_shoulder_press | repetition | exercises/shoulder_press.py |
| push_up | repetition | exercises/push_up.py |
| ytw_raise | repetition | exercises/ytw_raise.py |
| plank | hold | exercises/plank.py |

기존 `ExerciseAnalyzer.reset()` 및 `analyze(keypoints, context)`를 사용합니다.
새 반환 타입 `AnalysisResult`는 exercise_name, exercise_type, phase, timestamp,
선택적 repetition_event, target_duration_sec, hold_duration_sec를 담습니다.
기존 `RepCompleted`와 `None` 반환도 계속 지원합니다. Score/Posture Error는 포함하지 않습니다.

SquatAnalyzer, DumbbellShoulderPressAnalyzer, PushUpAnalyzer, YTWRaiseAnalyzer는 아래 상태
머신으로 Rule-based 반복 이벤트를 생성합니다. PlankAnalyzer는 팔꿈치 플랭크의
유효 연속 유지 시간을 측정하며 반복 이벤트를 생성하지 않습니다. 공통 timestamp는 단조 시계의 초 값이며 UTC DB 시각이 아닙니다.
다섯 분석기 모두 입력 표본의 monotonic timestamp를 사용합니다.

기본 연결 흐름:

1. 운동 시작 시 SessionManager가 등록표의 이름을 검증합니다.
2. 추론 워커의 첫 유효 프레임에서 `RegisteredExerciseAnalyzer`가 해당 운동의 새 분석기를 만듭니다.
3. 운동·세션·세대 변경 시 기존 분석기의 reset을 호출하고 폐기합니다.
4. Pause/Resume가 프레임 사이에 모두 일어나도 기존 revision 검증으로 이전 결과를 버립니다.
5. 분석은 SessionManager 잠금 밖에서 수행하고 반복 반영은 기존 원자적 검증을 사용합니다.

Plank의 목표는 `get_analyzer('plank', target_duration_sec=30)` 또는 세션 운동 시작의
선택 필드 target_duration_sec로 지정합니다. 유효 자세 구간은 PlankAnalyzer가 측정하며
SessionManager에 현재 연속 구간만 저장합니다. active_duration_sec나 rep_count를 대신 쓰지 않습니다.
Hold 전용 완료 통지는 `HoldCompleted`이며 기존 `RepCompleted`와 분리합니다.

실제 알고리즘 구현 시 각 운동 클래스에서 analyze/reset을 구현하세요. 반복 이벤트의
ID는 같은 세대 내에서 고유하게 유지하고, reset에서 미완료 동작 단계와 내부 시간 기준을
폐기해야 합니다. Y/T/W의 반복 계산 단위는 아직 정하지 않았습니다. 학습 모델의 점수·자세
오류는 별도 결과로 결합할 예정이며 현재는 모델 로딩·영상 버퍼·비동기 큐가 없습니다.

카메라/GPU 없는 회귀 실행:

```bash
python -m unittest discover -s tests -p 'test_*session*.py' -v
```


## SquatAnalyzer

기존 `analyze(keypoints, context)`, `reset()`, `AnalysisResult`, `RepCompleted`를 유지합니다.
내부 `KeypointFrame`은 list 호환 입력이며 `(width, height)`와 monotonic 표본 시각을 담습니다.
PoseStreamService에서 기존 추론 결과를 게시한 뒤 이 메타데이터를 분석기에만 전달합니다.
`/pose/latest`의 keypoints 및 REST Snapshot 필드는 바뀌지 않습니다.

좌우 Hip/Knee/Ankle(23~28)의 정규화 x/y를 영상 너비/높이로 보정하고, Knee를 꼭짓점으로
2D 내각을 계산합니다. 양쪽 관절 신뢰도를 모두 확인하며 두 각도의 평균을 사용합니다.
관절 누락, 낮은 신뢰도, 좌표 범위 밖/NaN/퇴화 벡터, 큰 좌우 각도 차이, 잘못된 영상 크기는
`invalid`로 반환하고 미완료 동작을 폐기합니다. 영상 크기 없는 일반 list도 비율을 가정하지 않고
invalid로 처리합니다. presence는 제공될 때 검사하며 visibility는 반드시 필요합니다.

상태 흐름은 `unknown → standing → descending → bottom → ascending → standing`입니다.
마지막 상승 완료에서만 이벤트가 생성됩니다. initial unknown은 선 자세를 먼저 확인하므로
앉은 상태에서 시작하거나 관절 누락 후 다시 나타나 일어나는 동작은 세지 않습니다.
깊이가 부족하면 descending에서 standing으로 이벤트 없이 돌아갑니다. 상승 중 다시 내려가면
bottom으로 복귀하며 최종 선 자세까지 완료해야 한 번으로 셉니다.

`SquatConfig`의 현재 코드 적용값(이전 제안값이 아니라 실제 소스 확인 기준):

| 설정 | 값 |
|---|---:|
| 선 자세 진입 / 이탈 | 165° / 150° |
| 최하단 진입 / 상승 시작 | 120° / 135° |
| 최소 visibility / 제공된 presence | 0.6 / 0.5 |
| 최대 좌우 무릎 각도 차이 | 25° |
| 각 상태 전환 안정화 | 0.15초 + 서로 다른 시각의 2표본 이상 |
| 유효 반복 시간(하강 시작 후보부터 최종 선 자세 확인) | 0.8~10초 |
| 허용 최대 표본 간격 | 0.75초 |

### Bottom 임계값 변경 및 실제 ESP32-CAM 확인

| 항목 | 이전 설정 | 현재 적용값 |
|---|---:|---:|
| Standing 진입 | 165° | 165° |
| Standing 이탈 | 150° | 150° |
| Bottom 진입 | 95° | 120° |
| Ascending 진입 | 110° | 135° |

이전 Bottom 95° 기준에서는 사용자가 매우 깊게 앉아야 반복이 측정되는 문제가 실제
ESP32-CAM 테스트에서 관찰됐습니다. Bottom을 120°, 상승 시작을 135°로 완화한 뒤
**사용자가 실제 스쿼트 반복이 정상 측정됨을 확인했습니다.** 이는 사용자가 수행한 해당
환경의 기능 확인이며, 이 문서 작성 과정에서 새 촬영 실험을 수행하거나 정답 데이터 기반의
정확도·오검출률을 산출한 결과는 아닙니다. 촬영 거리/각도/FPS의 상세 수치는 제공되지 않았습니다.

AIFit의 Rule-based Analyzer는 반복 동작의 한 사이클을 측정합니다. Bottom은 반복 관측에
필요한 전환 기준이며 올바른 자세나 충분한 운동 깊이를 인증하는 점수가 아닙니다.
Score와 자세 오류 분류는 향후 별도 학습 모델이 담당합니다. 이번 문서 갱신에서 스쿼트
알고리즘과 설정값은 변경하지 않았습니다.

같은 timestamp는 표본 수/시간을 진행시키지 않으며 이벤트를 다시 생성하지 않습니다.
역행 timestamp, 긴 공백, 반복 시간 초과 시 현재 동작을 폐기하고 선 자세부터 재확인합니다.
reset 시 미완료 단계와 시각은 지우되 인스턴스의 완료 이벤트 번호는 재사용하지 않습니다.
이벤트 ID는 인스턴스 UUID + 완료 번호로 생성하며 DB ID가 아닙니다. Bridge는 기존 세대/ID
재검증을 거쳐 SessionManager.record_analysis_rep → record_rep으로만 횟수를 반영합니다.
분석기가 rep_count를 직접 변경하지 않습니다.

합성 테스트는 1회/5회, 부분 동작, 한 프레임 노이즈, 중복 표본, 관절 손실, 재등장,
Pause/Resume/운동 변경, 영상 비율, 시간 이상, ID 중복 및 REST live 조회를 검증합니다.
합성 통과와 사용자 실카메라 기능 확인만으로 모든 환경의 정확도를 보장하지 않습니다.
촬영 환경과 FPS에 따라 다음 항목을 확인하고 추가 튜닝이 필요할 수 있습니다.

- 전신/양쪽 하체 가시성, 거리·촬영 각도에 따른 2D 투영 오차와 좌우 각도 차이.
- 사용자별 정상 깊이/선 자세에서 임계값과 시간 안정화에 따른 누락·오검출.
- 실제 FPS/네트워크 지연에서 0.75초 공백 및 0.15초 안정화 설정의 적합성.
- 가림/이탈/재등장, Pause/Resume 및 느리거나 빠른 반복의 수작업 정답 대비 횟수.
- 서버 표본 시각은 촬영 시각이 아닙니다. 같은 영상이 새 시각으로 재전달되는 디코더/카메라
  중복은 현재 입력만으로 완전히 구별할 수 없으므로 실제 스트림에서도 별도 확인해야 합니다.


## DumbbellShoulderPressAnalyzer

`exercises/shoulder_press.py`의 `DumbbellShoulderPressAnalyzer`와 `ShoulderPressConfig`가
구현을 담당합니다. 기존 Registry의 `ShoulderPressAnalyzer` 이름은 호환 별칭으로 유지하므로
`get_analyzer('dumbbell_shoulder_press')`가 같은 구현의 새 인스턴스를 반환합니다.
기존 analyze/reset, AnalysisResult/RepCompleted, Bridge, 세대 검증, REST 스키마를 재사용합니다.

### 좌표와 양팔 관측

사용 관절은 좌/우 Shoulder(11/12), Elbow(13/14), Wrist(15/16)입니다. x는 영상 너비,
y는 영상 높이로 픽셀 비율을 보정한 뒤 팔꿈치 내각을 계산합니다. 손목 높이는 각 팔에서
`(shoulder_y - wrist_y) * image_height / shoulder_elbow_pixel_length`로 구합니다.
따라서 어깨보다 위는 양수이며 같은 팔의 상완 길이가 1인 단위입니다. 단순한 픽셀 높이보다
거리에 따른 크기 변화 영향을 줄이지만, 2D 투영/팔의 회전에 따른 상완 길이 변화까지
없애지는 못합니다. Hip이나 새 영상/모델 입력은 추가하지 않았습니다.

두 팔의 각도/높이를 평균내어 한쪽 미완료를 덮지 않습니다. READY/TOP/전환 조건은
양쪽 각각 충족해야 합니다. 관절 누락, 낮은 Visibility/제공된 Presence, 비정상 좌표,
퇴화 벡터/상완 길이, 큰 좌우 차이는 invalid로 처리하고 부분 동작을 폐기합니다.
Visibility는 필수이며 Presence는 MediaPipe가 제공한 경우 검사합니다.

### 상태와 반복 완료 정책

`unknown → ready → pressing → top → lowering → ready(RepCompleted)`

- unknown: 먼저 양팔이 굽혀지고 손목이 어깨 부근인 READY를 안정적으로 확인합니다.
  TOP에서 처음 나타나 내려오는 동작만으로는 반복을 만들지 않습니다.
- ready → pressing: 양팔이 더 펴지고 손목이 올라간 것을 함께 확인합니다.
- pressing → top: 양팔 각도와 어깨 대비 손목 높이가 모두 TOP 기준을 만족해야 합니다.
  부분 신전 뒤 READY로 돌아오면 이벤트 없이 해당 시도를 끝냅니다.
- top → lowering: 양팔 각도가 굽혀지고 손목 높이가 낮아져야 합니다. TOP에 머무르는 동안
  반복은 발생하지 않습니다.
- lowering → ready: 양팔이 최초 READY 범위로 복귀하고 시간 조건도 만족해야 한 번 셉니다.
  불완전 복귀 뒤 다시 TOP으로 올라간 것은 별도의 반복으로 세지 않습니다.

각 전환은 최소 0.15초와 서로 다른 timestamp의 2표본 이상이 필요합니다. 프레임을 많이
고정 요구하는 방식 대신 경과 시간과 작은 최소 표본 수를 사용합니다. 관측이 드문 경우
중간 자세나 최종 자세를 충분히 보지 못하면 안전하게 누락할 수 있으므로 실제 FPS로 검증해야 합니다.

### 현재 적용값과 실제 ESP32-CAM 확인

아래 값은 현재 `ShoulderPressConfig` 소스 기준입니다. 초기 설정에서는 실제 테스트 중 반복이
증가하지 않았고, 사용자가 측정한 각도에 맞춰 완화한 뒤 정상 증가를 확인했습니다.
이 결과는 해당 환경의 사용자 기능 확인이며 모든 사용자·촬영 환경의 정확도 검증은 아닙니다.

| 조건 | 양팔 각각의 각도 | 양팔 각각의 정규화 손목 높이 |
|---|---|---|
| READY | 40~120° | -0.35~1.05 |
| PRESSING 진입 | 110° 이상 | 1.10 이상 |
| TOP 진입 | 135° 이상 | 1.35 이상 |
| LOWERING 진입 | 125° 이하 | 1.15 이하 |

사용자가 보고한 4:3 계산값(READY 좌/우 55.4°/45.0°, 손목 높이 0.26/0.35,
TOP 139.7°/149.3°, 높이 2.04/1.97)을 반영해 각도 기본값을 완화했습니다.
READY와 PRESSING의 각도 범위는 일부 겹치지만 손목 높이의 hysteresis 간격은 유지합니다.
TOP↔LOWERING의 각도·높이 hysteresis, 안정화 시간과 양팔 동시 충족 조건도 유지합니다.
실제 각도는 정규화 x/y에 입력 영상의 width/height를 곱해 계산하므로 640×480은
4:3 계산과 동일합니다. 손목 높이는 픽셀 수직 거리 / 픽셀 상완 길이입니다.
코드 테스트는 원본 좌표 대신 위 각도/높이를 재현한 합성 좌표를 사용합니다. 사용자 실영상
기능 확인과 별개로 촬영 구도·사용자 변화에 따른 재검증 및 Config 튜닝이 필요합니다.
초기 설정 대비 각도 변경 내역:

| 조건 | 초기 | 현재 |
|---|---:|---:|
| READY 최소 / 최대 | 60° / 120° | 40° / 120° |
| PRESSING 진입 | 130° | 110° |
| TOP 진입 | 155° | 135° |
| TOP 이탈(LOWERING 진입) | 140° | 125° |

손목 높이와 Visibility/Presence 조건은 유지했습니다. 촬영 방향에 따른 2D 투영 차이가
크므로 양팔 전체가 잘 보이는 방향에서 READY/TOP의 양쪽 측정값을 확인해야 합니다.
보고된 오른쪽 손목 y=-0.0539는 기존 [0,1] 좌표 검사에서 계속 무효입니다.
임계값 완화로 이를 허용하지 않으므로 손목이 화면 안에 들어오도록 구도를 조정해야 합니다.
손목만 올리거나 팔꿈치만 옆으로 펴면 TOP의 결합 조건을 만족하지 않습니다.

| 가시성·시간 조건 | 기본값 |
|---|---:|
| Visibility / 제공된 Presence | 0.6 / 0.5 |
| 최대 좌우 팔꿈치 각도 차이 | 30° |
| 최대 좌우 정규화 손목 높이 차이 | 0.6 |
| 전환 안정화 | 0.15초 + 서로 다른 시각의 2표본 이상 |
| 반복 시간(PRESSING 시작 후보부터 READY 복귀 확인) | 0.8~12초 |
| 최대 표본 간격 | 0.75초 |

같은 timestamp는 상태 안정화/이벤트를 진행시키지 않습니다. 시간 역행·긴 공백·반복 시간
초과 시 미완료 동작을 폐기합니다. 오래 TOP을 유지해 12초를 넘긴 반복도 폐기하므로
정지 반복을 지원하려면 이 설정을 실측에 맞춰 조정해야 합니다.

### 세션과 이벤트

이벤트 ID는 인스턴스 UUID와 완료 번호를 조합합니다. reset에서 완료 번호는 되돌리지
않으므로 같은 세대 안에서도 재사용하지 않습니다. 분석기는 SessionManager를 직접 수정하지
않으며 기존 Bridge의 session_id/exercise_id/revision/event_id 검증 후 record_rep 경로로만
반영합니다. Pause/Resume, 운동·세션 전환, 관절 손실 후에는 READY부터 다시 확인합니다.
Score, Grade, Posture Error, Feedback은 이 분석기에서 계산하지 않습니다.

### 검증 범위와 실제 카메라 확인

합성 테스트는 정상 1회/연속 5회, READY/TOP/복귀 순서, 양팔 완료, 부분 동작, TOP 유지,
각도·높이 단독 변화, 가시성/관절 손실, 노이즈, timestamp 중복/역행/불규칙 간격,
스케일/영상 비율, Pause/Resume/운동·세션 변경, 중복 이벤트, SessionManager 및 REST live,
기존 PoseStreamService 입력 전달과 결과 게시를 검증합니다. 실제 카메라/GPU 없이 실행하며
**합성 성공은 실제 영상 정확도를 보장하지 않습니다.** 사용자는 임계값 완화 후 실제
ESP32-CAM에서 반복 횟수가 정상 증가함을 확인했습니다. 이 문서 작업에서 별도 촬영 실험을
수행하지 않았으며 정답 데이터 기반 정확도·오검출률이나 상세 FPS 수치는 제공되지 않았습니다.

실제 ESP32-CAM에서는 양쪽 팔 전체가 보이는지, 팔을 올릴 때 손목이 화면 밖으로 나가지
않는지, READY/TOP에서 좌우 각도와 정규화 높이가 기준을 만족하는지 확인해야 합니다.
거리·촬영 각도·팔 길이 비율·가림·낮은 FPS·긴 TOP 정지에 따른 누락/오검출을 수동 횟수와
비교하세요. 새 timestamp로 재전달되는 동일 영상은 입력만으로 완전히 구별할 수 없습니다.


### 숄더 프레스 실영상 진단 로그

환경변수 또는 `.env`의 `SHOULDER_PRESS_DEBUG=1`에서만 `[ShoulderPressDebug]` JSON 로그를
출력합니다. 객체 생성 시 읽으므로 변경 후 서버를 재시작합니다. 기본값과 1 이외 값은 꺼짐입니다.
실제 키/인증 설정은 이 진단에 포함하지 않습니다.

```bash
SHOULDER_PRESS_DEBUG=1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

현재 `.env`의 모델/카메라 설정과 내부 세션 API를 사용해
`dumbbell_shoulder_press` 운동을 시작하고 수행합니다. `uvicorn.error` INFO 로거를 사용하며
별도 DEBUG 로그 레벨 설정은 필요 없습니다. INFO를 필터링하면 표시되지 않습니다.
각 단계의 일반 로그는 약 1초 간격입니다. phase 전환, reset, 반복 이벤트, 입력 무효 사유
변경은 즉시 기록하므로 그 순간에는 1초에 여러 줄이 나올 수 있습니다.

| kind | 확인할 내용 |
|---|---|
| pipeline | 기존 워커 동작, 모델/Bridge 설정, 현재 세션/운동 상태, 분석 선택 게이트 |
| analyzer | 호출 여부, ID/revision, 표본 시각, 좌우 각도/높이, 신뢰도, phase·전환·안정화·이벤트 |
| reset | 초기화 전후 phase와 마지막 표본 시각 |
| bridge | 호출/건너뜀, 실제 선택 클래스, 이벤트와 Manager 수락/거부, 현재 상태/세대 |

진단 순서는 다음과 같습니다.

1. **Analyzer 호출 없음**: pipeline의 `pose_model`, `mediapipe_selected`, `bridge_configured`,
   `gate_reason`, `session_status`, `exercise`를 확인합니다. `session_not_running`,
   `exercise_missing`, `exercise_not_running`, `other_exercise_selected`는 해당 상태를 뜻합니다.
   `stage=no_camera_frame`은 카메라 read가 프레임을 반환하지 않은 경우입니다.
   Bridge의 `analyzer_called=false`와 reason은 `no_running_exercise`, `no_keypoints`,
   `stale_context_before_analysis` 등을 구분합니다. No-op/다른 분석기 주입 여부는
   `selected_wrapper`/`selected_analyzer`로 확인합니다.
2. **초기 READY 실패**: analyzer의 `ready_left/right`와 각각의 `_reasons`를 봅니다.
   `left_elbow`, `right_elbow`, `left_wrist_height`, `right_wrist_height`는 실제 계산값이며
   `thresholds`는 해당 인스턴스가 실제 사용하는 Config입니다.
3. **PRESSING/TOP 실패**: `pressing_left/right`, `top_left/right` 및 `_reasons`로 부족한 팔과
   각도/높이를 확인합니다. 예를 들어 `right_elbow_below_top_threshold`는 오른쪽 팔꿈치가
   TOP 각도 기준 미만이라는 뜻입니다. 이는 원인 코드 설명이며 실측 수치 예시가 아닙니다.
4. **복귀 실패**: `phase=top`이면 lowering 조건, `phase=lowering`이면 ready 조건을 봅니다.
   `previous_phase`/`next_phase`, candidate, candidate_samples, candidate_duration_sec,
   transition_stable은 조건이 충족돼도 안정화가 아직 끝나지 않은 경우를 구분합니다.
   `completion_rejection=cycle_too_short` 등은 READY 복귀 시 이벤트가 없었던 이유입니다.
5. **입력 무효**: `input_valid=false`, `invalid_reason`, `invalid_detail`과 `joints`의 각 관절
   Visibility/Presence를 확인합니다. 관절 ID 11/12=좌/우 Shoulder, 13/14=좌/우 Elbow,
   15/16=좌/우 Wrist이며 name/observed도 함께 기록합니다. `missing_joints_[...]`,
   `joint_15_visibility_invalid_or_below_threshold`, `joint_16_presence_invalid_or_below_threshold`,
   `arm_angle_difference_exceeded` 등으로 구분합니다. 계산할 수 없는 값은 null이며
   NaN/Infinity는 JSON 문자열로 기록합니다. 값을 추정해 채우지 않습니다.
6. **낮은 FPS/표본 시간 이상**: frame_timestamp, frame_gap_sec, duplicate_timestamp,
   reset_reason=frame_gap_exceeded/cycle_timeout, invalid_reason=timestamp_reversed와
   thresholds의 간격·시간 기준을 비교합니다. cycle_duration_sec는 현재 부분 반복 시간입니다.
7. **이벤트 반영 실패**: analyzer의 `rep_completed=true` 및 event_id와 Bridge 레코드를
   맞춰 봅니다. `manager_called=true, event_accepted=true`가 실제 수락입니다.
   `manager_rejected`는 실제 반영 함수가 False를 반환한 경우로, 기존 상태/ID/세대 또는
   중복 검증에 의해 거부된 것입니다. `stale_context_after_analysis, manager_called=false`는
   결과 생성 이후 상태가 바뀌어 Manager 호출 전에 폐기된 경우입니다.
   Bridge의 manager/current_context는 로그 시점에 읽은 참고 Snapshot이며 원자적 수락 시점의
   상태를 재구성한 값은 아닙니다. event_accepted는 실제 반환값입니다.

pipeline 상태 출력은 별도 스레드를 추가하지 않고 기존 워커가 camera.read 전/프레임 없음
분기를 지날 때 수행합니다. GPU 호출 등에서 워커 자체가 장시간 막히면 이 출력도 멈춥니다.
진단 활성화는 상태/임계값/이벤트 정책을 바꾸지 않습니다. 로그에는 관절 신뢰도·2D 파생값과
세션/운동 ID를 기록하며 영상 저장이나 새 REST 필드를 추가하지 않습니다.
실제 실패 원인은 이 로그를 수집한 뒤 판단해야 하며 아직 특정 조건으로 확정하지 않았습니다.


## PushUpAnalyzer

`exercises/push_up.py`의 `PushUpAnalyzer` / `PushUpConfig`를 사용합니다.
기존 Registry의 `get_analyzer('push_up')`가 새 인스턴스를 생성하며 공통 구조를 바꾸지 않습니다.
측면 촬영용 Rule-based 반복 측정으로, Score/Posture Error/피드백은 계산하지 않습니다.

### 필수 관절, 선택 팔 및 방향 검사

- 선택 팔의 필수 관절은 Shoulder(11/12), Elbow(13/14), Wrist(15/16)입니다.
  Hip(23/24)은 팔 품질 계산과 분리된 몸통 보조 관절이며, 양쪽 중 하나는 유효해야 합니다.
  선택 팔 쪽 Hip을 우선하고 그 Hip이 무효면 유효한 반대쪽 Hip을 사용합니다.
- Ankle(27/28)은 선택적입니다. 누락·낮은 신뢰도·화면 밖이면 추정 없이 생략합니다.
  관측할 수 있으면 Hip→Ankle 방향은 진단 정보로 확인하며, 초과해도 팔을 무효화하지 않습니다. 발목이 잘려도 팔·골반이 유효하면 분석할 수 있습니다.
- HIGH 시작 자세의 선택에서는 세 팔 관절의 Visibility/제공된 Presence 최솟값이 더 높은 HIGH 후보를 고릅니다.
  동률이면 왼쪽입니다. HIGH 후보 안정화 중부터 품질 순위가 바뀌어도 팔을 변경하지 않습니다.
- 선택 팔/몸통이 무효인 프레임은 invalid이며 상태·카운트를 진행시키지 않습니다.
  마지막 유효 관측부터 기본 0.3초까지 기다리고 이를 초과하면 부분 반복을 폐기합니다.
  이후 안정적인 HIGH에서만 팔을 재선택합니다.
  반대 팔로 미완료 반복을 이어 세지 않습니다. 두 팔 모두 무효면 분석할 수 없습니다.
- 정규화 x/y에 실제 영상 width/height를 곱해 팔꿈치 내각, 몸통 길이와 방향을 계산합니다.
  사용하는 관절은 [0,1] 좌표, Visibility≥0.5, Presence가 제공되면 ≥0.4이어야 합니다.
- Shoulder→Hip은 수평에서 최대 55°, Hip→Ankle 65°는 진단 기준으로만 사용합니다.
  이는 서서 팔만 굽히는 동작을 배제하기 위한 넓은 방향 조건이며 허리 처짐 점수가 아닙니다.

### 상태 머신 및 반복 완료

`unknown → high → descending → bottom → ascending → high(RepCompleted)`

먼저 HIGH를 확인하고 어깨 y와 몸통 길이를 기준으로 저장합니다. 각도 하강 이후 BOTTOM에는
기준보다 어깨가 몸통 길이의 0.04 이상 화면 아래로 이동해야 진입합니다. ASCENDING 이후에는
팔이 펴지고 어깨가 HIGH 기준의 ±0.18 몸통 길이 안으로 복귀해야 완료합니다.
부분 하강 후 HIGH 복귀, BOTTOM만 관측, HIGH 유지, 어깨 이동 없는 팔꿈치 굽힘은 세지 않습니다.
상승 중 다시 BOTTOM으로 내려가도 최종 HIGH 복귀 전에는 새 반복을 생성하지 않습니다.

| 설정 | 초기 실험값 |
|---|---:|
| HIGH 진입 / 복귀 | ≥145° |
| DESCENDING 진입 | ≤140° |
| BOTTOM 진입 | ≤135° |
| ASCENDING 진입 | ≥140° |
| Visibility / 제공된 Presence | ≥0.5 / ≥0.4 |
| 각 전환 안정화 | 0.15초 + 서로 다른 시각의 2표본 이상 |
| 반복 시간(하강 후보 시작~HIGH 복귀 확인) | 0.8~10초 |
| 최대 표본 간격 | 0.75초 |
| 몸통 / 보이는 다리의 수평 대비 최대 기울기 | 55° / 65° |
| 최소 어깨 하강 / HIGH 복귀 허용 거리 | 몸통 길이의 0.04 / ±0.18 |

이전 작업에서 BOTTOM을 110°→125°→135°로 조정했습니다. 이번 작업 시작 시 실제 소스는
HIGH 145° / DESCENDING 140° / BOTTOM 135° / ASCENDING 140°였습니다.
이번 한쪽 팔·누락 유예 개선에서는 이 각도와 기존 어깨 이동·방향·시간 설정을 변경하지 않았습니다.
합성 좌표에서는 130°/90° 하강 후 복귀는 1회, 145°까지만 하강한 동작은 0회로 검증합니다.
변경 후 실제 카메라 정확도는 별도로 재검증해야 합니다.

모두 `PushUpConfig`에서 조정합니다. HIGH↔DESCENDING 및 BOTTOM↔ASCENDING 각도에
hysteresis를 둡니다. 같은 timestamp는 표본 수·안정화 시간·이벤트를 진행시키지 않습니다.
시간 역행, 긴 프레임 공백, 최대 반복 시간 초과는 부분 반복을 폐기합니다.
무효 관절에는 아래 누락 유예를 적용하며 누락 중 안정화는 초기화합니다.
같은 영상에 새 timestamp가 붙는 경우는 입력만으로 완전히 구분할 수 없습니다.

### 세션 연동과 검증 한계

분석기는 UUID+완료 번호의 `RepCompleted.event_id`만 반환하며 rep_count를 직접 수정하지 않습니다.
reset은 이벤트 번호를 재사용하지 않습니다. 기존 Bridge/SessionManager가 ID·revision·event_id를
검증해 반영합니다. Pause/Resume 및 운동/세션 변경 시 HIGH부터 재동기화하며 분석 중 변경된
세대의 결과도 폐기합니다. `/internal/sessions/{session_id}/live` 스키마는 유지합니다.

`tests/test_session_push_up.py`는 카메라/GPU 없이 합성 좌표로 상태 전환, 1회/5회,
가림·선택 팔 고정·관절 손실·방향·어깨 이동·시간 조건·중복 이벤트·Pause/Resume·세션 전환·REST를 검증합니다.
전체 회귀 실행 명령: `python -m unittest discover -s tests -v`.
기존 `test_logger_during_idle_and_stop` 실패는 별도 기존 실패이며 이번 작업에서 수정하지 않습니다.

**푸시업 실제 ESP32-CAM 정확도 검증은 미실시입니다.** 고정 카메라에서 측면 팔·골반이 보이는
구도를 사용하고, 실제 HIGH/BOTTOM 각도·어깨 이동·관절 신뢰도·표본 간격을 수동 횟수와 비교해야 합니다.
기울기/원근 왜곡·몸의 영상 내 크기·가림·낮은 FPS에 따라 누락될 수 있습니다. 몸통 길이 정규화와
넓은 기울기 허용은 영향을 줄이는 초기 방식이며 카메라 이동 보정이나 3D 자세 검증은 아닙니다.
촬영 중 카메라가 움직이거나 인물이 평행 이동하면 어깨 이동 검사에 영향을 줄 수 있습니다.


### 푸시업 BOTTOM 미진입 진단

`PUSHUP_DEBUG=1`을 기존 서버 실행 명령 앞에 지정하고 서버를 재시작하면
`[PushUpDebug]` JSON 로그를 출력합니다. 기본은 꺼짐이며 일반 로그는 초당 1회 이내, 상태 전환·입력 무효 사유 변경·
반복 완료 시 즉시 출력합니다. 유효 입력의 대기 사유만 바뀐 경우에는 주기 제한을 유지합니다. 이 로그는 분석기가 호출된 프레임만 다룹니다.

- `elbow_angle_deg`, `bottom_angle_met`: 선택 팔의 실측 각도와 BOTTOM 각도 충족 여부.
- `shoulder_drop_px`, `baseline_torso_length_px`, `shoulder_drop_ratio`, `bottom_drop_met`:
  `(현재 어깨 y - HIGH 어깨 y) * 영상 높이 / HIGH의 픽셀 Shoulder–Hip 길이`입니다.
  영상 아래쪽이 양수이며, 현재 프레임의 몸통 길이로 다시 정규화하지 않습니다.
  `shoulder_drop_below_minimum`이면 각도를 만족해도 기본 0.04 하강 조건에 막힌 것입니다.
- `left`/`right`의 `torso_inclination_deg`, `torso_valid`, `leg_valid`, `invalid_reason`:
  선택 시 검사한 팔 또는 추적 중인 팔의 방향/관절 거부 사유입니다.
  `torso_direction`은 몸통 55° 조건입니다. `leg_valid=false`는 다리 65° 초과를 알리는 비차단 정보입니다.
- `bottom_phase_met`, `candidate`, `candidate_samples`, `candidate_duration_sec`, `transition_stable`:
  BOTTOM 이전에 DESCENDING을 확인해야 하며, 각 상태에서 0.15초와 2표본을 모두 만족해야 합니다.
  낮은 FPS에서 중간 자세를 짧게 통과하면 각도가 충분해도 안정화되지 않을 수 있습니다.
- `frame_gap_sec`, `reset_reason`은 0.75초 표본 공백/10초 반복 초과를 구분합니다.
  `shoulder_not_returned`는 HIGH 복귀 시 ±0.18 몸통 길이 조건에 막힌 경우입니다.
- `session_id`, `exercise_id`, phase, `rep_completed`, `event_id`, `thresholds`를 함께 기록합니다.
  이벤트의 SessionManager 수락 여부는 기존 Bridge가 검증하며 이 로그에는 포함하지 않습니다.

보조 조건이 실제 환경에서 과도한 제한인지 판단할 실측값은 아직 없습니다. 따라서 해당
기본값은 그대로 유지했습니다. 하강량·복귀량·몸통/다리 기울기·안정화 시간은 이미 Config로
조정 가능합니다. 진단에서 특정 제한을 확인한 뒤 조정해야 하며, 어깨가 움직이지 않거나
매우 작게 움직이는 동작을 무조건 반복으로 허용하지 않습니다.


### 불완전한 스켈레톤에서 카운트 실패 확인

환경변수는 `PUSHUP_DEBUG=1`입니다. 이전 `PUSH_UP_DEBUG=1`도 호환하며 두 변수가
동시에 있으면 `PUSHUP_DEBUG`를 우선합니다. 객체 생성 시 읽으므로 설정 후 재시작하세요.
진단은 선택된 분석기가 호출될 때만 출력되며 임계값과 판정 정책은 변경하지 않습니다.

양쪽 발목은 **필수가 아닙니다**. 몸통 확인에 사용하는 Hip과 같은 쪽 발목도 관절이 없거나 신뢰도/좌표가
무효면 보조 검사를 생략합니다. 반면 신뢰도·좌표 검사를 통과한 발목이 잘못된 위치에
검출돼도 Hip→Ankle 기울기는 보조 로그에만 남기며 이 이유로 반복을 폐기하지 않습니다.
몸통에 사용하지 않는 쪽 발목은 판정에 영향을 주지 않습니다. Hip은 양쪽 중 하나가 필요하며
Hip 오류는 몸통 방향과 어깨 하강 정규화 길이에 영향을 줄 수 있습니다.

로그의 `joints`에는 양쪽 Shoulder/Elbow/Wrist/Hip/Ankle의 실제 x/y, Visibility, Presence,
유효 여부와 사유를 기록합니다. `missing`, `duplicate`, `coordinates_invalid`,
`visibility_below_threshold_or_invalid`, `presence_below_threshold_or_invalid`를 구분합니다.
정확한 위치와 다르게 검출됐어도 높은 신뢰도가 부여될 수 있어 신뢰도만으로 정확도를 보장하지 않습니다.

`selected_side`, `elbow_angle_deg`, `previous_phase`, `phase`, `next_phase`, `input_valid`,
`invalid_reason`, `transition_failures`, `rep_completed`를 확인하세요. 선택 팔의 Hip/다리만
무효인 경우에도 팔 관절이 유효하면 각도를 진단에 남깁니다. 각도를 계산할 수 없으면 null입니다.
`conditions`는 HIGH/DESCENDING/BOTTOM/ASCENDING의 조건 충족(`met`)과 현재 단계에서
전환 가능한지(`phase_allowed`)를 구분합니다. 입력 무효·중복 시각 등 평가하지 않은 조건은 null입니다.

| 단계 | 각도 외 카운트를 막을 수 있는 조건 |
|---|---|
| 모든 단계 | 선택 팔 Shoulder/Elbow/Wrist 및 몸통에 사용하는 Hip의 좌표·신뢰도, 몸통 방향, 발목/다리 방향은 비차단 보조 정보 |
| HIGH 최초 인식 | 각도≥145°, 0.15초·서로 다른 시각 2표본 |
| DESCENDING | HIGH를 먼저 관측, 각도≤140°, 동일 안정화 조건 |
| BOTTOM | DESCENDING을 먼저 관측, 각도≤135°, HIGH 몸통 길이 대비 화면 아래 방향 어깨 하강≥0.04, 안정화 |
| ASCENDING | BOTTOM을 먼저 관측, 각도≥140°, 안정화 |
| HIGH 복귀/완료 | 각도≥145°, 어깨가 HIGH 기준 ±0.18 이내 복귀, 안정화, 반복 시간 0.8~10초 |

추적 팔은 관절 품질 순위가 달라져도 교체하지 않습니다. 선택 팔이 무효가 되면 상태 진행을 멈추고,
유예 초과 시 미완료 동작을 버린 뒤 HIGH부터 재시작합니다. 0.75초 초과 표본 공백도 재동기화합니다.
실제 ESP32-CAM 로그가 없어 현재 실패 원인은 아직 확정할 수 없습니다. 위 로그로 관절/방향
무효와 각도·하강량·중간 상태·안정화 실패를 먼저 구분해야 합니다.


### HIGH 팔 선택과 짧은 가림 유예

이전 구현도 양팔 동시 충족을 요구하지는 않았지만 Hip을 팔 품질에 포함했고, 선택 팔/몸통이
한 프레임만 무효여도 즉시 폐기했습니다. 이제 HIGH 후보 중 유효한 세 팔 관절 품질이 가장
높은 팔을 선택합니다. HIGH 후보를 한 번 선택하면 안정화 도중에도 고정하며, 안정화된 이후에도
신뢰도 순위가 역전되어도 팔을 바꾸지 않습니다. 팔꿈치 각도를 계산할 수 없는 팔은 제외합니다.
Presence가 제공되지 않으면 Visibility로 평가하고, 제공되면 양쪽 중 낮은 값을 사용합니다.
최종 팔 품질은 Shoulder/Elbow/Wrist의 이 값 중 최솟값입니다.

`occlusion_grace_sec=0.3`(양수, max_frame_gap_sec 이하) 동안 선택 팔/몸통이 일시 무효이면
phase와 선택 팔을 보존하되 안정화 후보·표본을 초기화합니다. 무효 프레임에서는 상태 전환이나
이벤트가 없습니다. 복구 후에도 같은 팔과 새 안정화 표본을 요구하므로 누락 중 BOTTOM을
지났다고 추정하지 않습니다. 유예는 최초 무효 프레임이 아니라 **마지막 유효 관측 시각**부터
계산합니다. 유예 만료 후에는 반대 팔로 이어 세지 않고 HIGH부터 재선택합니다.
몸통 길이 기준은 HIGH에서 저장하고, 어깨 이동·몸통 방향 조건은 유지하며 보이는 다리 방향은 비차단 진단으로만 사용합니다.
이 정책은 고신뢰도지만 잘못 추정된 관절 위치를 자동 교정하는 기능은 아닙니다.

기존 Bridge는 사람 전체의 Keypoints가 빈 목록이면 분석기를 즉시 reset합니다. 이번에는
공통 파이프라인을 변경하지 않았으므로 전체 포즈 미검출은 유예 대상이 아니고 안전하게 폐기됩니다.
Pause/Resume 및 운동/세션 변경도 기존 generation/reset 정책을 그대로 사용합니다.

`PUSHUP_DEBUG=1` 로그에는 `selected_arm`, `left_arm_quality`, `right_arm_quality`,
`selected_elbow_angle`, `selection_reason`, phase/next_phase, 관절 유효성, 전환 실패 조건,
`rep_completed`를 기록합니다. 품질을 계산할 수 없는 팔은 null입니다.
`highest_quality_high_arm`은 HIGH 후보 선택, `selected_arm_locked`는 추적 고정,
`occlusion_grace`는 누락 대기, `occlusion_timeout`은 미완료 폐기,
`selected_arm_recovered`는 같은 팔 복구입니다. 선택/재선택과 상태 전환 시 즉시 출력하며
일반 로그는 1초에 한 번 이내입니다.

현재 관찰에 맞춰 확인된 코드상의 취약점은 즉시 초기화 정책과 Hip이 섞인 팔 품질입니다.
실제 카운트 실패의 단일 원인은 촬영 로그 없이 확정할 수 없습니다. ESP32-CAM에서 선택 팔의
일관성, 좌우 품질·가림 지속 시간, 몸통 관측, 모든 단계의 안정화 여부와 실제 수동 횟수를
비교하고, 고신뢰도 오검출과 낮은 FPS로 인한 중간 상태 누락도 확인해야 합니다.


### 런타임 초기화·상태·이벤트 반영 진단

`PUSHUP_DEBUG=1`을 기존 GPU 서버 실행 명령 앞에 지정하고 프로세스를 재시작합니다.
예: `PUSHUP_DEBUG=1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
기존 PoseStreamService나 REST 응답은 변경하지 않습니다. 로그는 `[PushUpDebug]` JSON입니다.

- `kind=initialization`: 실제 Analyzer가 생성됐다는 표시입니다. `source_file`로 로드 경로,
  `thresholds`로 해당 인스턴스의 모든 설정을 확인합니다. Registry는 첫 분석 호출에서 생성하므로
  서버 시작 직후 아직 이 로그가 없을 수 있습니다. 현재 소스의 Config/Analyzer 생성은 단위 테스트로
  확인했으며 `occlusion_grace_sec` 등 코드에서 참조하는 Config 필드의 존재도 검사합니다.
- `kind=analyzer`: `analyzer_called=true`, 선택 팔/좌우 품질/각도, shoulder_drop_ratio,
  shoulder_return_ratio(하강 비율의 절댓값), 몸통/다리 기울기, frame_gap_sec, 이전/다음 phase,
  candidate_phase/candidate_duration_sec/candidate_samples, input_valid/invalid_reason,
  transition_failures, rep_completed를 출력합니다. 관측하지 못한 값은 null이며 추정하지 않습니다.
  `leg_inclination_deg=null`은 발목을 보지 못해 선택적 검사를 생략한 경우일 수 있습니다.
- `kind=bridge`: `analyzer_called=false`와 `no_running_exercise`, `no_keypoints`,
  `stale_context_before_analysis` 등으로 호출 이전 차단을 구분합니다. 생성·분석 예외는
  `analysis_exception`과 detail로 확인합니다. 다른 운동만 실행 중일 때는 푸시업 로그를 출력하지 않습니다.
- `rep_completed=true`인 Analyzer 레코드의 event_id를 같은 ID의 Bridge 레코드와 연결합니다.
  `manager_accepted`/`event_accepted=true`는 반영 성공, `manager_rejected`는 중복/유효성 거부,
  `stale_context_after_analysis`는 생성 이후 세대 변경으로 폐기된 경우입니다.
  SessionManager나 REST 스키마를 바꾸지 않고 기존 반영 지점만 관측합니다.

진단 작업 시작 시 소스에서 확인한 값: HIGH 145°, DESCENDING 140°, BOTTOM 135°,
ASCENDING 140°, Visibility 0.5, 제공된 Presence 0.4, 하강 비율 0.04, 복귀 비율 0.18,
몸통 55°/다리 65°, occlusion_grace_sec 0.3초. **이번 진단 작업에서는 값을 변경하지 않았습니다.**
다른 서버 프로세스가 이전 모듈을 사용 중인지는 실제 initialization 로그로 확인해야 합니다.

관절 무효는 joints의 구체적 사유, 몸통 방향 무효는 torso_direction, 다리 기울기 초과는 leg_valid=false(비차단),
각 단계 조건 실패는 high_angle_not_met/descending_angle_not_met/bottom_angle_not_met/
ascending_angle_not_met로 구분합니다. 어깨 이동은 shoulder_drop_below_minimum 또는
shoulder_not_returned, 안정화는 insufficient_distinct_samples/insufficient_stability_duration,
표본 공백은 frame_gap_exceeded, 너무 빠른 완료는 completion_rejection=cycle_too_short로 확인합니다.

한 프레임에 여러 각도 임계값을 통과해도 상태 머신의 if/elif는 현재 phase에서 한 단계만
후보로 정합니다. 각 중간 단계는 별도 안정화 기간을 요구합니다. HIGH 이후 BOTTOM 각도를
0.2초 정도 관측하면 DESCENDING까지만 확정되고, 곧바로 HIGH로 돌아오면 BOTTOM을
확인하지 못해 0회가 될 수 있음을 합성 테스트로 재현했습니다. 같은 낮은 자세도 추가로
안정화 기간을 관측하면 BOTTOM으로 진행합니다. 급격한 상승도 ASCENDING과 HIGH를
각각 확인해야 합니다. 진단 기능은 이 정책을 변경하거나 누락된 중간 상태를 추정하지 않습니다.

현재 소스에서는 한쪽 팔만 유효해도 분석하며, HIGH에서 선택한 팔은 반복 중 유지합니다.
반대 팔 누락은 선택 팔의 품질·단계를 무효화하지 않습니다. 한쪽/양쪽 발목 누락도 필수 조건이
아니며 유효 관절로 간주된 잘못된 발목 추정도 팔 전체를 무효화하지 않습니다.
실제 ESP32-CAM 실패의 원인은 아직 실측 로그가 없어 확정하지 않았습니다.


### 실제 로그에서 확인된 HIGH 실패 수정

사용자 로그에서 오른팔 170.65°·품질 0.72·몸통 34.71°는 HIGH 후보가 될 수 있으나,
다리 79.86°가 65°를 초과해 `leg_direction`으로 제외됐습니다. 또 UNKNOWN의 HIGH 후보를
매 프레임 다시 골라 right→left→right로 바꾸면서 안정화 표본이 계속 1로 초기화됐습니다.

이번 수정은 두 정책만 바꿉니다.

- 다리 기울기는 측정·기준 비교를 유지하되 비차단 정보로 사용합니다.
  `leg_valid=false`, `leg_check_blocking=false`여도 팔·몸통이 유효하면 분석합니다.
  발목 누락/화면 이탈/잘못된 다리 방향만으로 선택 팔을 무효화하지 않습니다.
- HIGH 후보 선택 시점부터 팔을 고정합니다. 품질 순위가 역전되어도 후보 표본을 유지해
  0.15초·서로 다른 시각 2표본을 만족하면 HIGH가 확정됩니다. 선택 팔이 짧게 무효이면
  기존 유예 정책으로 기다리고, 유예 초과 후 HIGH부터 재선택합니다.

Shoulder/Elbow/Wrist의 좌표·신뢰도·각도 계산 검증은 유지합니다. Hip은 몸통 방향과
하강 정규화용 보조 관측으로 양쪽 중 하나를 사용하며, 두 Hip 모두 관측 불가능하면
몸통 검사를 생략해 추정 카운트하지 않고 `no_valid_hip`으로 보류합니다.
몸통 방향, 어깨 하강·복귀, 각도/시간 임계값, 이벤트 중복 방지는 변경하지 않았습니다.

사용자 로그의 각도·품질·기울기를 재현한 합성 좌표로 HIGH 및 5회 완료를 검증합니다.
이는 원본 카메라 좌표나 수정 후 실영상 정확도 검증을 대신하지 않습니다.


## YTWRaiseAnalyzer

정면에서 서서 양팔로 수행하는 **Y → T → W 전체가 1 Rep**입니다. 각 자세를 각각 한 번으로
세지 않습니다. `exercises/ytw_raise.py`의 `YTWRaiseAnalyzer`가 구현이며 기존 이름
`YtwRaiseAnalyzer`는 호환 별칭입니다. 기존 Registry 연결을 그대로 사용합니다.
Score/Grade/Error Codes/Feedback은 별도 학습 모델의 책임이며 이번에는 구현하지 않습니다.

### 좌표와 자세 분류

필수 관절은 양쪽 Shoulder(11/12), Elbow(13/14), Wrist(15/16)입니다. Visibility는 필수,
Presence는 제공될 때 검사하며, [0,1] 밖 좌표·NaN·중복/누락 관절·퇴화한 벡터를 거부합니다.
Hip(23/24)은 선택적입니다. 두 Hip이 유효하면 어깨 중심→골반 중심으로 몸통 위쪽 축을
계산합니다. 사용할 수 없으면 어깨 선에 수직인 영상 위쪽 축으로 대체합니다.
영상 너비/높이를 곱한 픽셀 공간에서 각도를 구하고, 위치는 어깨 폭으로 정규화합니다.
좌우 팔의 바깥쪽 부호를 반대로 적용하므로 좌우 미러링도 같은 조건을 사용합니다.

각 팔의 자기 어깨 기준에서 위쪽과 바깥쪽이 양수입니다. 양팔이 **동시에 같은 자세**여야
해당 자세 후보입니다. 한쪽이 먼저 움직이는 짧은 시차는 전환을 보류하고 안정화 후보만 초기화합니다.

| 설정 (`YTWRaiseConfig`) | 초기 실험값 |
|---|---:|
| Y/T 팔꿈치 신전 | ≥145° |
| Y 손목 방향(바깥 수평 대비 위쪽) | 30~75° |
| T 손목 방향 | ±25° |
| Y/T 손목의 최소 바깥 위치 | 어깨 폭의 0.6 |
| W 팔꿈치 각도 | 60~130° |
| W 팔꿈치 바깥 위치 | 어깨 폭의 0.12 이상 |
| W 팔꿈치 높이 | 어깨 기준 -0.85~0.25 어깨 폭 |
| W 손목-팔꿈치 높이 차 | 0.15 어깨 폭 이상 |
| W 손목 높이 | 어깨 기준 -0.1~1.4 어깨 폭 |
| Visibility / 제공된 Presence | 0.6 / 0.5 |
| 각 전환 안정화 | 0.15초 + 서로 다른 관측 2표본 이상 |
| 관절 누락 유예 | 마지막 유효 양팔 관측부터 0.6초 |
| 최대 유효 관측 간격 / 전체 사이클 | 0.75초 / 12초 |

W는 신전하지 않은 팔꿈치만으로 판정하지 않습니다. 팔꿈치가 바깥쪽에 있고 손목이
팔꿈치보다 위에 있어야 합니다. 팔꿈치가 어깨보다 조금 낮은 W를 허용하며, 팔을 내리거나
손목을 아래로 둔 채 팔꿈치만 굽힌 동작은 W로 인정하지 않습니다. 모든 수치는 Config로
관리하는 초기값이며 실제 촬영·사용자별 측정으로 조정해야 합니다.

### 순서·누락·이벤트

`wait_y → y_confirmed → t_confirmed → W 안정화 + RepCompleted → wait_next_y`

새 Y를 확인하면 재무장합니다. W 유지, Y만, Y→T만, Y→W, T→W, 역순은 이벤트가 없습니다.
Y 뒤 T 없이 안정적인 W가 나오면 진행 상태를 폐기합니다. T 이후 다시 Y가 나오면 Y부터
다시 시작합니다. 분류되지 않는 중간 움직임은 표본을 보류하지만 12초 넘게 오래된 진행은 폐기합니다.

한쪽 필수 관절이 짧게 무효이면 phase는 보존하고 후보 표본을 초기화하며 전진하지 않습니다.
복구 후에도 새 안정화 표본을 요구합니다. 유예 만료(0.6초 이상)·큰 표본 공백·시각 역행은 부분 사이클을
폐기합니다. 전체 포즈가 없으면 기존 Bridge의 reset 정책을 따릅니다.

같은 timestamp와 동일한 관절 좌표·신뢰도 재전달은 새 안정화 표본으로 세지 않습니다.
완전히 정지한 실제 새 프레임과 같은 좌표의 재전달을 구분할 원본 프레임 ID가 없으므로
보수적으로 동일 관측으로 취급합니다. 따라서 완전히 동일한 추정 좌표만 지속되면 안정화가
지연될 수 있습니다. 합성의 정상 시퀀스는 작은 평행 이동으로 서로 다른 관측을 재현합니다.

이벤트 ID는 인스턴스 UUID+완료 번호로 만들고 reset에서도 번호를 되돌리지 않습니다.
분석기는 rep_count를 직접 수정하지 않습니다. 기존 Bridge/SessionManager가 session_id,
exercise_id, revision, event_id를 검증해 반영하며 Pause/Resume·운동/세션 변경 시 초기화합니다.
`/internal/sessions/{session_id}/live`의 기존 rep_count 필드로 조회하며 REST 스키마는 유지합니다.

### 진단과 실제 카메라 검증

`YTW_DEBUG=1`을 기존 실행 명령 앞에 지정하고 재시작합니다. `[YTWDebug]` 로그에는 phase,
감지/안정화 후보, 양팔 각도·방향·손목/팔꿈치 위치, 필수 관절 신뢰도, 양팔별 Y/T/W 실패 조건,
candidate_samples/duration, 전환 실패 사유와 이벤트를 기록합니다. 일반 로그는 약 1초 간격,
상태 전환·무효/순서 오류 변경과 이벤트는 즉시 기록하며 디버그는 기본적으로 꺼져 있습니다.

`tests/test_session_ytw_raise.py`는 합성 좌표로 순서, 1회/5회, 미러링/기울기/영상 비율,
관절 누락·중복 입력·불규칙 간격·세션 제어·중복 이벤트·REST live 반영을 검증합니다.
**YTW의 실제 ESP32-CAM 검증은 미실시입니다.** 정면에서 양쪽 어깨·팔꿈치·손목이 Y까지
화면 안에 들어오는 구도를 사용하고, 가능하면 골반도 보이게 촬영하세요. 실제 FPS에서 각
자세를 자연스럽게 통과할 때 후보 안정화가 가능한지, 좌우 시차·가림·촬영 거리/기울기에 따른
오분류와 수작업 횟수 차이를 확인해야 합니다. 합성 통과는 실영상 정확도 검증을 대신하지 않습니다.
기존 푸시업의 마지막 설정/수정에 대한 최종 정확도가 검증됐다는 뜻도 아닙니다.


### 일시 가림과 지속 무효 구분

실제 사용자 로그에서 왼쪽 Elbow/Wrist의 Visibility가 0.30~0.45로 계속 낮아
`joint_13_visibility_invalid_or_low` 및 `occlusion_timeout`이 발생했습니다.
양팔 자세를 계산하기 전에 필수 관절이 거부되므로 이때 Y/T/W recognized=null은
자세 판정이 실패했다는 뜻이 아니라 **평가할 유효한 양팔 입력이 없었다**는 뜻입니다.

필수 6개 Shoulder/Elbow/Wrist를 모두 검증하고 Hip 2개는 선택적 기준축에만 사용합니다.
Visibility 0.6 / 제공된 Presence 0.5 및 Y/T/W 각도·위치 임계값은 유지했습니다.
이번에는 `occlusion_grace_sec`만 0.35→0.6초로 늘렸습니다. 기존 검증식
`0 < occlusion_grace_sec <= max_frame_gap_sec < max_cycle_duration_sec`를 유지하며
0.6초는 max_frame_gap_sec=0.75초와 양립합니다. 전체 포즈가 빈 목록이면 기존 Bridge가
즉시 reset하므로 이 유예는 일부 관절 가림에 적용됩니다.

각 관절의 마지막 유효 x/y는 외부 입력과 분리된 사본으로 저장하고, 유효 시각부터
0.6초가 지나면 좌표를 폐기합니다. 마지막 유효 시각 자체는 지속 무효 진단에 남깁니다.
캐시는 좌표 대입·보간·자세 분류에 사용하지 않습니다. 짧은 가림 동안 phase만 보존하고
안정화 표본은 초기화하며, **양팔이 새 프레임에서 모두 유효해진 뒤** 새 표본으로 재개합니다.
유예는 마지막 유효한 양팔 관측부터 계산하고 복구 시점에 이미 0.6초 이상이면 폐기합니다.

좌우 팔의 서로 다른 시각 좌표를 합치면 실제로 동시에 확인하지 않은 Y/T/W를 만들 수 있어
이번에는 조합하지 않았습니다. 한쪽이 번갈아 보이거나 이전 T 좌표만 남은 상태에서는
새 W를 확정하지 않습니다. 사용자가 단계 사이를 짧게 어긋나게 움직여도 새 양팔 관측이
확보되는 동안 기존 시퀀스를 보존하는 방식입니다.

`YTW_DEBUG=1`의 joints 항목에 다음 값을 추가합니다.

- `last_valid_timestamp`, `last_valid_age_sec`: 해당 관절의 마지막 유효 관측 시각과 경과 시간.
- `last_valid_coordinates`: 유예 안에만 보관하는 마지막 유효 좌표. 만료 시 null.
- `continuous_invalid_duration_sec`: 첫 연속 무효 표본부터 경과 시간. 유예 계산의 기준인
  마지막 유효 시각과 구분합니다. 회복 시 0으로 돌아갑니다.
- `recovered`: 직전까지 무효였던 관절이 해당 프레임에서 회복됐는지 여부.
- 최상위 `cycle_discard_reason`: 현재 가림 구간에서 사이클을 버린 원인.
  `reset_reason`은 실제 초기화가 발생한 순간, `cycle_discard_reason`은 이후에도 지속되는 원인입니다.

기존에는 장기 무효의 매 프레임마다 재초기화하면서 강제 로그를 출력했습니다. 이제 같은
가림 구간에서는 한 번만 폐기합니다. 동일 실패는 1초에 한 번 이하, 관절 무효 사유 변경·상태
전환·이벤트 생성은 즉시 출력합니다. 0.2/0.4초 가림 복구, 0.6초 이상 만료, 지속 무효,
과거 좌표를 통한 허위 W 방지, 새 YTW 5회, W 중복 방지와 로그 주기를 합성 테스트합니다.

이 변경은 지속적으로 낮은 Visibility를 유효 관절로 바꾸는 기능이 아닙니다. 왼팔이 계속
무효이면 0.6초 뒤 폐기하며, 카메라 구도·조명·몸과 팔의 겹침을 조정해 실제 양팔 관측을
확보해야 합니다. 수정 후 실제 ESP32-CAM 재검증은 별도로 필요합니다.


### 영상·분석 입력의 프레임/사람 출처 확인

`YTW_DEBUG=1`로 재시작하면 `/stream`에 처리 결과의 frame_id, 분석용 monotonic timestamp,
선택 pose 인덱스, 어깨 중심의 정규화/픽셀 좌표, 관절 11~16의 ID·Visibility·YTWvalid를
추가 표시합니다. 낮은 신뢰도 관절도 진단 라벨은 표시하되 YTW의 판정에는 그대로 무효입니다.
YTWvalid는 현재 YTW Config의 좌표/Visibility/Presence 검사 결과이며, 처음 분석기를 만들기
전에는 YTW 기본값을 사용합니다(`validation_config_source`). Y/T/W 자세 인정 자체와는 다릅니다.
추가 표시와 출처 비교는 DEBUG를 끄면 수행하지 않습니다.

동일한 frame_id로 다음 `[YTWDebug]` 레코드를 비교하세요.

- `kind=frame_result`: 방금 게시한 JPEG의 추론 결과, source timestamp, MediaPipe timestamp_ms,
  configured_num_poses/returned_num_poses, 선택/그리기 pose 인덱스, 어깨 중심 및 이전 결과 대비
  중심 이동량, 원본→추출 비교 결과입니다. 약 1초마다 출력합니다.
- `kind=frame_input`: Bridge가 복사해 전달하는 데이터와 원본/추출 데이터의 ID·순서·x/y/z·
  Visibility/Presence 일치 여부, 메타데이터 일치와 관절 11~16의 비교값입니다. 약 1초마다 출력합니다.
- `kind=analyzer`: 실제 YTW 입력의 frame_id 및 frame_provenance를 기록합니다. 원본과 분석 입력의
  fingerprint/필드 비교가 일치하는지 확인합니다. 기존 상태/이벤트 로그의 즉시 출력 정책은 유지합니다.
- `kind=bridge`: 기존 이벤트 반영 진단에도 같은 frame_id를 기록합니다.

frame_id는 카메라 원본 촬영 ID가 아니라 **성공적으로 게시된 추론/JPEG 결과 번호**입니다.
인코딩 실패 결과는 게시/분석하지 않습니다. 분석 timestamp는 워커가 프레임을 가져온 시각이며
촬영 시각이 아닙니다. MediaPipe에 전달하는 timestamp_ms는 별도로 함께 기록하며, 기존 VIDEO
타임스탬프 생성은 바꾸지 않습니다. REST의 processed_at(UTC) 및 기존 응답 스키마는 유지합니다.
브라우저에 보이는 최신 프레임과 다른 시각의 로그를 비교하지 말고 영상의 frame_id를 기준으로
비교해야 합니다. 촬영/네트워크/브라우저 지연의 크기는 이 메타데이터만으로 측정하지 않습니다.

현재 소스 확인 결과:

- predict 한 번의 result로 스켈레톤을 그린 후, 같은 result.pose_landmarks[0]을 추출합니다.
  추출은 ID를 유지한 float 변환이며 크기 조정/좌우 반전은 없습니다. 비유한 x/y/z만 제외합니다.
  Bridge는 좌표·신뢰도를 수정하지 않고 deepcopy하며 YTW는 영상 크기로 각도만 계산합니다.
- num_poses=1이며 인물 중심·ROI·고유 track ID를 이용한 사용자 고정은 없습니다. 반환 인덱스 0은
  '중앙 운동 사용자'라는 의미가 아닙니다. 여러 사람 중 배경 사람이 반환될 가능성을 코드가
  차단하지는 않지만, 실제로 전환됐는지는 영상과 프레임별 어깨 중심/좌표를 대조해야 합니다.
- 스켈레톤은 반환된 모든 pose를 그립니다. 현재 num_poses=1에서는 추출 대상과 동일하지만,
  향후 여러 pose를 반환하도록 바꾸면 '모두 그리기/첫 pose 분석' 차이가 생길 수 있어 로그에
  drawn_pose_indices를 명시했습니다. 이번에는 num_poses나 사람 선택을 변경하지 않았습니다.
- 기존 시각화는 Visibility 0.35 이상을 허용하고 Presence를 검사하지 않습니다. YTW는
  Visibility 0.6 및 제공된 Presence 0.5를 검사하므로 그림이 보여도 분석에서는 무효일 수 있습니다.
  반대로 동일 프레임의 Visibility가 0.02~0.05라면 해당 관절은 현재 시각화 기준도 통과하지 않습니다.
  이 경우 다른 프레임/사람/실행 버전을 보고 있지는 않은지 출처 로그로 확인해야 합니다.

원본→분석 비교에 누락이 있으면 비유한 좌표 필터인지, ID/필드 변경인지 mismatches를 확인합니다.
정상 유한 입력에서는 합성 테스트로 동일 프레임·값 전달을 검증했으며, 실제 ESP32-CAM에서
사람 전환이나 프레임 혼용이 발생했다는 결론은 아직 내리지 않았습니다. 대상 혼선이 실제로
확인되면 중앙 ROI/사용자 선택 및 추적 유지 정책이 최소 후속 후보입니다. 이번 진단에서는
임계값·상태 머신·반복 계산·모델 옵션 값을 변경하지 않습니다.


## Forearm Plank: 유효 자세 유지 시간

`exercises/plank.py`의 `PlankAnalyzer` / `PlankConfig`가 측면 팔꿈치 플랭크만 지원합니다.
Registry의 기존 plank/hold 등록을 사용하고 다른 네 운동의 분석 로직은 변경하지 않습니다.
Score, 자세 오류 분류, 허리 처짐 점수 및 피드백은 별도 학습 모델의 책임입니다.
여기서 몸통 정렬과 지지 조건은 유효 시간을 측정하기 위한 진입 조건입니다.

### 관절·자세 조건

선택 측면의 Shoulder(11/12), Elbow(13/14), Hip(23/24), Ankle(27/28)이 필수입니다.
한쪽이 가려지면 정상 자세로 관측되는 반대쪽을 사용할 수 있으며, 선택한 측면은 정상 구간과
짧은 가림 중 고정합니다. 필수 네 관절의 Visibility/제공된 Presence 최솟값을 품질로 사용합니다.
양쪽 필수 관절을 동시에 요구하지 않습니다. Wrist(15/16)는 선택적이며 신뢰도가 낮거나
없으면 보조 검사를 생략합니다. 어깨부터 발목까지 선택 측면의 필수 관절은 화면에 있어야 합니다.

정규화 x/y를 영상 width/height로 픽셀 보정한 뒤 계산합니다.

| 조건 | 초기값 |
|---|---:|
| Shoulder-Hip-Ankle 내각 | ≥155° |
| Shoulder-Ankle 선에서 Hip 수직 거리 / 해당 선 길이 | ≤0.12 |
| Shoulder-Ankle 수평 대비 기울기 | ≤35° |
| Visibility / 제공된 Presence | ≥0.6 / ≥0.5 |
| 어깨 아래 팔꿈치 거리 / 몸 길이 | 0.05~0.45 |
| 어깨·팔꿈치 가로 차이 / 몸 길이 | ≤0.15 |
| 관측 가능한 Wrist의 팔꿈치 내각 | 55~125° |
| 관측 가능한 Wrist·Elbow 세로 차이 / 몸 길이 | ≤0.10 |
| 초기 안정화 | 0.3초 + 서로 다른 시각 2표본 이상 |
| 가림/자세 이탈 유예 | 0.5초 |
| 최대 프레임 간격 | 0.75초 |

Hip 수직 편차의 부호로 과도한 상승/처짐을 구분합니다. 몸통 정렬·기울기·팔꿈치가 어깨
아래에 있는지를 함께 확인하며 Wrist가 보이면 굽힌 팔꿈치와 수평에 가까운 전완도 확인합니다.
Wrist가 보이지 않을 때는 하이 플랭크와 팔꿈치 플랭크를 완벽히 구분할 수 없습니다.
해당 보조 검사를 생략했다는 wrist_observed 값을 로그로 확인하고 실촬영 구도를 조정해야 합니다.
위 값은 실영상 검증 전 Config 초기값입니다.

### 상태와 시간 정책

`waiting → stabilizing → holding → completed`

유효 자세가 0.3초·2표본 이상 연속 관측된 후 holding에 진입합니다. 안정화 시간은 유지
시간에 넣지 않습니다. holding의 서로 인접한 새 유효 관측 사이 시간만 누적합니다.
5초 유지 테스트는 안정화 뒤 추가로 5초간 관측한 시간입니다. 목표 미지정(None)이면
무제한으로 측정하고, 목표 지정 시 current_hold_sec가 목표에 도달하면 completed가 됩니다.

자세 붕괴/관절 누락은 interrupted로 전환하고 시간을 멈춥니다. 마지막 유효 관측부터
0.5초 미만에 같은 측면이 회복되면 구간을 이어가되, 누락 및 복구 첫 표본까지의 시간은
전부 제외합니다. 0.5초 이상이면 current_hold_sec를 0으로 만들고 새 안정화를 요구합니다.
짧은 가림 뒤 반대 측면으로 바꾸어 이어 세지 않습니다. 큰 frame gap도 구간을 폐기합니다.
현재 구간만으로 목표를 판정하며, 여러 구간을 합쳐 목표를 달성하지 않습니다.

동일 timestamp, 음수 시간 차이, 제공된 동일 frame_id는 누적하지 않습니다. 서로 다른
실제 정지 관측은 유효합니다. 원본 프레임 ID가 없는 입력에서는 동일 좌표가 새 영상인지
재전달인지 구분할 수 없으므로 좌표 자체를 중복 프레임으로 단정하지 않습니다.
유효 표본 사이에서 실제 자세가 잠깐 무너졌지만 관측되지 않은 경우까지 판별하지는 못합니다.

### SessionManager·REST 계약

AnalysisResult의 hold_duration_sec / target_duration_sec를 재사용하고 선택적인
hold_completion_event: HoldCompleted를 추가했습니다. 이는 RepCompleted가 아니며
플랭크의 rep_count는 항상 분석 경로에서 증가하지 않습니다. Bridge는 기존 세대·ID·running
검증 후 관리자에 원자적으로 Hold 결과를 저장합니다. 늦거나 같은 timestamp 결과를 거부합니다.
목표 달성은 한 번만 수락하고 이후 해당 운동의 분석을 중지합니다. 운동/세션 자체를 자동
종료하지 않으므로 기존 finish API를 명시적으로 호출하세요.

운동 시작 요청은 기존 필드를 그대로 지원하며 플랭크에만 선택 목표를 받습니다.

```json
{"exercise_name":"plank","order_no":1,"target_duration_sec":30}
```

`POST /internal/sessions/{session_id}/exercises/{exercise_id}/start`에 전송합니다.
미지정 시 목표는 null입니다. target_duration_sec는 양의 유한한 초이며 다른 운동에는 400입니다.
기존 X-Internal-API-Key 인증을 그대로 사용합니다.

`GET /internal/sessions/{session_id}/live`의 exercise는 플랭크에만 네 필드를 추가합니다.
기존 응답 필드의 이름/의미는 유지하고 다른 네 운동의 응답 필드는 확장하지 않습니다.

```json
{
  "exercise_id":10,"exercise_name":"plank","order_no":1,
  "target_reps":null,"rep_count":0,"exercise_status":"running",
  "current_hold_sec":5.0,"target_duration_sec":30.0,
  "hold_status":"holding","hold_completed":false
}
```

위 JSON은 응답 형식 예시입니다. 목표 완료 시 hold_status=completed, hold_completed=true입니다.
유지 시간 조회는 마지막 분석값이며 서버 시계로 외삽하지 않으므로 카메라가 멈춰도 시간이
늘지 않습니다. 새 관측이 없을 때 hold_status가 마지막 관측 상태로 남을 수 있습니다.

Pause는 미완료 current_hold_sec를 0으로 초기화합니다. Resume은 새 generation과 새 안정화를
요구하므로 Pause 경과 시간이 더해지지 않습니다. 완료된 목표는 Pause/Resume에도 유지합니다.
finish_exercise는 마지막 유효 시간을 보존하고 측정을 중단합니다. 다음 운동/새 세션은 별도
Hold 상태로 초기화합니다. 기존 인메모리 정책대로 재시작 시 기록은 소실됩니다.

Backend/GUI 담당자는 plank에서 target_duration_sec를 전송하고 current_hold_sec/hold_status/
hold_completed를 표시해야 합니다. rep_count를 초 단위로 해석하지 마세요. 이 네 추가 응답
필드를 허용하도록 엄격한 클라이언트 DTO만 갱신하면 됩니다. Backend DB나 GUI 코드는 수정하지 않았습니다.

### 진단·실제 ESP32-CAM 검증

기존 GPU 실행 명령 앞에 `PLANK_DEBUG=1`을 지정하고 재시작합니다. `[PlankDebug]` 로그에는
selected_side, 필수/보조 관절 신뢰도, body_alignment_deg, hip_deviation_ratio, body_tilt_deg,
shoulder_elbow 계산값, posture_valid/invalid_reason, phase/유지 시간/목표, frame_gap_sec,
occlusion_elapsed_sec, reset_reason과 completed를 출력합니다. 일반 로그는 약 1초마다,
상태 변화·유예 초과·완료는 즉시 출력합니다.

합성 테스트는 sleep 없이 timestamp를 주입해 자세 진입, 5초/30초, 잘못된 자세, 가림·시간
제외·새 구간, 중복 timestamp/frame_id, 세션/목표/완료 이벤트와 REST live를 검증합니다.
전체 테스트: `python -m unittest discover -s tests -v`.
**실제 ESP32-CAM 플랭크 검증은 미실시입니다.** 측면에서 어깨~발목과 가능하면 손목을
화면에 담고, 30초 목표로 시작해 안정화 후 스톱워치와 비교하세요. 0.2초 가림에는 시간이
멈추었다 이어지는지, 0.6초 이탈에는 새 안정화와 0초 시작이 필요한지, Pause/Resume 및
허리 처짐/상승·서기·하이 플랭크가 시간을 늘리지 않는지 실제 신뢰도와 함께 확인해야 합니다.
합성 통과는 촬영 각도·사용자·프레임 간격 전반의 정확도를 보장하지 않습니다.
