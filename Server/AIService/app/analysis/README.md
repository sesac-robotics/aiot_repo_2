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

SquatAnalyzer는 아래 상태 머신으로 실제 Rule-based 반복 이벤트를 생성합니다. 다른 네 운동은
독립 No-op 클래스이며 `not_implemented` phase, 반복 이벤트/유지 시간 None을 반환합니다.
공통 timestamp는 단조 시계의 초 값이며 UTC DB 시각이 아닙니다. 스쿼트는 입력 표본의 시각을
사용하고, 다른 네 운동의 뼈대는 호출 시각만 채웁니다.

기본 연결 흐름:

1. 운동 시작 시 SessionManager가 등록표의 이름을 검증합니다.
2. 추론 워커의 첫 유효 프레임에서 `RegisteredExerciseAnalyzer`가 해당 운동의 새 분석기를 만듭니다.
3. 운동·세션·세대 변경 시 기존 분석기의 reset을 호출하고 폐기합니다.
4. Pause/Resume가 프레임 사이에 모두 일어나도 기존 revision 검증으로 이전 결과를 버립니다.
5. 분석은 SessionManager 잠금 밖에서 수행하고 반복 반영은 기존 원자적 검증을 사용합니다.

Plank의 내부 설정은 `get_analyzer('plank', target_duration_sec=30)`으로 표현할 수 있습니다.
기존 REST/DB 필드는 바꾸지 않았으며 이 설정은 현재 API에서 전달되지 않습니다.
향후 유효 자세 구간의 유지 시간만 PlankAnalyzer에서 누적해야 합니다. 세션의
active_duration_sec를 대신 사용하지 않습니다. holding/invalid 등의 phase는 구현 단계에서
반환할 수 있습니다. Hold 결과는 rep_count로 변환하지 않습니다.

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

`SquatConfig` 기본값(실제 영상에서 조정·검증해야 하는 초기 설정):

| 설정 | 값 |
|---|---:|
| 선 자세 진입 / 이탈 | 165° / 150° |
| 최하단 진입 / 상승 시작 | 95° / 110° |
| 최소 visibility / 제공된 presence | 0.6 / 0.5 |
| 최대 좌우 무릎 각도 차이 | 25° |
| 각 상태 전환 안정화 | 0.15초 + 서로 다른 시각의 2표본 이상 |
| 유효 반복 시간(하강 시작 후보부터 최종 선 자세 확인) | 0.8~10초 |
| 허용 최대 표본 간격 | 0.75초 |

같은 timestamp는 표본 수/시간을 진행시키지 않으며 이벤트를 다시 생성하지 않습니다.
역행 timestamp, 긴 공백, 반복 시간 초과 시 현재 동작을 폐기하고 선 자세부터 재확인합니다.
reset 시 미완료 단계와 시각은 지우되 인스턴스의 완료 이벤트 번호는 재사용하지 않습니다.
이벤트 ID는 인스턴스 UUID + 완료 번호로 생성하며 DB ID가 아닙니다. Bridge는 기존 세대/ID
재검증을 거쳐 SessionManager.record_analysis_rep → record_rep으로만 횟수를 반영합니다.
분석기가 rep_count를 직접 변경하지 않습니다.

합성 테스트는 1회/5회, 부분 동작, 한 프레임 노이즈, 중복 표본, 관절 손실, 재등장,
Pause/Resume/운동 변경, 영상 비율, 시간 이상, ID 중복 및 REST live 조회를 검증합니다.
합성 통과는 실제 영상 정확도를 의미하지 않습니다. 실제 카메라에서는 다음을 확인해야 합니다.

- 전신/양쪽 하체 가시성, 거리·촬영 각도에 따른 2D 투영 오차와 좌우 각도 차이.
- 사용자별 정상 깊이/선 자세에서 임계값과 시간 안정화에 따른 누락·오검출.
- 실제 FPS/네트워크 지연에서 0.75초 공백 및 0.15초 안정화 설정의 적합성.
- 가림/이탈/재등장, Pause/Resume 및 느리거나 빠른 반복의 수작업 정답 대비 횟수.
- 서버 표본 시각은 촬영 시각이 아닙니다. 같은 영상이 새 시각으로 재전달되는 디코더/카메라
  중복은 현재 입력만으로 완전히 구별할 수 없으므로 실제 스트림에서도 별도 확인해야 합니다.
