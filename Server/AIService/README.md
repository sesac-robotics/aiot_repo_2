

<!-- integration-standards-20261025 -->

# AI 서비스 구성요소

## 실행과 경계

이 서비스는 별도 Python 환경에서 `Server/AIService`를 작업 디렉터리로 두고 `python -m uvicorn app.main:app`으로 실행합니다. `app` 패키지는 이 서비스의 공개 실행 경계입니다. 백엔드·GUI 패키지 경로와 섞어 한 프로세스에서 모두 실행하지 않습니다. 모델 파일·카메라 URL·내부 인증 키는 해당 서비스 환경에서 지정합니다. 실행하면 실제 카메라/모델 수명이 시작되므로 문서 확인만을 위해 실행하지 않습니다.

`main.lifespan`이 SessionManager, 모델 팩터리와 PoseStreamService를 구성합니다. 모델 팩터리 주입은 모델 생성·추론·종료를 동일 워커에 귀속시키기 위한 것입니다. 카메라 입력 선택과 네트워크 읽기는 camera.stream, 모델 호출은 inference, 관절 변환과 운동 전략은 analysis, API 명령은 routers, 상태 소유는 session이 맡습니다.

## 운동 완료와 자세 판정

한 프레임은 추론기를 거쳐 관절 좌표·가시성으로 바뀝니다. RegisteredExerciseAnalyzer가 현재 운동의 전략을 선택하고, ExerciseAnalysisBridge가 분석 전후 AnalysisContext를 대조합니다. Context의 session_id/exercise_id/revision이 달라졌으면 그 결과를 폐기합니다. pause→resume 사이에 늦게 도착한 결과도 이전 revision이므로 반영되지 않습니다.

완료된 Rep 이벤트는 SessionManager가 event_id와 현재 문맥을 확인해 한 번만 반영합니다. Hold는 유지 초와 완료 상태를 따로 관리하며 세션 활동 시간이나 프레임 수를 유지 시간으로 바꾸지 않습니다. 자세 판정의 확률·규칙 결과는 별도 투영이고 운동 횟수와 직접 같은 값이 아닙니다. 관절 미관측은 입력 보류이며 실제 오류로 확정하지 않습니다.

`session/catalog.py`는 운동 이름·반복/유지 유형만 정의합니다. 구체 분석기의 생성자는 `analysis/registry.py`에 있습니다. 따라서 세션 입력 검증이 모델·분석기 구현을 가져오지 않습니다. session/models와 projections도 표준 라이브러리만 사용합니다. SessionManager는 인메모리 상태이므로 프로세스 재시작·다중 worker의 영속성은 제공하지 않습니다.

## 학습 모델과 채점

`analysis/posture`는 기존 시계열 오류 분류기와 명시 기하 규칙을 연결합니다. `scoring`은 검토된 구간의 항목별 특징·평가 자격·학습 모델·규칙 부분 점수를 조합하는 별도 패키지입니다. 이 두 출력을 하나의 동일한 모델로 설명하지 않습니다. ScoringEngine은 신뢰하는 번들 메타데이터와 SHA-256을 확인하고 종목별 모델을 지연 로드합니다. 임의 업로드 pickle을 읽는 API가 아닙니다.

CNN은 시간축의 국소 패턴, GRU는 순서가 있는 관절 변화, Hybrid는 시간 특징과 통계 특징의 결합, ExtraTrees 계열 항목 모델은 고정 특징에서 항목 목표값을 학습합니다. 어떤 모델을 실제 사용했는지는 각 번들 manifest/실험 설정/학습 영수증으로 확인해야 합니다. 소스에 클래스가 존재한다는 사실만으로 현재 배포 모델이라고 결론 내리지 않습니다. 제공자 오류 유무 분류와 프로젝트의 0/1/2 다등급 채점은 라벨 과제가 다릅니다.

기존 모델·정규화·분할·실험 파일은 이번 통합에서 수정하지 않습니다. 스쿼트 전략의 실행 판정식과 모델도 유지합니다. 서식과 설명을 바꾼 파일은 실행 AST 비교로 알고리즘 불변을 확인합니다. 현재 측정 권위는 외부 프로젝트 기준서의 CURRENT 보고서이며 이 통합 README는 성능 보고서가 아닙니다.

## 검증

저장소 루트의 `python -m pytest`는 채점 계약과 새 통합 테스트를 실행합니다. `test_session_boundaries.py`는 세션 계층의 의존 방향, 중복 이벤트, pause/resume 이전 문맥 차단을 확인합니다. 라이브 카메라·GPU·동시 클라이언트 실기 검증과 독립 최종 Test는 별도입니다. 모델 파일이 있는 연구 체크와 합성 테스트를 구분해 결과를 읽어야 합니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [app/__init__.py](app/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/analysis/exercise.py](app/analysis/exercise.py) | 운동 분석 전략의 프로토콜·완료 이벤트와 세션 브리지를 정의합니다. 프레임 분석 전후 세션 문맥을 확인해 이전 운동 결과가 새 세션에 기록되지 않도록 합니다. | `RepCompleted`, `HoldCompleted`, `KeypointFrame`, `AnalysisResult`, `ExerciseAnalyzer`, `NoOpExerciseAnalyzer`, `PlaceholderExerciseAnalyzer`, `ExerciseAnalysisBridge` |
| [app/analysis/exercises/__init__.py](app/analysis/exercises/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/analysis/exercises/plank.py](app/analysis/exercises/plank.py) | 관측 가능한 플랭크 유지 상태와 연속 유지 시간을 추적하는 운동 전략입니다. 관절 누락과 자세 이탈 시 유지 구간의 연속성을 처리합니다. | `PlankConfig`, `PlankAnalyzer` |
| [app/analysis/exercises/push_up.py](app/analysis/exercises/push_up.py) | 푸시업의 운동 단계와 완료 반복 이벤트를 만드는 전략입니다. 프레임별 상태와 한 번의 완료 사건을 구분해 중복 카운트를 방지합니다. | `PushUpConfig`, `PushUpAnalyzer` |
| [app/analysis/exercises/shoulder_press.py](app/analysis/exercises/shoulder_press.py) | 숄더 프레스의 준비·밀기·복귀 단계와 완료 반복을 추적합니다. 하단 관측 여부와 완결된 복귀를 구분하며 미관측을 곧바로 오류로 확정하지 않습니다. | `ShoulderPressConfig`, `elbow_angle`, `DumbbellShoulderPressAnalyzer` |
| [app/analysis/exercises/squat.py](app/analysis/exercises/squat.py) | 기존 스쿼트 운동 전략입니다. 이 통합에서 판정식·임계값·모델을 개선하지 않으며 설명과 서식 변경은 실행 AST 보존으로 확인합니다. | `SquatConfig`, `knee_angle`, `SquatAnalyzer` |
| [app/analysis/exercises/ytw_raise.py](app/analysis/exercises/ytw_raise.py) | Y·T·W 단계 순서와 완결된 운동 구간을 추적합니다. 단계별 관절 관측과 반복 완료를 구분하며 부분 동작을 전체 반복으로 합산하지 않습니다. | `YTWRaiseConfig`, `YTWRaiseAnalyzer` |
| [app/analysis/frame_debug.py](app/analysis/frame_debug.py) | 프레임 단계별 진단 정보를 구성합니다. 진단용 필드는 운동 횟수·등급의 정답으로 사용하지 않습니다. | `fingerprint`, `compare`, `FrameDebug` |
| [app/analysis/keypoints.py](app/analysis/keypoints.py) | 추론기별 관절 출력을 운동 분석기가 사용할 좌표·가시성 형식으로 변환합니다. 좌표의 정규화 방식과 누락 관절을 명시적으로 전달합니다. | `extract_mediapipe_keypoints` |
| [app/analysis/posture/__init__.py](app/analysis/posture/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/analysis/posture/predictor.py](app/analysis/posture/predictor.py) | 저장된 자세 분류기와 규칙 평가를 공통 호출 형식으로 감쌉니다. 모델별 입력 계약과 미지원 항목을 호출 계층에 전달합니다. | `landmark_array`, `json_safe`, `PosturePredictor` |
| [app/analysis/posture/rule_based/__init__.py](app/analysis/posture/rule_based/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/analysis/posture/rule_based/base.py](app/analysis/posture/rule_based/base.py) | 학습 확률과 별개인 관절 기하 규칙 평가를 구성합니다. 관측 가능한 입력과 종목별 규칙·지원 항목을 명시적으로 연결합니다. | `Check`, `TemporalGate`, `MotionRange` |
| [app/analysis/posture/rule_based/config.py](app/analysis/posture/rule_based/config.py) | 학습 확률과 별개인 관절 기하 규칙 평가를 구성합니다. 관측 가능한 입력과 종목별 규칙·지원 항목을 명시적으로 연결합니다. | `RuleConfig` |
| [app/analysis/posture/rule_based/exercises.py](app/analysis/posture/rule_based/exercises.py) | 학습 확률과 별개인 관절 기하 규칙 평가를 구성합니다. 관측 가능한 입력과 종목별 규칙·지원 항목을 명시적으로 연결합니다. | `unavailable`, `asymmetry`, `SquatPosture`, `BodyPosture`, `PushUpPosture`, `PlankPosture`, `YtwPosture`, `PressPosture` |
| [app/analysis/posture/rule_based/geometry.py](app/analysis/posture/rule_based/geometry.py) | 학습 확률과 별개인 관절 기하 규칙 평가를 구성합니다. 관측 가능한 입력과 종목별 규칙·지원 항목을 명시적으로 연결합니다. | `PoseGeometry` |
| [app/analysis/posture/rule_based/registry.py](app/analysis/posture/rule_based/registry.py) | 학습 확률과 별개인 관절 기하 규칙 평가를 구성합니다. 관측 가능한 입력과 종목별 규칙·지원 항목을 명시적으로 연결합니다. | `RuleBasedPredictor` |
| [app/analysis/posture/service.py](app/analysis/posture/service.py) | 세션 문맥과 최신 관절 입력을 자세 판정 워커에 연결합니다. 오래된 문맥의 결과는 현재 운동에 적용하지 않습니다. | `PostureService` |
| [app/analysis/posture/vendor/__init__.py](app/analysis/posture/vendor/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/analysis/posture/vendor/classifier.py](app/analysis/posture/vendor/classifier.py) | 연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다. | `summary`, `PostureClassifier`, `SequenceSession` |
| [app/analysis/posture/vendor/feedback_policy.py](app/analysis/posture/vendor/feedback_policy.py) | 연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다. | `FeedbackPolicy` |
| [app/analysis/posture/vendor/networks.py](app/analysis/posture/vendor/networks.py) | 연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다. | `CNN`, `GRU`, `Hybrid` |
| [app/analysis/posture/vendor/squat_features.py](app/analysis/posture/vendor/squat_features.py) | 연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다. | `angle`, `features`, `resample` |
| [app/analysis/posture/vendor/suite_features.py](app/analysis/posture/vendor/suite_features.py) | 33개 관절 시계열에서 공통 기하·시간 특징과 관측 가능 비율을 계산합니다. 학습과 추론이 동일한 전처리 순서를 사용해야 합니다. | `make_features`, `prepare_landmarks` |
| [app/analysis/posture_state.py](app/analysis/posture_state.py) | 세션의 최신 자세 판정 투영을 보관합니다. 운동 이벤트의 횟수를 결정하는 소유자와 화면에 보여 줄 최신 상태를 구분합니다. | 패키지 경계/선언 |
| [app/analysis/registry.py](app/analysis/registry.py) | 운동 식별자를 분석 전략 팩터리에 연결합니다. 세션별로 새 전략을 생성해 다른 사람·운동의 단계 상태가 섞이지 않도록 합니다. | `ExerciseDefinition`, `list_exercises`, `get_exercise_definition`, `get_analyzer`, `RegisteredExerciseAnalyzer` |
| [app/analysis/shoulder_debug.py](app/analysis/shoulder_debug.py) | 숄더 프레스의 프레임·단계·차단 사유 진단을 기록합니다. 진단 로그의 유무가 운동 판정 결과를 바꾸지 않도록 분리합니다. | `context_fields`, `session_fields`, `ShoulderPressDebug` |
| [app/analysis/squat_debug.py](app/analysis/squat_debug.py) | 기존 스쿼트 검출 상태를 관찰하는 진단 보조 모듈입니다. 보존 대상 스쿼트 모델이나 임계값을 새로 학습하지 않습니다. | `SquatDebug` |
| [app/camera/__init__.py](app/camera/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/camera/stream.py](app/camera/stream.py) | HTTP 또는 RTSP 카메라 연결의 생성·프레임 읽기·종료를 감쌉니다. 재연결과 입력 선택은 운동 전략의 업무 규칙과 분리됩니다. | `CameraStream` |
| [app/debug_logging.py](app/debug_logging.py) | 운동 진단 로그만 JSONL 파일로 분리합니다. 기존 디버그 플래그/판정은 유지합니다. | `debug_log_files` |
| [app/inference/__init__.py](app/inference/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/inference/mediapipe_pose.py](app/inference/mediapipe_pose.py) | MediaPipe Tasks 추론기를 동일 워커에서 생성·호출·종료합니다. 관절 좌표와 가시성을 반환하는 외부 모델 어댑터입니다. | `MediapipePoseEstimator` |
| [app/inference/yolo_pose.py](app/inference/yolo_pose.py) | YOLO 자세 모델의 로드·추론·시각화 출력을 감쌉니다. 원시 추론 결과는 관절 변환기를 거쳐 분석 전략에 전달됩니다. | `YOLOPoseEstimator` |
| [app/main.py](app/main.py) | AI 서버의 설정·모델 팩터리·세션 관리자·추론 스트림을 조립하는 진입점입니다. 실제 카메라·모델 작업은 앱 수명 시작 이후에만 실행합니다. | `read_camera_settings`, `create_pose_estimator`, `lifespan`, `stream_video`, `health_check`, `get_latest_pose`, `get_metrics` |
| [app/metrics.py](app/metrics.py) | 파이프라인 단계별 시간·FPS·지연 정보를 모읍니다. 실행 속도 계측은 모델 정확도나 자세 점수와 별개입니다. | `PipelineMetrics` |
| [app/routers/__init__.py](app/routers/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/routers/internal_sessions.py](app/routers/internal_sessions.py) | 내부 키로 보호된 세션·운동 명령을 SessionManager에 전달합니다. HTTP 오류 변환을 담당하고 관절 분석을 라우터에서 직접 실행하지 않습니다. | `verify_api_key`, `InternalSessionRoute`, `get_session_manager`, `start_session`, `start_exercise`, `pause_session`, `resume_session`, `finish_exercise`, `finish_session`, `live_session` |
| [app/schemas/exercise.py](app/schemas/exercise.py) | API 요청·응답의 전송 자료형을 정의합니다. 유효한 JSON 형식과 실제 자세 판정의 정확성은 별개입니다. | `SessionStartRequest`, `KeypointData`, `LiveAnalysisResult`, `ExerciseRepRecord`, `SessionResult` |
| [app/schemas/pose.py](app/schemas/pose.py) | API 요청·응답의 전송 자료형을 정의합니다. 유효한 JSON 형식과 실제 자세 판정의 정확성은 별개입니다. | `PoseKeypoint`, `PoseResponse` |
| [app/schemas/session.py](app/schemas/session.py) | API 요청·응답의 전송 자료형을 정의합니다. 유효한 JSON 형식과 실제 자세 판정의 정확성은 별개입니다. | `ExerciseStartRequest`, `SessionSnapshot`, `ExerciseSnapshot`, `HoldExerciseSnapshot`, `PostureAnalysisSnapshot`, `SessionLiveResponse` |
| [app/scoring/__init__.py](app/scoring/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/scoring/__main__.py](app/scoring/__main__.py) | 패키지를 명시적으로 실행할 때 명령행 진입점에 제어를 전달합니다. import만으로 작업을 시작하지 않습니다. | `demo_sequence`, `main` |
| [app/scoring/engine.py](app/scoring/engine.py) | 검증된 관절 구간에서 항목별 모델과 규칙 점수를 조합합니다. 관측 불가·미지원 항목을 구분하며 해당 구간의 결과를 반환할 뿐 DB 세션 횟수를 소유하지 않습니다. | `ScoringEngine` |
| [app/scoring/features.py](app/scoring/features.py) | 채점 모델에 전달할 관절 기하·구간 통계 특징을 구성합니다. 특징 순서·누락 처리·시점 뱅크를 학습과 추론에서 동일하게 유지해야 합니다. | `describe` |
| [app/scoring/geometry.py](app/scoring/geometry.py) | 관절 좌표의 각도·거리·정규화 기하 함수를 제공합니다. 좌표계와 관측 불가 입력을 다루며 외부 통신은 수행하지 않습니다. | `angle_deg`, `channels`, `model_features` |
| [app/scoring/held_geometry.py](app/scoring/held_geometry.py) | 유지형 동작에서 관절 기하 특징의 관측 가능성을 계산합니다. 누락 관절로 만들 수 없는 특징은 평가 보류로 전달합니다. | `plank_clearance`, `plank_support_grade`, `squat_foot_change` |
| [app/scoring/media.py](app/scoring/media.py) | 저장 영상과 관절 추출을 채점 구간에 연결하는 파일·미디어 어댑터입니다. 모델 특징 계산과 파일 수명을 분리합니다. | `extract_video`, `load_landmarks`, `save_landmarks` |
| [app/scoring/partition.py](app/scoring/partition.py) | 학습·검증·시험 분할에서 사람·출처·겹치는 구간의 누수를 검사합니다. 시험 데이터를 학습으로 옮기는 보정은 수행하지 않습니다. | `audit_partitions` |
| [app/scoring/plank_item_observation.py](app/scoring/plank_item_observation.py) | 플랭크 채점 항목별 관측 가능 조건을 계산합니다. 자세가 보이지 않는 상태와 실제 오류 라벨을 구분합니다. | `assess`, `features` |
| [app/scoring/recovery_w_phase.py](app/scoring/recovery_w_phase.py) | YTW 복귀 구간과 W 단계 특징을 분리합니다. 운동 단계 범위와 채점 대상 구간의 일치를 확인하는 보조 규칙입니다. | `detect` |
| [app/scoring/report.py](app/scoring/report.py) | 구간 채점 결과를 영상 오버레이와 HTML로 표현합니다. 표시를 위한 변환이며 점수·라벨을 다시 판정하는 단계가 아닙니다. | `overlay`, `write_html` |
| [app/scoring/rule_score.py](app/scoring/rule_score.py) | 사람이 정의한 관절 기하 규칙의 항목 점수를 계산합니다. 학습 모델 점수와 구분해 결합 계층에 전달합니다. | 패키지 경계/선언 |
| [app/scoring/rules.py](app/scoring/rules.py) | 종목별 채점 항목과 지원 조건을 선언합니다. 항목 가중치와 지원 여부는 점수의 해석 계약이므로 모델 결과와 함께 검토해야 합니다. | `rule_grade`, `classify`, `grades` |
| [app/scoring/sequence.py](app/scoring/sequence.py) | 프레임 시각·관절 배열·화면 크기를 하나의 운동 구간으로 검증합니다. 영상과 관절의 구간 연결을 확인하고 다른 출처의 잘못된 결합을 막습니다. | `PoseSequence`, `verify_pair` |
| [app/scoring/three_items.py](app/scoring/three_items.py) | 항목별 특징·평가 자격·채점 뱅크를 구성합니다. 지원 근거가 없는 항목은 관측 보류로 남겨 총점의 해석과 분리합니다. | `apply_view_policy`, `mask_image_border`, `describe`, `total` |
| [app/scoring/training.py](app/scoring/training.py) | 동결된 출처·사람 분할을 감사한 뒤 항목별 후보 모델을 다시 학습합니다. 최종 시험 또는 규칙 계산값을 정답 학습 입력으로 받아들이지 않습니다. | `refit_heads` |
| [app/scoring/w_morphology.py](app/scoring/w_morphology.py) | W 자세의 관절 배치와 투영 형태 특징을 계산합니다. 원근·시점의 영향과 실제 운동 오류를 구분하기 위한 특징 계층입니다. | `evaluate` |
| [app/scoring/ytw_features.py](app/scoring/ytw_features.py) | YTW 구간에서 학습·추론에 공통으로 쓰는 수치 특징을 만듭니다. 특징 순서와 단위 변경은 기존 모델 입력 계약을 바꾸므로 별도 버전 검증이 필요합니다. | `describe_ytw`, `phase_matches`, `refined_w`, `observed_absence` |
| [app/scoring/ytw_partial_trial.py](app/scoring/ytw_partial_trial.py) | 부분 YTW 동작의 관측·지원 여부를 계산합니다. 불완전 구간을 완결된 실패 복귀의 확정 라벨로 사용하지 않습니다. | `detect` |
| [app/scoring/ytw_phases.py](app/scoring/ytw_phases.py) | YTW의 단계 후보와 구간 경계를 계산합니다. 프레임 좌표를 구간 정보로 변환하며 DB나 HTTP에 의존하지 않습니다. | `detect` |
| [app/session/__init__.py](app/session/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/session/catalog.py](app/session/catalog.py) | 모델 구현에 의존하지 않는 운동 식별자와 반복·유지 유형입니다. | `ExerciseSpec`, `get_exercise_spec` |
| [app/session/manager.py](app/session/manager.py) | 세션·현재 운동·완료 이벤트의 단일 상태 소유자입니다. 잠금과 문맥 revision 검증으로 중복·지연 이벤트를 차단하며 API와 분석 워커가 같은 인스턴스를 공유합니다. | `SessionManager` |
| [app/session/models.py](app/session/models.py) | 세션·운동·유지 상태와 분석 문맥·업무 예외를 표준 라이브러리 자료형으로 정의합니다. HTTP·DB·모델 라이브러리를 직접 참조하지 않습니다. | `AnalysisContext`, `SessionError`, `InvalidSessionInput`, `SessionNotFound`, `ExerciseNotFound`, `SessionConflict`, `SessionState`, `ExerciseState`, `HoldState` |
| [app/session/projections.py](app/session/projections.py) | 모델·웹 프레임워크에 의존하지 않는 세션 자세 투영의 기본값입니다. | `posture_state` |
| [app/streaming/__init__.py](app/streaming/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [app/streaming/output.py](app/streaming/output.py) | 표시할 프레임을 JPEG 바이트로 인코딩합니다. 인코딩 실패는 빈 영상 상태로 처리하며 운동 이벤트를 새로 생성하지 않습니다. | `encode_jpeg` |
| [app/streaming/pose_stream.py](app/streaming/pose_stream.py) | 카메라 수신, 추론 워커, 운동 분석, 자세 판정과 JPEG 출력을 조립합니다. 최신 영상 경로와 완료 운동 이벤트 경로의 수명·전달 책임을 구분합니다. | `PoseStreamService` |
| [tests/scoring/test_cli.py](tests/scoring/test_cli.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `CliTest` |
| [tests/scoring/test_engine.py](tests/scoring/test_engine.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `SequenceTest`, `EngineTest`, `TrainingSupportTest` |
| [tests/scoring/test_geometry.py](tests/scoring/test_geometry.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `HeldGeometryTests` |
| [tests/scoring/test_partition.py](tests/scoring/test_partition.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `row`, `PartitionTest` |
| [tests/scoring/test_training.py](tests/scoring/test_training.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `records`, `TrainingTest` |
