# 학습 기반 자세 평가 통합

## 모델 진단 로그

학습 모델 모드에서는 `logs/debug.jsonl`과 같은 폴더의 `logs/model_debug.json`에
진단을 자동 기록합니다. 파일명 확장자는 `.json`이지만 기존 디버그 로그처럼
한 줄에 JSON 객체 하나를 저장하며, 10 MiB마다 회전하고 백업 5개를 유지합니다.
첫 진단 시 파일이 생성됩니다. `MODEL_DEBUG=0`으로 기록을 끌 수 있습니다.

필수 관절 ID별 visibility/유효 여부/실패 이유, 누락·중복 관절, 입력 크기·영상 비율,
처리 전후 버퍼 길이와 초기화 원인, 샘플 간격, 원시 분류 점수와 최종 상태,
화면 피드백을 기록합니다. 원본 영상은 저장하지 않습니다.
`not_sampled_this_frame`은 기록하지 않으며, `timestamp`로 스트림 시각을 대조합니다.
`classification.model_outputs`가 없으면 해당 기록은 추론 전의 입력 실패·준비 상태입니다.

```bash
tail -f logs/model_debug.json
```

이 모듈은 팀에서 전달한 `posture_classification_handoff_v2_feedback`의 실제 PyTorch
모델·전처리·시퀀스·분류 및 V2 표시 정책을 사용합니다. Rule-based Reps/Plank 유지 시간은
변경하지 않습니다. 최종 화면에는 `display_feedback.text`를 사용합니다.

## 인수인계 분석

원본은 디렉터리이며 README/HANDOFF_STATUS/FEEDBACK_POLICY_V2, config 6개,
src/posture_classification 런타임, models, examples, tests, evaluation으로 구성됩니다.
모델은 state_dict와 학습 메타데이터를 담은 `.pt` 체크포인트입니다. 기본 오류 모델은
5종목 × CNN/GRU = 10개, 스쿼트 이진 품질 모델 2개, 별도 실험 모델 8개입니다.
실험 모델은 기본 통합에 사용하지 않습니다. 클래스/모델/전처리의 원본 해시는
[handoff_provenance.json](validation/handoff_provenance.json)에 기록했습니다.

CNN: Conv1d(F→48,k5) → ReLU → Conv1d(48→64,k3) → ReLU → 평균 pooling → Linear.
GRU: hidden 48, batch_first → 시간 평균 → Linear. 스쿼트 품질 모델은 별도
Hybrid temporal+759 요약 특징 헤드입니다. Dropout은 eval 모드에서 꺼집니다.
모델 eval + torch.inference_mode, weights_only=True 로딩을 사용합니다.
일반 pickle/joblib 및 weights_only=False fallback은 실행하지 않습니다.

| AIFit 이름 | 실제 모델 폴더 | 입력 shape | 오류 출력 수 |
|---|---|---|---|
| squat | squats | (N,32,69) + 품질 헤드 요약 (N,759) | 2 + 품질 sigmoid 1 |
| push_up | pushups | (N,32,109) | 7 |
| ytw_raise | ytw | (N,32,109) | 5 |
| plank | elbow_plank | (N,32,109) | 9 |
| dumbbell_shoulder_press | shoulder_press | (N,32,109) | 7 |

모델별 경로는 `models/<폴더>/cnn.pt` 또는 `gru.pt`이고 스쿼트에는
quality_cnn.pt/quality_gru.pt 및 quality_normalization.npz도 필요합니다.
정확한 관절/출력 순서와 저장 임계값은 [model_contract.json](validation/model_contract.json)에
있으며 런타임은 전달 디렉터리의 config/model_registry.json을 읽습니다.
모델은 운동 ID가 아닌 기존 Registry의 운동 이름으로 선택합니다.

MediaPipe의 첫 사람 **이미지 좌표** 33개를 ID별 x/y/z/visibility 순서로 전달합니다.
Presence/world landmarks는 학습 입력이 아니므로 추가하지 않습니다.
스쿼트 특징 관절 순서는 11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32입니다.
나머지는 0,7,8,11~32 순서입니다. x/z를 width/height로 보정하고 골반 중심화,
어깨-골반 길이 정규화, 기존 각도 특징, 체크포인트 mean/std를 그대로 사용합니다.
특징 계산 결과는 원본 학습 전처리와 배열 단위로 일치함을 테스트했습니다.
관절을 이름/좌우로 재배열하거나 미러링하지 않습니다.

모든 기본 모델은 독립 sigmoid의 **multi-label** 출력입니다. Softmax를 사용하지 않으며
출력을 보정된 정확도로 해석하지 않습니다. Score 정의가 없어 `form_score=null`입니다.
원래 학습 클래스의 실제 의미:

- squat: shallow(얕은 동작), back not straight(등 정렬 오류).
- push_up: shallow, butt sinking, elevated hips, head down, moving head, side to side.
- plank: arms facing inward, arms too wide, butt sinking, head down, head up, high hips,
  legs and hips on the floor, shoulders shrugged up toward the ears.
- ytw_raise: arms too low, wrong order, too fast, too slow.
- dumbbell_shoulder_press: arms too wide at the top, incomplete upward range,
  hands move behind the head, forearms tilt inward, elbow path narrower than reference,
  uneven arm height and timing. **검증되지 않은 관찰 초안이며 최종 판정/교정에 사용하지 않습니다.**

스쿼트 이외 마지막 출력은 `any_selected_error` 보조 헤드이며 별도 자세 오류로 만들지
않습니다. 스쿼트만 이진 모델의 제한된 참조 범위에서 NORMAL을 지원합니다. 나머지
운동의 오류 미검출을 정상으로 바꾸지 않으며 숄더는 원본 제한대로 UNKNOWN입니다.
`is_correct=true` 역시 모든 자세에 대한 인증이 아닙니다. 원본 `normal_scope`,
`normal_certified=false`, confidence_kind와 원시 model_outputs를 함께 보존합니다.

V2 화면 피드백은 ERROR + 허용된 클래스 + 원시 점수 0.85 이상, 최근 9회 중 5회,
연속 4회 및 1초 지속을 모두 요구합니다. 조건 미충족 시 `Keep going.`입니다.
이 문구는 정상 인정이 아닙니다. 원시 classification.feedback을 바로 화면에 표시하지
않습니다. 전달받은 영어 피드백을 그대로 사용하며 새 교정 문구를 만들지 않았습니다.

## 실행 구조와 설정

```text
기존 CameraStream → MediaPipe → 기존 Rule-based Bridge → Reps/Hold
                             └→ 최신 관절 슬롯 1개 → Posture 워커 → 세대 재검증
기존 GET /internal/sessions/{session_id}/live ← 동일 SessionManager Snapshot
```

`.env`에서 설정하고 서버를 재시작합니다. 키가 없으면 기본 OFF이며 실제 `.env`에는
기존 값을 덮어쓰거나 중복 키를 만들지 않고 새 항목만 추가했습니다.

```ini
POSTURE_MODEL_ENABLED=true
POSTURE_MODEL_DIR=/home/ssw/Desktop/posture_classification_handoff_v2_feedback
POSTURE_MODEL_ARCHITECTURE=cnn
POSTURE_MODEL_DEVICE=cpu
```

기본 CPU는 MediaPipe의 GPU와 추가 VRAM을 공유하지 않도록 선택했습니다. GRU 선택과
명시적 `cuda`도 지원하며 CUDA 장치/모델 이동 실패는 unavailable로 반환합니다.
현재 NumPy 2.5.4, torch 2.14.1+cu130에서 검증했습니다. 인계 requirements의 NumPy
1.26.4/torch 2.14.0으로 다운그레이드하지 않았고 패키지를 설치하지 않았습니다.
기존 requirements에 설치된 버전과 일치하는 torch 의존성만 명시했습니다.

모델은 자세 워커에서 최초 선택 시 1회 로딩·캐시됩니다. 운동 시작 후 카메라가 없어도
워커의 0.25초 점검에서 준비합니다. 시퀀스 32개(약 6Hz)를 모으기 전에는 warming_up입니다.
학습이 허용한 관절 마스킹/보간만 원본 코드에서 사용하며, 누락한 관절을 외부 어댑터에서
채우거나 낮은 FPS를 새 보간 방식으로 높이지 않습니다. 33개 좌표가 누락/비유한이면
기존 SequenceSession 정책으로 버퍼를 초기화합니다.

순간 프레임이 많으면 원본처럼 6Hz로 내려 샘플링합니다. 32개 시퀀스에서 최대 간격
0.35초 이하, 중앙 간격 0.12~0.22초 조건을 검사합니다. 3FPS 같은 저속 입력은
sampling_interval_mismatch로 보류합니다. 1초 초과 끊김/시간 역행/영상 비율 변경도
원본 정책대로 초기화합니다. 실제 촬영 시각이 아닌 기존 워커의 monotonic 관측 시각을
재사용하므로 네트워크/디코더 내부 지연의 영향을 없애지는 못합니다.

Pause/Resume/Finish/운동·세션 전환 시 기존 AnalysisContext revision을 재사용하여
시퀀스·피드백 이력을 폐기합니다. 모델은 재로딩하지 않습니다. 상태 전이 후 즉시 Live
조회는 이전 예측을 숨기고, 분석 전후 재검증과 저장 잠금에서 지연 결과를 거부합니다.
잠금 중에는 모델 추론을 실행하지 않습니다. 동일 프레임은 중복 처리하지 않습니다.
대기 및 실행 후 1초 초과된 표본은 새 예측으로 게시하지 않으며, 조회 시에도 1초 이상
갱신이 없으면 stale_prediction으로 보류합니다. 카메라가 멈춰도 캐시를 계속 신규
결과처럼 전송하지 않습니다. 단일 슬롯은 늦은 작업을 덮어쓰며 원시 영상을 보관하지 않습니다.

## Live 계약

현재 AIService에는 Backend 전송 HTTP 클라이언트/Live POST가 없고,
`Server/AIFitService/main.py`는 0바이트입니다. 따라서 새 POST API를 만들지 않았고
실제 Backend 스키마 호환성과 POST 성공은 검증할 수 없습니다. 현재 요청 방향은
AIFit → AI Server의 GET 조회입니다. 기존 URL/메서드/인증 Header는 유지합니다.

ON에서는 기존 session/exercise 및 Reps/Hold 필드와 함께 최상위 `posture_analysis`가
추가됩니다. OFF에서는 이 필드 자체를 생략해 기존 JSON 형식을 유지합니다.
다음은 **버퍼 수집 전 상태 예시**이며 실제 분류 측정값이 아닙니다.

```json
{
  "session": {"session_id": 5, "session_status": "running"},
  "exercise": {"exercise_id": 7, "exercise_name": "push_up", "rep_count": 3},
  "posture_analysis": {
    "status": "warming_up",
    "reason": "new_analysis_context",
    "is_correct": null,
    "form_score": null,
    "posture_errors": [],
    "predicted_at": null,
    "classification": null,
    "display_feedback": null
  }
}
```

위 session/exercise는 설명을 위해 일부 필드만 적었습니다. 원래 실제 응답 필드는 모두
유지합니다. ready가 되면 classification에 원본 NORMAL/ERROR/UNKNOWN, 실제 오류명,
class_indices, model_outputs를 포함하며 display_feedback에 V2 표시 결과를 담습니다.
`predicted_at`/`observed_at`은 서버 monotonic 초이며 UTC/DB 저장 시각이 아닙니다.
frame_id는 해당 공유 Pose/JPEG 게시 프레임 ID입니다.

모델 파일 누락/설정 오류/추론 예외는 unavailable, 관절·샘플링 부족은 insufficient_pose,
준비 중 warming_up, 일시정지는 paused, 운동 없음/완료는 inactive입니다. ready는 모델
계산 완료라는 뜻이며 classification.status가 UNKNOWN일 수도 있습니다.
배열/Tensor/NaN/Infinity를 직접 JSON으로 전송하지 않습니다. 별도 원시 오류 점수는
classification.model_outputs에서 확인하며 form_score로 변환하지 않습니다.
Backend 담당자는 위 선택 필드를 GET 응답 모델에 허용하고 원시 feedback 대신
`display_feedback.text`를 표시하도록 계약을 확정해야 합니다. POST를 추가하려면 실제
수신 URL·인증·주기·재시도 계약이 필요합니다. DB/GUI/Backend 코드는 변경하지 않았습니다.

## 성능과 검증

[cnn_benchmark.json](validation/cnn_benchmark.json),
[gru_benchmark.json](validation/gru_benchmark.json)은 현재 CPU에서 제공된 관절 캐시로
각 30회 측정한 전처리+모델+원본 판정 시간입니다. 호출 전 1회 warm-up을 분리했습니다.
CNN 평균(ms): squat 4.48, push_up 13.95, ytw_raise 2.78, plank 6.63, shoulder_press 2.88.
GRU 평균(ms): squat 4.86, push_up 1.76, ytw_raise 1.96, plank 2.13, shoulder_press 1.87.
측정 환경은 torch 2.14.1+cu130, NumPy 2.5.4, PyTorch CPU 스레드 14개입니다.
정확한 P95는 JSON에 기록했습니다. 시스템 부하가 통제된 성능 비교가 아니므로 이 결과로
기본 모델/전역 스레드 수를 바꾸지 않았습니다. 모델 선택은 인계 기본 CNN을 따릅니다.

제공 샘플은 QEVD 관절 캐시 1개뿐입니다. 스쿼트 외 실행은 동일 입력 규격의 smoke test이며
각 운동의 실제 영상 정확도를 검증한 것이 아닙니다. 현재 CPU 프로세스의 peak RSS는
약 501MB(CNN)/503MB(GRU)이며 torch 런타임 포함 전체 프로세스 값입니다. 모델 추가에
따른 메모리 증가량이나 실제 서버 VRAM 사용량이 아닙니다.

실제 카메라의 OFF/ON FPS, MediaPipe GPU 동시 구동 및 CUDA 자세 모델의 정상 실행,
실제 Backend POST, 장시간 네트워크 연결은 아직 검증하지 않았습니다. 기존 /metrics
스키마는 유지하고 별도 [PostureMetrics] 로그에서 camera_receive_fps,
mediapipe_inference_ms, posture_inference_ms/fps, end_to_end_analysis_ms, 모델/운동 이름,
prediction_status를 초당 최대 1회 기록합니다. posture_inference_ms는 이번 push의 특징
계산/모델/후처리를 포함하고, end_to_end는 기존 MediaPipe 시작부터 자세 워커 완료까지
단일 표본의 소요 시간입니다. 전송/센서 촬영 지연 전체를 뜻하지 않습니다.

가상환경에서 실행:

```bash
python -m unittest discover -s tests -p 'test_posture_integration.py' -v
POSTURE_MODEL_ENABLED=false CAMERA_SOURCE_TYPE=http ESP32_STREAM_URL=http://camera/stream \
python -m unittest discover -s tests -v
python scripts/benchmark_posture.py --architecture cnn --iterations 30
python scripts/benchmark_posture.py --architecture gru --iterations 30
```

실장비는 기존 GPU 실행 명령에 --workers 1을 유지합니다. .env에서 자세 기능만 OFF/ON해
동일 장면/해상도로 각각 5분 실행하고 /metrics와 [PostureMetrics]를 저장해 비교합니다.
각 운동을 시작한 뒤 32개 유효 표본 수집, Pause/Resume, 운동 전환, 카메라 중단,
누락 모델, /stream 1개/여러 클라이언트를 확인합니다. CUDA를 선택하려면 별도로
`nvidia-smi`의 사용률·메모리와 MediaPipe 동시 지연을 측정해야 합니다.

이 저장소는 tests/와 scripts/가 Git ignore되어 있습니다. 새 테스트/벤치마크를
커밋할 때 해당 파일을 명시적으로 `git add -f`해야 합니다. 모델 가중치는 외부 인계
디렉터리에 두며 Git에 중복 추가하지 않습니다.

## 원본 보존 범위

vendor/에는 검토하고 manifest 해시를 확인한 인계 런타임 5개만 복사했습니다.
networks/squat_features/suite_features는 동일 코드입니다. classifier는 안전 로딩,
체크포인트 계약 검증, 명시 장치 이동만 추가했고 피드백은 root 경로 주입만 추가했습니다.
원래 MediaPipeFrameAdapter는 또 다른 Pose 객체를 생성하므로 가져오지 않습니다.
Reps/Hold 운동 분석기, 카메라 수신, MediaPipe 모델 코드는 이번 통합에서 수정하지
않았습니다. 기존 작업 트리의 YTW/카메라 변경도 되돌리거나 덮어쓰지 않았습니다.


## 이번 변경 파일 및 최종 테스트

추가: posture/__init__.py, predictor.py, service.py, vendor/의 런타임 5개와 __init__.py,
posture_state.py, 본 문서 및 validation/의 계약·출처·실측·테스트 결과,
tests/test_posture_integration.py, scripts/benchmark_posture.py.
수정: app/main.py, streaming/pose_stream.py(주입/submit만), session/manager.py(결과 캐시),
schemas/session.py, routers/internal_sessions.py(선택 필드 직렬화), .env.example,
requirements.txt, app/README.md. 실제 .env에는 없는 POSTURE 항목만 추가했습니다.

최종 전체 391개 중 383개 통과, 기존 실패 8개입니다. 신규 자세 테스트 32개는 모두
통과했습니다. 기존 실패는 로거 1개, PushUp 2개, YTW 진단 로그 5개이며 사용자 변경을
보존한 별도 변경 전 복사본에서도 같은 실패를 재현했습니다. 신규 회귀 실패는 없습니다.
기존 운동별 Reps/Hold 동작 테스트는 유지됐으며 YTW 로그 실패를 수정하지 않았습니다.
카메라/MediaPipe/5종 Rule-based 파일은 통합 시작 시점과 바이트 단위로 동일함을
검증했습니다. [테스트 요약](validation/test_summary.json)에 상세 이름과 미실시 항목이 있습니다.

## 별도 Rule-based 자세 오류 평가

`POSTURE_EVALUATION_MODE=rule_based/model/off`로 방식을 하나만 선택할 수 있습니다.
키를 생략하면 기존 POSTURE_MODEL_ENABLED 동작을 유지합니다. 기존 학습 모델의 전처리,
분류/피드백 정책은 변경하지 않았습니다. 새 15개 오류의 초기 실험 기준·촬영 방향·부분 평가
및 모드 우선순위는 [Rule-based 문서](rule_based/README.md)를 참고하세요.
