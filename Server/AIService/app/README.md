# AIService 성능 모니터링

기존 `CameraStream` 수신 스레드 → 최신 프레임 슬롯 → `PoseStreamService` 단일 추론
스레드 → JPEG/관절 결과 공유 구조를 유지합니다. `/stream` 구독자와 `/pose/latest`,
`/health`, `/metrics` 요청은 같은 서비스 객체를 사용합니다. 추가 GPU 객체나 카메라
연결은 생성하지 않습니다. 로그 전용 스레드는 모델을 호출하지 않습니다.

## 실행

이 디렉터리에서 기존 AI 서버 가상환경을 활성화하고 기존 `.env`의 카메라 URL,
Heavy 모델 경로, `POSE_MODEL=mediapipe`, `MEDIAPIPE_DELEGATE=gpu` 설정을 유지합니다.

```bash
METRICS_LOG_INTERVAL_SECONDS=5 python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
curl http://localhost:8000/metrics
```

`METRICS_LOG_INTERVAL_SECONDS`는 양의 유한한 초 값이며 기본값은 5입니다.
`uvicorn.error` INFO 로거에서 `[Performance]` JSON을 출력합니다. INFO 로그를
비활성화하면 출력되지 않습니다. 프레임이 오지 않아도 로그는 계속 출력됩니다.
단일 프로세스(`--workers 1`)로 실행해야 카메라/모델이 프로세스별로 중복 생성되지 않습니다.

## 지표 정의

| JSON 필드 | 실제 측정 범위 |
|---|---|
| `camera_capture_fps` | OpenCV `capture.read()`로 정상 수신·디코딩한 프레임 수 / 시간 |
| `pose_inference_fps` | predict, 관절 추출, 기존 동기화까지 성공한 횟수 / 시간 |
| `processing_fps` | JPEG 인코딩 후 공유 결과 게시에 성공한 횟수 / 시간 |
| `mediapipe_inference_ms` | 동기식 `detect_for_video()` 호출의 벽시계 소요 시간 |
| `inference_ms` | 전처리, 모델 호출, 프레임 복사, 스켈레톤 그리기, 관절 추출을 포함한 추론 단계 |
| `jpeg_encoding_ms` | `encode_jpeg()` 호출 전체: `cv2.imencode`와 `tobytes()` 포함 |
| `latest_frame_age_ms` | 마지막 정상 수신 프레임이 서버의 최신 슬롯에 기록된 후 경과 시간 |
| `skipped_frames` | 소비되기 전에 새 수신 프레임으로 덮어써진 최신 슬롯의 누적 프레임 수 |
| `captured_frames`, `inferred_frames`, `processed_frames` | 서비스 객체 생성 이후 각 단계의 누적 성공 횟수 |

FPS는 최근 5초 구간의 완료 건수 / 구간 길이입니다. 시작 후 5초 미만이면 서비스
생성 이후 경과 시간을 분모로 사용합니다. 5초 동안 새 완료가 없으면 0으로 떨어집니다.
로그 주기를 바꿔도 FPS 구간은 바뀌지 않습니다. 영상에 표시하는 기존 30프레임 FPS는
호환성을 위해 유지하므로 `/metrics`의 FPS와 다를 수 있습니다.

시간 지표는 각각 최근 300개 표본의 `latest`, `mean`, `p95`, `samples`를 제공합니다.
P95는 정렬한 표본에서 `ceil(0.95 * N)`번째 값(nearest-rank)입니다. 시간 통계는
시간 구간이 아니라 표본 구간이므로 정지 중에는 마지막 통계를 유지합니다. 표본이
없으면 시간은 `null`, 표본 수는 0입니다. JPEG 인코딩이 `None`을 반환해도 소요 시간은
기록하지만 처리 성공 건수는 올리지 않습니다. 예외로 중단된 호출은 시간 표본에서 제외합니다.

모든 경과 시간은 단조 시계로 측정합니다. 프레임 경로에서는 짧은 잠금으로 이벤트와
표본만 기록하며, 정렬은 API 조회/로그 시에 복사한 표본으로 잠금 밖에서 수행합니다.

## API 응답 예시

다음은 **프레임 수신 전 초기 상태**의 스키마 예시이며 실장비 성능 수치가 아닙니다.

```json
{
  "fps_window_seconds": 5.0,
  "timing_sample_limit": 300,
  "camera_capture_fps": 0.0,
  "pose_inference_fps": 0.0,
  "processing_fps": 0.0,
  "latest_frame_age_ms": null,
  "skipped_frames": 0,
  "captured_frames": 0,
  "inferred_frames": 0,
  "processed_frames": 0,
  "inference_ms": {"latest": null, "mean": null, "p95": null, "samples": 0},
  "mediapipe_inference_ms": {"latest": null, "mean": null, "p95": null, "samples": 0},
  "jpeg_encoding_ms": {"latest": null, "mean": null, "p95": null, "samples": 0}
}
```

기존 MJPEG 형식과 Pose JSON 필드는 유지합니다. `/health`는 결과 조회 시 사용하던
`process_at` 오타를 실제 필드인 `processed_at`으로 수정했습니다.

## 테스트

집계기 테스트는 Python 표준 라이브러리만 사용합니다.

```bash
python -m unittest discover -s tests -p test_metrics.py -v
```

전체 테스트는 기존 서버 의존성과 FastAPI TestClient용 `httpx`가 필요합니다.
카메라, 모델 생성, CUDA 호출은 하지 않으며 카메라/추론은 대역으로 검증합니다.

```bash
python -m unittest discover -s tests -v
```

검증 항목: 초기 null 값, FPS/age/정지 상태, 평균/P95/표본 제한, 동시 조회,
슬롯 덮어쓰기 누락 수, MediaPipe 호출 측정, JPEG 실패, 구독자 간 결과 공유,
API 정상/초기/오래된 결과, 주기 로그, 반복 start의 중복 생성 방지.
기존 `app/test_main.py`는 실행 가능한 독립 테스트가 아닌 예제 코드이므로
테스트 탐색 경로는 `tests`로 지정합니다.

## 측정 한계와 잠재적 병목

- Capture FPS는 ESP32 센서 촬영 FPS가 아닙니다. 네트워크 전송, MJPEG 디코딩,
  FFmpeg 내부 버퍼의 영향을 받습니다. 카메라 타임스탬프가 없어 촬영부터 서버까지
  지연 및 네트워크/디코더에서 버려진 프레임 수는 알 수 없습니다.
- Age는 서버 수신 시점 기준입니다. 마지막 추론 결과의 나이나 브라우저 표시 지연이
  아닙니다. 연결이 끊겨도 마지막 수신 시각부터 계속 증가합니다. 첫 수신 전에는 null입니다.
- Skipped는 최신 슬롯 덮어쓰기만 집계합니다. 재연결/종료 시 슬롯 폐기, 추론 예외,
  JPEG 실패, 느린 MJPEG 구독자의 전송 누락은 포함하지 않습니다.
- MediaPipe 시간은 API 호출 전체의 CPU/GPU 작업과 내부 대기를 포함합니다.
  순수 GPU 커널 실행 시간이나 GPU 사용률은 분리할 수 없습니다.
  MediaPipe GPU delegate에는 PyTorch CUDA 동기화를 추가하지 않았습니다.
- BGR→RGB 변환, `mp.Image` 구성, 프레임 복사, Python 스켈레톤 그리기,
  관절 추출이 `inference_ms`와 `mediapipe_inference_ms` 사이 차이를 만들 수 있습니다.
- JPEG 압축/bytes 복사는 같은 추론 스레드에서 직렬 실행됩니다. 인코딩이 느리면
  다음 추론 시작이 늦어지고 슬롯 덮어쓰기 누락이 증가할 수 있습니다.
- CameraStream의 프레임 복사는 Condition 잠금 안에서 수행되어 큰 프레임에서는
  수신 스레드가 잠시 대기할 수 있습니다. 이 구조는 변경하지 않았습니다.
- 전체 처리 FPS는 서버 게시 처리량입니다. HTTP 전송 완료나 클라이언트 렌더링
  FPS가 아닙니다. 느린 구독자는 최신 결과만 가져오며 중간 결과를 건너뜁니다.
- 기존 관절 추출 경로는 MediaPipe 결과를 전제로 합니다. YOLO 결과 호환성 문제는
  이번 계측 작업 범위에 포함하지 않았습니다.

실제 RTX 3050 Ti 및 ESP32-CAM 성능과 계측 오버헤드는 실장비로 측정해야 합니다.
위 항목은 코드에서 확인한 병목 후보이며 실측 병목으로 확정한 결과가 아닙니다.

## 카메라 입력 선택: HTTP MJPEG / RTSP

별도 카메라 config 파일 없이 `app/main.py`가 `.env`에서 로드한 환경변수를 읽습니다.
`.env`의 `CAMERA_SOURCE_TYPE`을 바꾼 뒤 서버를 재시작합니다. 미지정 시 `http`입니다.
선택한 소스의 URL만 필수이고, 잘못된 타입/URL/RTSP transport는 모델 생성 전에
startup 오류로 보고합니다. 실제 `.env`와 인증정보는 커밋하지 않습니다.

ESP32-CAM 입력(기존 동작):

```ini
CAMERA_SOURCE_TYPE=http
ESP32_STREAM_URL=http://YOUR_ESP32_CAM_IP:81/stream
```

아이폰 OctoStream 입력:

```ini
CAMERA_SOURCE_TYPE=rtsp
RTSP_STREAM_URL=rtsp://YOUR_IPHONE_IP:PORT/PATH
RTSP_TRANSPORT=tcp
```

1. 아이폰과 AI Server를 서로 통신 가능한 같은 네트워크에 연결합니다.
2. OctoStream에서 RTSP 송출을 시작하고 **앱에 표시된 실제 URL 전체**를 복사합니다.
   포트/경로는 추정하지 않습니다. 가능하면 H.264를 선택합니다.
3. 위 URL을 `.env`에 설정하고 기존 GPU 실행 명령을 유지하여 재시작합니다.

   ```bash
   __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```

4. `http://AI_SERVER_IP:8000/stream`에서 기존 스켈레톤 MJPEG 영상을 확인합니다.
   PyQt/AIFit은 이 주소를 그대로 사용하며 RTSP 수신 기능이 필요하지 않습니다.
5. 송출을 중단/재개하여 `frame_read_failed` → 재연결 → `connected`와 영상 복구를
   확인합니다. 서버 종료 후 `worker_stopped`가 출력되는지도 확인합니다.

입력별 클래스를 추가하지 않고 기존 CameraStream 수신 스레드와 최신 프레임 슬롯을
공유합니다. 기존 HTTP 수신도 OpenCV FFmpeg를 사용했으며 별도 JPEG 파서는 없었습니다.
HTTP의 VideoCapture 호출 인수, BGR 프레임, frame_id/Condition, 프레임 복사와
단일 추론/다중 구독자 공유 방식은 유지합니다. 연결 실패 시 수신 워커에서 release 후
1초간 Event로 대기하고 재연결합니다. 재연결 중에는 최신 슬롯을 비우며 frame_id는
증가를 이어갑니다. 종료 시 동일 워커에서 release하고 join합니다. 기존 open/read
타임아웃은 각각 5초, join 제한은 7초이며 백엔드가 타임아웃을 지키지 않으면
`worker_stop_delayed`를 기록합니다.

현재 가상환경 OpenCV 5.0.0 빌드에서 FFmpeg YES, avcodec/avformat YES를 확인했습니다.
새 패키지를 설치하지 않았으며 다른 실행 환경은 아래 명령으로 확인합니다.

```bash
python -c 'import cv2; print(cv2.getBuildInformation())'
```

RTSP TCP/UDP 선택에는 OpenCV의 `OPENCV_FFMPEG_CAPTURE_OPTIONS`를 연결 open 동안만
사용하고 예외 시에도 이전 값을 복원합니다. 기존 비전송 옵션은 유지합니다.
프로젝트 내 CameraStream의 HTTP/RTSP open은 공통 잠금으로 직렬화하므로 서로의
옵션을 읽지 않습니다. 환경변수는 프로세스 전역이므로 향후 별도 코드에서 직접
VideoCapture를 열 경우에도 같은 경로를 사용해야 합니다. 구현 근거는
[OpenCV FFmpeg 소스](https://github.com/opencv/opencv/blob/5.x/modules/videoio/src/cap_ffmpeg_impl.hpp)의
capture options 파싱입니다. CAP_PROP_BUFFERSIZE 같은 옵션의 지원을 가정하거나
버퍼 지연이 제거된다고 보장하지 않습니다.

카메라 애플리케이션 로그에는 URL과 예외 원문을 출력하지 않습니다. OpenCV/FFmpeg의
별도 verbose/debug 로깅은 애플리케이션 로거의 제어 밖이므로 인증 URL을 사용하는
운영 환경에서는 해당 원시 디버그 출력을 활성화하지 않습니다.

### FPS와 수신 지연 비교

수신 워커의 `[CameraMetrics]` JSON은 정상 프레임 수신 또는 재연결 시점에 최대
초당 1회 출력합니다. read/open이 블로킹 중인 동안에는 이 로그도 잠시 멈춥니다.
연결/실패/재연결/종료 이벤트는 즉시 별도 기록합니다. 기존 REST 스키마는 변경하지 않습니다.

- `camera_source_type`, `camera_connected`: 선택 입력과 연결 상태
- `camera_receive_fps`: 최근 5초간 수신·디코딩 완료 FPS (센서 FPS 아님)
- `inference_fps`: 기존 PipelineMetrics의 pose_inference_fps
- `frame_id`: 수신 프레임 번호
- `frame_read_duration_ms`: 마지막 capture.read 호출 소요 시간 (대기+디코딩 포함)
- `reconnect_count`: 최초 연결을 제외한 재연결 시도 횟수
- `last_frame_age_ms`: 마지막 정상 수신 시점 이후 경과 시간 (촬영부터의 지연 아님)

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/health
```

HTTP와 RTSP를 각각 같은 해상도/장면/네트워크에서 5분씩 실행하고 로그와 `/metrics`의
camera_capture_fps, pose_inference_fps, processing_fps, 지연, skipped_frames를 비교합니다.
`/stream` 클라이언트 1개와 여러 개도 비교합니다. RTSP가 더 빠르다고 가정하지 않습니다.
실제 아이폰 연결, 장시간 수신, FPS 개선 여부는 Mock 단위 테스트로 검증할 수 없습니다.

카메라/GPU 없이 실행하는 입력 테스트와 전체 회귀 테스트:

```bash
python -m unittest discover -s tests -p 'test_camera_sources.py' -v
python -m unittest discover -s tests -v
```

## 학습 기반 자세 평가

5종 모델의 실제 입력·출력, V2 피드백, `.env`, Live 확장 및 검증 한계는
[자세 평가 통합 문서](analysis/posture/README.md)를 참고하세요.
Reps/Hold 계산과 독립된 단일 슬롯 워커를 사용하며 기능 OFF에서는 기존 응답을 유지합니다.
