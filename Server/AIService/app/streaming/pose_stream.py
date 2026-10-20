
import cv2
import time
import torch
import threading

from collections import deque

from app.camera.stream import CameraStream
from app.streaming.output import encode_jpeg


class PoseStreamService:
    def __init__(self, stream_url, pose_estimator, model_name):
        self.camera = CameraStream(stream_url)
        self.pose_estimator = pose_estimator
        self.model_name = model_name

        # 모든 클라이언트가 공유할 최신 추론 영상
        self.latest_jpeg = None

        # 추론이 완료된 프레임 번호
        self.frame_id = 0

        # 클라이언트 간 최신 프레임 공유 및 동기화
        self.condition = threading.Condition()

        # 추론 워커 종료 관리
        self.stop_event = threading.Event()
        self.worker = None

    def start(self):
        # 추론 워커 중복 실행 방지
        if self.worker is not None and self.worker.is_alive():
            return

        self.stop_event.clear()

        # 이전에 처리했던 영상 결과 초기화
        with self.condition:
            self.latest_jpeg = None
            self.frame_id = 0

        # ESP32-CAM 수신 스레드 실행
        self.camera.connect()

        # 단일 Pose 추론 워커 생성
        self.worker = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="pose-inference-worker"
        )

        self.worker.start()

        print("[AI Service] Pose 추론 워커 시작")

    # CUDA 연산 동기화
    def _sync_inferecnce(self):
        # CUDA 비동기 연산 완료까지 측정
        if (
            self.model_name == "yolo"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()
        elif self.model_name == 'mediapipe':
            '''
            detect_for_video()는 동기식 함수
            함수가 반환되면 해당 프레임의 추론이 완료됨
            MediaPipe GPU delegate는 PyTorch CUDA와 별도이기 때문에
            torch.cuda.synchronize()를 호출하지 않음
            '''
            pass

    def _inference_loop(self):
        # 최근 30프레임 기준 성능 측정
        frame_times = deque(maxlen=31)
        metrics = deque(maxlen=30)
        frame_count = 0

        while not self.stop_event.is_set():
            try:
                total_start = time.perf_counter()

                # 1. ESP32-CAM 최신 프레임 가져오기
                frame = self.camera.read()

                if frame is None:
                    continue

                read_ms = (
                    time.perf_counter() - total_start
                ) * 1000

                # 2. Pose Estimation
                self._sync_inferecnce()
                infer_start = time.perf_counter()

                annotated_frame, result = (
                    self.pose_estimator.predict(frame)
                )

                self._sync_inferecnce()
                infer_ms = (
                    time.perf_counter() - infer_start
                ) * 1000

                # 3. 최근 처리 FPS 계산
                frame_times.append(time.perf_counter())

                if len(frame_times) >= 2:
                    duration = (
                        frame_times[-1] - frame_times[0]
                    )
                    fps = (
                        (len(frame_times) - 1) / duration
                        if duration > 0 else 0.0
                    )
                else:
                    fps = 0.0

                # 4. 영상에 성능 정보 표시
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | "
                    f"{self.model_name}: {infer_ms:.1f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # 5. JPEG 인코딩
                encode_start = time.perf_counter()

                jpeg_bytes = encode_jpeg(annotated_frame)

                encode_ms = (
                    time.perf_counter() - encode_start
                ) * 1000

                if jpeg_bytes is None:
                    continue

                total_ms = (
                    time.perf_counter() - total_start
                ) * 1000

                # 6. 최신 추론 결과 공유
                with self.condition:
                    self.latest_jpeg = jpeg_bytes
                    self.frame_id += 1

                    # 대기 중인 모든 클라이언트에 알림
                    self.condition.notify_all()

                # 7. 성능 측정값 기록
                frame_count += 1

                metrics.append({
                    "read": read_ms,
                    "infer": infer_ms,
                    "encode": encode_ms,
                    "total": total_ms
                })

                # 8. 30프레임마다 성능 로그
                if frame_count % 30 == 0:
                    def avg(key):
                        return sum(
                            m[key] for m in metrics
                        ) / len(metrics)

                    print(
                        f"[Performance] "
                        f"FPS: {fps:.2f} | "
                        f"Read: {avg('read'):.1f}ms | "
                        f"Inference: {avg('infer'):.1f}ms | "
                        f"Encode: {avg('encode'):.1f}ms | "
                        f"Server: {avg('total'):.1f}ms"
                    )

            except Exception as e:
                print(f"[AI Service] 추론 오류: {e}")
                self.stop_event.wait(0.2)

        with self.condition:
            self.condition.notify_all()

        print("[AI Service] 추론 워커 종료")

    # 여러 클라이언트에 영상 전송
    def generate_frames(self):
        # 각 클라이언트의 마지막 전송 프레임 번호
        last_sent_id = 0

        while not self.stop_event.is_set():
            with self.condition:
                # 새로운 추론 결과가 생성될 때까지 대기
                ready = self.condition.wait_for(
                    lambda: (
                        self.frame_id > last_sent_id
                        or self.stop_event.is_set()
                    ),
                    timeout=1.0
                )

                if self.stop_event.is_set():
                    break

                if not ready or self.latest_jpeg is None:
                    continue

                # 최신 JPEG를 참조하고 번호 갱신
                jpeg_bytes = self.latest_jpeg
                last_sent_id = self.frame_id

            # Lock 밖에서 클라이언트에 영상 전송
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + (
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode("ascii")
                + jpeg_bytes
                + b"\r\n"
            )

    def stop(self):
        # 추론 워커 종료 요청
        self.stop_event.set()

        # 프레임을 기다리는 클라이언트 깨우기
        with self.condition:
            self.condition.notify_all()

        # 카메라 수신 스레드 종료
        self.camera.release()

        # 추론 워커 종료 대기
        if self.worker is not None:
            self.worker.join(timeout=3.0)

            if self.worker.is_alive():
                print("[AI Service] 추론 워커 종료 지연")
            else:
                self.worker = None

        if self.worker is None or not self.worker.is_alive():
            close_fn = getattr(self.pose_estimator, 'close', None)

            if callable(close_fn):
                close_fn()

        print("[AI Service] 종료 요청 완료")
