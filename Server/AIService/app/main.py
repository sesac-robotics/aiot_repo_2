'''
FastAPI 애플리케이션을 생성하는 진입점
'''
import os
import cv2
import time
import torch

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.camera.stream import CameraStream
from app.inference.yolo_pose import YOLOPoseEstimator
from app.inference.mediapipe_pose import MediapipePoseEstimator
from app.streaming.output import encode_jpeg

from dotenv import load_dotenv
from collections import deque

app = FastAPI()
load_dotenv()  # .env 파일에서 환경 변수 로드

ESP32_STREAM_URL = os.getenv('ESP32_STREAM_URL')
POSE_MODEL = os.getenv('POSE_MODEL', 'yolo')  # 기본값은 'yolo'로 설정
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/YOLO/yolo26m-pose.pt')
MEDIAPIPE_MODEL_PATH = os.getenv('MEDIAPIPE_MODEL_PATH', 'models/MediaPipe/pose_landmarker_full.task')

def create_pose_estimator():
    if POSE_MODEL == 'yolo':
        print("[AI Service] Pose Model: YOLO26m")

        return YOLOPoseEstimator(model_path=YOLO_MODEL_PATH, device='cuda:0', conf=0.5)
    elif POSE_MODEL == 'mediapipe':
        print("[AI Service] Pose Model: MediaPipe Pose Landmarker Full")

        return MediapipePoseEstimator(
            model_path=MEDIAPIPE_MODEL_PATH,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        raise ValueError(f"지원하지 않는 Pose 모델입니다: {POSE_MODEL}")

pose_estimator = create_pose_estimator()

def generate_frames():
    camera = CameraStream(ESP32_STREAM_URL)

    # 최근 30프레임 기준으로 시간 측정
    frame_times = deque(maxlen=31)
    metrics = deque(maxlen=30)
    frame_count = 0

    def sync_gpu():
        # GPU 비동기 실행 시간을 정확하게 측정하기 위한 동기화
        if POSE_MODEL == 'yolo' and torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        camera.connect()
        print(f'[Camera] 연결 성공')

        while True:
            total_start = time.perf_counter()

            # 1. ESP32-CAM에서 한 프레임 읽기
            frame = camera.read()

            if frame is None:
                continue

            read_ms = (time.perf_counter() - total_start) * 1000

            # 2. YOLO Pose 추론
            sync_gpu()
            infer_start = time.perf_counter()

            annotated_frame, result = pose_estimator.predict(frame)

            sync_gpu()
            infer_ms = (time.perf_counter() - infer_start) * 1000

            # 3. 최근 처리 FPS
            if len(frame_times) >= 2:
                duration = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / duration if duration > 0 else 0
            else:
                fps = 0

            # 4. 영상 위에 측정값 표시
            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.1f} | {POSE_MODEL}: {infer_ms:.1f} ms',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 5. JPEG 인코딩
            # OpenCV frame -> JPEG bytes
            encode_start = time.perf_counter()
            jpeg_bytes = encode_jpeg(annotated_frame)
            encode_ms = (time.perf_counter() - encode_start) * 1000

            if jpeg_bytes is None:
                continue

            # 6. 서버 처리시간
            total_ms = (time.perf_counter() - total_start) * 1000

            frame_count += 1
            frame_times.append(time.perf_counter())

            metrics.append({
                'read': read_ms,
                'infer': infer_ms,
                'encode': encode_ms,
                'total': total_ms,
            })

            # 7. 30프레임마다 통계 출력
            if frame_count % 30 == 0:
                avg = lambda key: sum(m[key] for m in metrics) / len(metrics)

                if len(frame_times) >= 2:
                    duration = frame_times[-1] - frame_times[0]
                    fps = (len(frame_times) - 1) / duration

                print(
                    f'[Performance] '
                    f'FPS: {fps:.2f} | '
                    f'Read: {avg('read'):.1f}ms | '
                    f'Inference: {avg('infer'):.1f}ms | '
                    f'Encode: {avg('encode'):.1f}ms | '
                    f'Server: {avg('total'):.1f}ms'
                )

            # 8. MJPEG 스트리밍
            # MJPEG 형식으로 한 프레임 전송
            # return 대신 yield를 사용하면 끊임없이 프레임을 전송할 수 있음
            # --frame은 MJPEG 형식에서 각 프레임을 구분하는 boundary 역할
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + jpeg_bytes
                + b'\r\n'
            )
    finally:
        camera.release()
        print(f'[Camera] 연결 종료')

@app.get('/stream')
def stream_video():
    # AIFit Server가 get을 요청하면 generate_frames()를 실행
    # StreamingResponse를 사용해 FastAPI에게 계속 생성되는 streaming response라고 알려줌
    return StreamingResponse(
        generate_frames(),
        # HTTP 응답 안에는 여러 데이터 조각이 계속 들어오고, 각 데이터는 frame이라는 boundary로 구분된다.
        media_type='multipart/x-mixed-replace; boundary=frame'
    )