'''
FastAPI 애플리케이션을 생성하는 진입점
'''

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.camera.stream import CameraStream
from app.inference.yolo_pose import YOLOPoseEstimator
from app.inference.mediapipe_pose import MediapipePoseEstimator
from app.streaming.output import encode_jpeg

app = FastAPI()

ESP32_STREAM_URL = 'http://'

# POSE_MODEL = 'yolo'
POSE_MODEL = 'mediapipe'

YOLO_MODEL_PATH = 'models/YOLO/yolo26m-pose.pt'
MEDIAPIPE_MODEL_PATH = 'models/MediaPipe/pose_landmarker_full.task'

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

    try:
        camera.connect()

        while True:
            # ESP32-CAM에서 한 프레임 읽기
            frame = camera.read()

            if frame is None:
                continue

            # YOLO Pose 추론
            annotated_frame, result = pose_estimator.predict(frame)

            # OpenCV frame -> JPEG bytes
            jpeg_bytes = encode_jpeg(annotated_frame)

            if jpeg_bytes is None:
                continue

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

@app.get('/stream')
def stream_video():
    # AIFit Server가 get을 요청하면 generate_frames()를 실행
    # StreamingResponse를 사용해 FastAPI에게 계속 생성되는 streaming response라고 알려줌
    return StreamingResponse(
        generate_frames(),
        # HTTP 응답 안에는 여러 데이터 조각이 계속 들어오고, 각 데이터는 frame이라는 boundary로 구분된다.
        media_type='multipart/x-mixed-replace; boundary=frame'
    )