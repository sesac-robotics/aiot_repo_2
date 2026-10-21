
"""
FastAPI AI Server
"""

import os

from dotenv import load_dotenv
from datetime import datetime, timezone
# FastAPI의 시작과 종료 시점에 특정 작업을 수행할 수 있도록 도와주는 데코레이터
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.pose import PoseResponse
from app.session import SessionManager
from app.routers.internal_sessions import router as internal_sessions_router
from app.inference.yolo_pose import YOLOPoseEstimator
from app.inference.mediapipe_pose import MediapipePoseEstimator
from app.streaming.pose_stream import PoseStreamService

load_dotenv()

ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL")
POSE_MODEL = os.getenv("POSE_MODEL", "yolo")
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH", 
    "models/YOLO/yolo26m-pose.pt"
)
MEDIAPIPE_MODEL_PATH = os.getenv(
    "MEDIAPIPE_MODEL_PATH",
    "models/MediaPipe/pose_landmarker_full.task"
)
MEDIAPIPE_DELEGATE = os.getenv('MEDIAPIPE_DELEGATE', 'cpu').lower()


def create_pose_estimator():
    if POSE_MODEL == "yolo":
        print("[AI Service] Pose Model: YOLO26m")

        return YOLOPoseEstimator(
            model_path=YOLO_MODEL_PATH,
            device="cuda:0",
            conf=0.5
        )
    
    elif POSE_MODEL == "mediapipe":
        print("[AI Service] Pose Model: MediaPipe")

        return MediapipePoseEstimator(
            model_path=MEDIAPIPE_MODEL_PATH,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            min_visibility=0.35,
            delegate=MEDIAPIPE_DELEGATE,
        )

    else:
        raise ValueError(
            f"지원하지 않는 Pose 모델: {POSE_MODEL}"
        )

'''
서버의 시작과 종료를 관리

yield를 기준으로 yield 이전은 서버 시작시 실행
yield 이후는 종료시 실행되는 구조
'''
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 프로세스당 하나를 생성하여 API와 향후 추론 워커가 공유합니다.
    app.state.session_manager = SessionManager()
    app.state.internal_api_key = os.getenv("INTERNAL_API_KEY")
    # 모델 팩터리만 전달하여 생성/추론/종료가 같은 워커에서 실행되게 합니다.

    # 새로운 스트리밍 서비스 객체 생성
    pose_stream = PoseStreamService(
        stream_url=ESP32_STREAM_URL,
        pose_estimator=None,
        estimator_factory=create_pose_estimator,
        session_manager=app.state.session_manager,
        model_name=POSE_MODEL,
        # 로그 주기만 환경변수로 조절하며 FPS 측정 구간은 5초입니다.
        metrics_log_interval=float(os.getenv("METRICS_LOG_INTERVAL_SECONDS", "5"))
    )

    '''
    FastAPI 애플리케이션에서 공유할 객체를 보관하는 공간
    이렇게 저장하면 이후 API 요청에서 같은 객체에 접근 가능
    다만 모든 서버에서 공유되는 것이 아니라 하나의 FastAPI 프로세스 내부에서 공유되는 객체
    '''
    app.state.pose_stream = pose_stream

    try:
        pose_stream.start()
        print("[AI Service] 스트리밍 서비스 시작")

        yield
    # finally를 사용하는 이유는 종료 과정에서 예외가 발생해도 정리 작어븡ㄹ 수행하도록 하기 위해서
    finally:
        pose_stream.stop()
        print("[AI Service] 스트리밍 서비스 종료")


app = FastAPI(lifespan=lifespan)
app.include_router(internal_sessions_router)


@app.get("/stream")
def stream_video(request: Request):
    # 모든 클라이언트가 동일한 추론 결과를 공유
    pose_stream = request.app.state.pose_stream

    return StreamingResponse(
        pose_stream.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )

@app.get('/health')
def health_check(request: Request):
    pose_stream = request.app.state.pose_stream
    latest = pose_stream.get_latest_pose()

    if latest is None:
        return {
            'status': 'starting',
            'model': POSE_MODEL,
            'stream_ready': False
        }

    age_sec = (
        datetime.now(timezone.utc) - datetime.fromisoformat(latest['processed_at'])
    ).total_seconds()

    ready = age_sec < 3.0

    return {
        'status': 'ok' if ready else 'degraded',
        'model': POSE_MODEL,
        'stream_ready': ready,
        'frame_id': latest['frame_id'],
        'last_frame_age_sec': round(age_sec, 3)
    }

@app.get('/pose/latest', response_model=PoseResponse)
def get_latest_pose(request: Request):
    pose_stream = request.app.state.pose_stream

    result = pose_stream.get_latest_pose()

    if result is None:
        raise HTTPException(status_code=503, detail='아직 Pose 추론 결과가 없습니다.')

    # 오래된 프레임은 최신 결과로 반환하지 않음
    age_sec = (
        datetime.now(timezone.utc) - datetime.fromisoformat(result['processed_at'])
    ).total_seconds()

    if age_sec > 3.0:
        raise HTTPException(status_code=503, detail='Pose 스트림이 갱신되지 않고 있습니다.')

    return result

@app.get('/metrics')
def get_metrics(request: Request):
    # 기존 서비스의 스냅샷만 반환하므로 요청마다 추론하지 않습니다.
    return request.app.state.pose_stream.get_metrics()
