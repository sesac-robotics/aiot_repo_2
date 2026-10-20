
"""
FastAPI AI Server
"""

import os
# FastAPI의 시작과 종료 시점에 특정 작업을 수행할 수 있도록 도와주는 데코레이터
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

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
    # AI 모델 로드 (서버 시작 시 1회)
    # 클라이언트가 접속할 때마다 모델을 다시 로딩하지 않음
    pose_estimator = create_pose_estimator()

    # 새로운 스트리밍 서비스 객체 생성
    pose_stream = PoseStreamService(
        stream_url=ESP32_STREAM_URL,
        pose_estimator=pose_estimator,
        model_name=POSE_MODEL
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
