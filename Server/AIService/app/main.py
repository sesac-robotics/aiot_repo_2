"""AI 서버의 설정·모델 팩터리·세션 관리자·추론 스트림을 조립하는 진입점입니다. 실제 카메라·모델 작업은 앱 수명 시작 이후에만 실행합니다.

FastAPI AI Server"""

import os

# FastAPI의 시작과 종료 시점에 특정 작업을 수행할 수 있도록 도와주는 데코레이터
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.debug_logging import debug_log_files
from app.routers.internal_sessions import router as internal_sessions_router
from app.schemas.pose import PoseResponse
from app.session import SessionManager
from app.streaming.pose_stream import PoseStreamService

load_dotenv()

POSE_MODEL = os.getenv("POSE_MODEL", "yolo")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/YOLO/yolo26m-pose.pt")
MEDIAPIPE_MODEL_PATH = os.getenv(
    "MEDIAPIPE_MODEL_PATH", "models/MediaPipe/pose_landmarker_full.task"
)
MEDIAPIPE_DELEGATE = os.getenv("MEDIAPIPE_DELEGATE", "cpu").lower()


def read_camera_settings(environ=None):
    """.env에서 로드한 환경변수로 입력을 선택하고 선택된 URL만 검증합니다."""
    # lifespan에서 호출하여 설정 오류를 모델 생성보다 먼저 발견합니다.
    env = os.environ if environ is None else environ
    source = env.get("CAMERA_SOURCE_TYPE", "http").strip().lower()
    if source not in ("http", "rtsp"):
        raise ValueError("CAMERA_SOURCE_TYPE은 http 또는 rtsp여야 합니다.")
    key = "ESP32_STREAM_URL" if source == "http" else "RTSP_STREAM_URL"
    url = env.get(key, "").strip()
    if not url:
        raise ValueError(f"{source} 입력에 필요한 {key}가 없습니다.")
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme
            in (("http", "https") if source == "http" else ("rtsp", "rtsps"))
            and parsed.hostname
        )
        parsed.port  # 잘못된 포트도 URL 원문 없이 보고합니다.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{key}의 URL 형식이 올바르지 않습니다.")
    transport = (
        env.get("RTSP_TRANSPORT", "tcp").strip().lower()
        if source == "rtsp"
        else "tcp"
    )
    if transport not in ("tcp", "udp"):
        raise ValueError("RTSP_TRANSPORT는 tcp 또는 udp여야 합니다.")
    return source, url, transport


def create_pose_estimator():
    if POSE_MODEL == "yolo":
        from app.inference.yolo_pose import YOLOPoseEstimator

        print("[AI Service] Pose Model: YOLO26m")

        return YOLOPoseEstimator(
            model_path=YOLO_MODEL_PATH, device="cuda:0", conf=0.5
        )

    elif POSE_MODEL == "mediapipe":
        from app.inference.mediapipe_pose import MediapipePoseEstimator

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
        raise ValueError(f"지원하지 않는 Pose 모델: {POSE_MODEL}")


"""
서버의 시작과 종료를 관리

yield를 기준으로 yield 이전은 서버 시작시 실행
yield 이후는 종료시 실행되는 구조
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 생성/종료 과정의 디버그 로그까지 파일에 기록하고 일반 서버 로그는 유지합니다.
    with debug_log_files():
        async with _service_lifespan(app):
            yield


@asynccontextmanager
async def _service_lifespan(app: FastAPI):
    # 선택한 카메라 설정만 검증하며 잘못된 설정이면 GPU 생성 전에 실패합니다.
    camera_source_type, stream_url, rtsp_transport = read_camera_settings()
    # 프로세스당 하나를 생성하여 API와 향후 추론 워커가 공유합니다.
    app.state.session_manager = SessionManager()
    app.state.internal_api_key = os.getenv("INTERNAL_API_KEY")
    # OFF일 때는 자세 모델/torch 모듈도 추가 import/생성하지 않습니다.
    posture_service = None
    # 명시적 off는 기존 ENABLED=true보다 우선합니다.
    evaluation_mode = os.getenv("POSTURE_EVALUATION_MODE")
    if (
        evaluation_mode is not None and evaluation_mode.strip().lower() != "off"
    ) or (
        evaluation_mode is None
        and os.getenv("POSTURE_MODEL_ENABLED", "false").strip().lower()
        not in ("false", "0")
    ):
        from app.analysis.posture.service import PostureService

        posture_service = PostureService.from_env(app.state.session_manager)
    app.state.posture_service = posture_service
    # 모델 팩터리만 전달하여 생성/추론/종료가 같은 워커에서 실행되게 합니다.

    # 새로운 스트리밍 서비스 객체 생성
    pose_stream = PoseStreamService(
        stream_url=stream_url,
        camera_source_type=camera_source_type,
        rtsp_transport=rtsp_transport,
        posture_service=posture_service,
        pose_estimator=None,
        estimator_factory=create_pose_estimator,
        session_manager=app.state.session_manager,
        model_name=POSE_MODEL,
        # 로그 주기만 환경변수로 조절하며 FPS 측정 구간은 5초입니다.
        metrics_log_interval=float(
            os.getenv("METRICS_LOG_INTERVAL_SECONDS", "5")
        ),
    )

    """
    FastAPI 애플리케이션에서 공유할 객체를 보관하는 공간
    이렇게 저장하면 이후 API 요청에서 같은 객체에 접근 가능
    다만 모든 서버에서 공유되는 것이 아니라 하나의 FastAPI 프로세스 내부에서 공유되는 객체
    """
    app.state.pose_stream = pose_stream

    try:
        if posture_service is not None:
            posture_service.start()
        pose_stream.start()
        print("[AI Service] 스트리밍 서비스 시작")

        yield
    # finally를 사용하는 이유는 종료 과정에서 예외가 발생해도 정리 작어븡ㄹ 수행하도록 하기 위해서
    finally:
        try:
            if posture_service is not None:
                posture_service.stop()
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
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
def health_check(request: Request):
    pose_stream = request.app.state.pose_stream
    latest = pose_stream.get_latest_pose()

    if latest is None:
        return {
            "status": "starting",
            "model": POSE_MODEL,
            "stream_ready": False,
        }

    age_sec = (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(latest["processed_at"])
    ).total_seconds()

    ready = age_sec < 3.0

    return {
        "status": "ok" if ready else "degraded",
        "model": POSE_MODEL,
        "stream_ready": ready,
        "frame_id": latest["frame_id"],
        "last_frame_age_sec": round(age_sec, 3),
    }


@app.get("/pose/latest", response_model=PoseResponse)
def get_latest_pose(request: Request):
    pose_stream = request.app.state.pose_stream

    result = pose_stream.get_latest_pose()

    if result is None:
        raise HTTPException(
            status_code=503, detail="아직 Pose 추론 결과가 없습니다."
        )

    # 오래된 프레임은 최신 결과로 반환하지 않음
    age_sec = (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(result["processed_at"])
    ).total_seconds()

    if age_sec > 3.0:
        raise HTTPException(
            status_code=503, detail="Pose 스트림이 갱신되지 않고 있습니다."
        )

    return result


@app.get("/metrics")
def get_metrics(request: Request):
    # 기존 서비스의 스냅샷만 반환하므로 요청마다 추론하지 않습니다.
    return request.app.state.pose_stream.get_metrics()
