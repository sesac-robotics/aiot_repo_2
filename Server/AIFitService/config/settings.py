"""환경 변수에서 백엔드 설정을 읽습니다.

모듈을 가져오는 시점에는 파일·DB·네트워크에 접근하지 않습니다. 비밀 키는
저장소에 두지 않고 실행 환경에서 공급하며, 서버 시작 시 validate_settings로
검증합니다. DB 경로는 실행 위치와 관계없이 서비스 디렉터리를 기준으로 합니다.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.getenv("AIFIT_DB_PATH", str(BASE_DIR.parent / "AIFitDB" / "aifit.db"))
)
APP_HOST = os.getenv("AIFIT_HOST", os.getenv("APP_HOST", "127.0.0.1"))
APP_PORT = int(os.getenv("AIFIT_PORT", os.getenv("APP_PORT", "8000")))
JWT_SECRET = os.getenv("AIFIT_JWT_SECRET", os.getenv("JWT_SECRET", ""))
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "AIFIT_TOKEN_EXPIRE_MINUTES", os.getenv("TOKEN_EXPIRE_MINUTES", "60")
    )
)
AI_SERVER_BASE_URL = os.getenv("AI_SERVER_BASE_URL", "").rstrip("/")
AI_SERVER_API_KEY = os.getenv("AI_SERVER_API_KEY", "")
AI_SERVER_STREAM_PATH = os.getenv("AI_SERVER_STREAM_PATH", "/stream")
VIDEO_INGEST_API_KEY = os.getenv(
    "AIFIT_VIDEO_API_KEY", os.getenv("VIDEO_INGEST_API_KEY", "")
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
BODY_ANALYSIS_PROMPT = os.getenv(
    "AIFIT_BODY_ANALYSIS_PROMPT",
    "제공된 사진에서 관찰 가능한 자세만 분석하십시오. 의료 진단이나 체성분 측정으로 "
    "단정하지 마십시오. 인물을 확인할 수 없으면 recognized=false로 반환하고 "
    "부위별 상태와 routine을 null로 반환하십시오.",
)


def validate_settings() -> None:
    """실행에 필수인 인증·포트 설정을 검증합니다.

    Raises:
        ValueError: JWT 키가 32바이트 미만이거나 포트·유효 시간이 범위를 벗어납니다.
    """
    if len(JWT_SECRET.encode("utf-8")) < 32:
        raise ValueError(
            "JWT_SECRET 환경 변수에 32바이트 이상의 비밀 키가 필요합니다."
        )
    if not 1 <= APP_PORT <= 65535:
        raise ValueError("APP_PORT는 1~65535여야 합니다.")
    if TOKEN_EXPIRE_MINUTES <= 0:
        raise ValueError("TOKEN_EXPIRE_MINUTES는 양수여야 합니다.")
