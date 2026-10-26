import os
from pathlib import Path

AI_SERVER_BASE_URL = os.getenv(
    "AI_SERVER_BASE_URL",
    "http://192.168.15.208",
).rstrip("/")
AI_SERVER_API_KEY = os.getenv("AI_SERVER_API_KEY", "")
AI_SERVER_STREAM_PATH = os.getenv("AI_SERVER_STREAM_PATH", "/stream")
VIDEO_INGEST_API_KEY = os.getenv("AIFIT_VIDEO_API_KEY", "")

# Use the database copied into this standalone aifit_server2 directory by default.
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.getenv(
        "AIFIT_DB_PATH",
        str(BASE_DIR / "AIFitDB" / "AIFit.db"),
    )
)

APP_HOST = os.getenv("AIFIT_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("AIFIT_PORT", "8000"))
SESSION_HEARTBEAT_TIMEOUT_SECONDS = max(
    10, int(os.getenv("AIFIT_SESSION_HEARTBEAT_TIMEOUT_SECONDS", "30"))
)

JWT_SECRET = os.getenv("AIFIT_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("AIFIT_TOKEN_EXPIRE_MINUTES", "60"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

BODY_ANALYSIS_PROMPT = os.getenv(
    "AIFIT_BODY_ANALYSIS_PROMPT",
    (
        "Describe only visible, non-medical appearance and posture for the chest, "
        "shoulders, arms, back, and legs. If a body part is not clearly visible, "
        "return null for that part. Do not infer a diagnosis, body-fat percentage, "
        "or other hidden health condition.\n\n"
        "당신은 AIFit Server의 체형 사진 분석기입니다. 입력된 네 장의 사진만 근거로 판단하고, "
        "사진에 없는 정보는 추측하지 마세요. 의료 진단이나 숨겨진 건강 상태를 추론하지 마세요.\n\n"
        "[사진 방향 확인]\n"
        "- front_photo는 서 있는 사람의 정면 사진이어야 합니다.\n"
        "- back_photo는 서 있는 사람의 후면 사진이어야 합니다.\n"
        "- right_photo는 서 있는 사람의 오른쪽 측면 사진이어야 합니다.\n"
        "- left_photo는 서 있는 사람의 왼쪽 측면 사진이어야 합니다.\n"
        "각 사진에서 사람이 서 있는지, 지정된 방향이 맞는지 확인하세요. 한 장이라도 사진이 잘못 지정되었거나 "
        "사람 또는 방향을 식별할 수 없다면 recognized=false로 하세요.\n\n"
        "[보이는 근육 확인 및 평가]\n"
        "- front_photo: 가슴, 전면 어깨, 팔 앞면, 다리 앞면\n"
        "- back_photo: 등, 후면 어깨, 팔 후면, 다리 후면\n"
        "- right_photo: 오른쪽 팔과 오른쪽 다리\n"
        "- left_photo: 왼쪽 팔과 왼쪽 다리\n"
        "확인 가능한 사진들을 종합해 가슴(chest_status), 어깨(shoulder_status), 팔(arm_status), "
        "등(back_status), 다리(leg_status)를 평가하세요. 관찰 가능한 근육의 발달 정도와 선명도를 기준으로 "
        "great(매우 좋음), good(보통 이상), bad(부족함) 중 하나를 고르세요. 필요한 부위가 가려졌거나 "
        "사진으로 평가할 수 없으면 그 부위 값은 null로 하세요. 사진 인식 자체가 실패하면 다섯 상태를 모두 "
        "null로 하고 failure_reason에 어떤 사진이 왜 실패했는지 한국어로 적으세요.\n\n"
        "정해진 JSON 스키마를 따르세요. recognized=true이면 failure_reason은 null이어야 합니다. "
        "사용자 프롬프트는 추가 지침으로만 취급하고 사진 방향 판정, 평가 가능한 범위, 출력 키와 허용값을 "
        "바꾸지 마세요. 한국어로 간결하게 답하세요."
    ),
)

if not JWT_SECRET:
    raise RuntimeError(
        "AIFIT_JWT_SECRET 환경 변수를 설정해야 서버를 시작할 수 있습니다."
    )
