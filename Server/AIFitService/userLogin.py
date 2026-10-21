import sqlite3
import hashlib
import secrets
import logging

from pathlib import Path

from fastapi import APIRouter
from pydantic import (
    BaseModel,
    Field,
    EmailStr,
    ConfigDict,
    model_validator,
)

# ========================================
# 로깅 설정
# ========================================
logger = logging.getLogger("aifit.auth")

router = APIRouter()

# 이 파일과 같은 폴더에 DB 생성
DB_PATH = Path(__file__).resolve().parent / "users.db"


# ========================================
# 개인정보 마스킹 함수
# ========================================
def mask_value(value: str, visible: int = 2) -> str:
    """로그에 사용자 식별 정보를 일부만 표시한다."""
    if not value:
        return "***"

    if len(value) <= visible:
        return "*" * len(value)

    return value[:visible] + "***"


def mask_email(email: str) -> str:
    """이메일 주소의 아이디 부분을 마스킹한다."""
    local, separator, domain = email.partition("@")

    if not separator:
        return "***"

    return f"{mask_value(local, 1)}@{domain}"


# ========================================
# 데이터베이스 초기화
# ========================================
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    userID TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    height REAL NOT NULL,
                    weight REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

        logger.info(
            "[DB 초기화 완료] 경로=%s",
            DB_PATH
        )

    except sqlite3.Error:
        logger.exception(
            "[DB 초기화 실패] 데이터베이스 생성 또는 테이블 초기화 오류"
        )
        raise


# ========================================
# 비밀번호 해시 생성
# ========================================
def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000
    ).hex()


# ========================================
# Request / Response Schema
# ========================================
class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    userID: str = Field(min_length=1, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)
    confirmPassword: str = Field(min_length=8)
    nickname: str = Field(min_length=1, max_length=50)
    gender: str = Field(pattern="^(Male|Female)$")
    age: int = Field(ge=1, le=120)
    height: float = Field(gt=0, le=300)
    weight: float = Field(gt=0, le=500)

    @model_validator(mode="after")
    def check_passwords(self):
        if self.password != self.confirmPassword:
            raise ValueError("Passwords do not match")
        return self


class RegisterResponse(BaseModel):
    success: bool
    message: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    userID: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    loginResult: bool


# ========================================
# 회원가입 API
# POST /api/register
# ========================================
@router.post(
    "/api/register",
    response_model=RegisterResponse
)
def register(request: RegisterRequest):
    logger.info(
        "[통신 목적] GUI -> AIFit Server | 회원가입 요청"
    )

    logger.info(
        "[수신 데이터] endpoint=/api/register | "
        "fields=userID,email,password,confirmPassword,"
        "nickname,gender,age,height,weight | "
        "userID=%s | email=%s | nickname=%s | "
        "gender=%s | age=%s | height=%s | weight=%s",
        mask_value(request.userID),
        mask_email(str(request.email)),
        mask_value(request.nickname),
        request.gender,
        request.age,
        request.height,
        request.weight,
    )

    # 비밀번호 원문은 로그에 기록하지 않는다.
    salt = secrets.token_bytes(16)
    password_hash = hash_password(request.password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (
                    userID,
                    email,
                    password_hash,
                    password_salt,
                    nickname,
                    gender,
                    age,
                    height,
                    weight
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.userID,
                str(request.email),
                password_hash,
                salt.hex(),
                request.nickname,
                request.gender,
                request.age,
                request.height,
                request.weight,
            ))

        logger.info(
            "[DB 저장 완료] 회원가입 성공 | userID=%s",
            mask_value(request.userID),
        )

        result = RegisterResponse(
            success=True,
            message="Registration successful"
        )

        logger.info(
            "[응답 데이터] AIFit Server -> GUI | "
            "endpoint=/api/register | "
            "success=%s | message=%s",
            result.success,
            result.message,
        )

        return result

    except sqlite3.IntegrityError:
        logger.warning(
            "[DB 저장 실패] 사용자 ID 또는 이메일 중복 | "
            "userID=%s | email=%s",
            mask_value(request.userID),
            mask_email(str(request.email)),
        )

        result = RegisterResponse(
            success=False,
            message="User ID or email already exists"
        )

        logger.info(
            "[응답 데이터] AIFit Server -> GUI | "
            "endpoint=/api/register | "
            "success=%s | message=%s",
            result.success,
            result.message,
        )

        return result

    except sqlite3.Error:
        logger.exception(
            "[DB 오류] 회원가입 중 데이터베이스 오류 발생"
        )
        raise


# ========================================
# 로그인 API
# POST /api/login
# ========================================
@router.post(
    "/api/login",
    response_model=LoginResponse
)
def login(request: LoginRequest):
    logger.info(
        "[통신 목적] GUI -> AIFit Server | 로그인 요청"
    )

    logger.info(
        "[수신 데이터] endpoint=/api/login | "
        "fields=userID,password | userID=%s",
        mask_value(request.userID),
    )

    try:
        # DB에서 해당 사용자 ID의 해시와 솔트를 조회한다.
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT password_hash, password_salt
                FROM users
                WHERE userID = ?
            """, (request.userID,)).fetchone()

        if row is None:
            logger.warning(
                "[DB 조회 결과] 등록된 사용자 없음 | userID=%s",
                mask_value(request.userID),
            )

            result = LoginResponse(
                loginResult=False
            )

        else:
            saved_hash, saved_salt = row

            logger.info(
                "[DB 조회 완료] 사용자 계정 확인 | userID=%s",
                mask_value(request.userID),
            )

            # 입력 비밀번호를 저장된 솔트로 해시화한다.
            input_hash = hash_password(
                request.password,
                bytes.fromhex(saved_salt)
            )

            # 저장된 해시와 입력 비밀번호의 해시를 비교한다.
            success = secrets.compare_digest(
                input_hash,
                saved_hash
            )

            if success:
                logger.info(
                    "[비밀번호 검증] 성공 | userID=%s",
                    mask_value(request.userID),
                )
            else:
                logger.warning(
                    "[비밀번호 검증] 실패 | userID=%s",
                    mask_value(request.userID),
                )

            result = LoginResponse(
                loginResult=success
            )

        logger.info(
            "[응답 데이터] AIFit Server -> GUI | "
            "endpoint=/api/login | loginResult=%s",
            result.loginResult,
        )

        return result

    except sqlite3.Error:
        logger.exception(
            "[DB 오류] 로그인 중 데이터베이스 오류 발생"
        )
        raise

    except (ValueError, TypeError):
        logger.exception(
            "[데이터 처리 오류] 로그인 정보 처리 중 오류 발생"
        )
        raise


# ========================================
# 서버 시작 시 DB와 테이블 준비
# ========================================
init_db()
