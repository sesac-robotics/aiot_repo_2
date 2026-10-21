
# userLogin.py

import sqlite3
import hashlib
import secrets
from pathlib import Path

from fastapi import APIRouter
from pydantic import (
    BaseModel,
    Field,
    EmailStr,
    ConfigDict,
    model_validator,
)

router = APIRouter()

# 이 파일과 같은 폴더에 DB 생성
DB_PATH = Path(__file__).resolve().parent / "users.db"


# =========================
# 데이터베이스 초기화
# =========================
def init_db():
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


# 비밀번호 해시 생성
def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000
    ).hex()


# =========================
# Request Schema
# =========================
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


# =========================
# 회원가입 API
# =========================
@router.post(
    "/api/register",
    response_model=RegisterResponse
)
def register(request: RegisterRequest):
    salt = secrets.token_bytes(16)
    password_hash = hash_password(request.password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (
                    userID, email, password_hash, password_salt,
                    nickname, gender, age, height, weight
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

        return RegisterResponse(
            success=True,
            message="Registration successful"
        )

    except sqlite3.IntegrityError:
        return RegisterResponse(
            success=False,
            message="User ID or email already exists"
        )


# =========================
# 로그인 API
# =========================
@router.post(
    "/api/login",
    response_model=LoginResponse
)
def login(request: LoginRequest):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT password_hash, password_salt
            FROM users
            WHERE userID = ?
        """, (request.userID,)).fetchone()

    if row is None:
        return LoginResponse(loginResult=False)

    saved_hash, saved_salt = row

    input_hash = hash_password(
        request.password,
        bytes.fromhex(saved_salt)
    )

    success = secrets.compare_digest(
        input_hash,
        saved_hash
    )

    return LoginResponse(loginResult=success)


# 서버 시작 시 DB와 테이블 준비
init_db()
