"""사용자 등록과 로그인을 처리합니다. PBKDF2 비밀번호 해시 검증 후 JWT를 발급하고 SQL은 사용자 저장소에 위임합니다."""

import hashlib
import hmac
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from Server.AIFitService.config.settings import (
    JWT_ALGORITHM,
    JWT_SECRET,
    TOKEN_EXPIRE_MINUTES,
)
from Server.AIFitService.models.auth_models import LoginRequest, RegisterRequest
from Server.AIFitService.repositories import user_repository

logger = logging.getLogger(__name__)


class AuthError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def hash_password(password: str) -> str:
    """무작위 16바이트 salt와 PBKDF2-SHA256으로 비밀번호를 해시합니다.

    Args:
        password: 저장하지 않을 원문 비밀번호입니다.

    Returns:
        알고리즘·반복 횟수·salt·digest를 달러 기호로 구분한 저장 문자열입니다.
        암호화와 달리 원문으로 복원하지 않으며 로그인 때 다시 계산해 비교합니다.
    """
    salt = secrets.token_bytes(16)
    rounds = 310_000
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, rounds
    )
    return f"pbkdf2_sha256${rounds}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """저장된 해시 형식을 읽고 상수 시간 비교로 비밀번호 일치를 검사합니다.

    Returns:
        일치하면 True이며 형식이 잘못되었거나 다른 비밀번호이면 False입니다.
    """
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(rounds),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def register_user(data: RegisterRequest) -> dict[str, Any]:
    """아이디·이메일 중복을 확인한 뒤 비밀번호 해시와 사용자 정보를 저장합니다.

    Raises:
        AuthError: 빈 필드(422) 또는 중복 아이디·이메일(409)입니다.
    """
    username = data.username.strip()
    email = str(data.email_address).strip().lower()
    nickname = data.nickname.strip()
    if not username or not nickname:
        raise AuthError(422, "username과 nickname을 입력해 주세요.")
    if user_repository.username_exists(username):
        raise AuthError(409, "이미 사용 중인 username입니다.")
    if user_repository.email_exists(email):
        raise AuthError(409, "이미 사용 중인 이메일 주소입니다.")
    try:
        user_id = user_repository.create_user(
            username,
            hash_password(data.password),
            email,
            nickname,
            data.gender,
            data.age,
            data.height,
            data.weight,
        )
    except sqlite3.IntegrityError:
        raise AuthError(409, "아이디 또는 이메일이 이미 등록되어 있습니다.")
    logger.info("[AUTH] Registration successful")
    logger.info("[DB] UserInfo 저장 완료 (비밀번호는 해시 형태)")
    return {
        "success": True,
        "message": "회원가입이 완료되었습니다.",
        "user_id": user_id,
        "username": username,
    }


def login_user(data: LoginRequest) -> dict[str, Any]:
    """비밀번호 확인 후 만료 시간이 있는 JWT와 공개 사용자 정보를 반환합니다.

    Raises:
        AuthError: 사용자 부재와 비밀번호 불일치를 동일한 401 오류로 처리합니다.
    """
    username = data.username.strip()
    user = user_repository.find_by_username(username)
    if user is None or not verify_password(
        data.password, user["password_hash"]
    ):
        logger.info("[AUTH] Login failed")
        raise AuthError(401, "아이디 또는 비밀번호가 올바르지 않습니다.")

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {
            "sub": str(user["user_id"]),
            "username": user["username"],
            "iat": now,
            "exp": expires,
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )

    logger.info("[AUTH] Login successful")
    logger.info("[AUTH] Access token issued")
    return {
        "success": True,
        "message": "로그인에 성공했습니다.",
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "nickname": user["nickname"],
        },
    }
