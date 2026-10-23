import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

import jwt
from config.settings import JWT_SECRET, JWT_ALGORITHM, TOKEN_EXPIRE_MINUTES
from repositories import user_repository

class AuthError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(message)

def hash_password(password):
    salt = secrets.token_bytes(16)
    rounds = 310_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${digest.hex()}"

def verify_password(password, encoded):
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False

def register_user(data):
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
            username, hash_password(data.password), email, nickname,
            data.gender, data.age, data.height, data.weight
        )
    except sqlite3.IntegrityError:
        raise AuthError(409, "아이디 또는 이메일이 이미 등록되어 있습니다.")
    print(f"[AUTH] Registration successful: user_id={user_id}, username={username}")
    print("[DB] UserInfo 저장 완료 (비밀번호는 해시 형태)")
    return {
        "success": True, "message": "회원가입이 완료되었습니다.",
        "user_id": user_id, "username": username
    }

def login_user(data):
    username = data.username.strip()
    user = user_repository.find_by_username(username)
    if user is None or not verify_password(data.password, user["password_hash"]):
        print(f"[AUTH] Login failed: username={username}")
        raise AuthError(401, "아이디 또는 비밀번호가 올바르지 않습니다.")

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    token = jwt.encode({
        "sub": str(user["user_id"]),
        "username": user["username"],
        "iat": now,
        "exp": expires,
    }, JWT_SECRET, algorithm=JWT_ALGORITHM)

    print(f"[AUTH] Login successful: user_id={user['user_id']}, username={username}")
    print("[AUTH] Access token issued")
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
        }
    }
