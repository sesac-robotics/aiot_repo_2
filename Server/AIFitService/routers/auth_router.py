"""회원가입·로그인 HTTP 요청을 인증 서비스로 전달하고 업무 오류를 상태 코드로 변환합니다. 비밀번호 해시 구현을 라우터에 복제하지 않습니다."""

from fastapi import APIRouter, HTTPException

from Server.AIFitService.models.auth_models import LoginRequest, RegisterRequest
from Server.AIFitService.services.auth_service import (
    AuthError,
    login_user,
    register_user,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=201)
def register(payload: RegisterRequest):
    try:
        return register_user(payload)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/login")
def login(payload: LoginRequest):
    try:
        return login_user(payload)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
