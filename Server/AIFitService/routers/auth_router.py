from fastapi import APIRouter, HTTPException
from models.auth_models import RegisterRequest, LoginRequest
from Server.AIFitService.services.auth_service import AuthError, register_user, login_user

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
