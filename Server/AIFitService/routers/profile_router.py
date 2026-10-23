import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from models.auth_models import (
    DeleteAccountRequest,
    PasswordChangeRequest,
    ProfileUpdateRequest,
)
from repositories import user_repository
from services.auth_service import hash_password, verify_password
from services.token_service import get_current_user

router = APIRouter(prefix="/api", tags=["Authenticated"])

@router.get("/me")
def get_my_profile(current_user: dict = Depends(get_current_user)):
    """Protected endpoint: only a valid Bearer JWT can access this route."""
    return {"success": True, "user": current_user}


@router.put("/me")
def update_my_profile(
    payload: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    email = str(payload.email_address).strip().lower()
    nickname = payload.nickname.strip()
    if not nickname:
        raise HTTPException(status_code=422, detail="닉네임을 입력해 주세요.")
    if user_repository.email_exists_for_other_user(email, user_id):
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일 주소입니다.")

    try:
        updated = user_repository.update_profile(
            user_id, email, nickname, payload.gender,
            payload.age, payload.height, payload.weight,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일 주소입니다.")
    if not updated:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    user = user_repository.find_public_by_id(user_id)
    return {"success": True, "message": "프로필이 저장되었습니다.", "user": dict(user)}


@router.post("/me/change-password")
def change_my_password(
    payload: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    saved_hash = user_repository.get_password_hash(user_id)
    if saved_hash is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    if not verify_password(payload.current_password, saved_hash):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 올바르지 않습니다.")
    user_repository.update_password(user_id, hash_password(payload.new_password))
    return {"success": True, "message": "비밀번호가 변경되었습니다."}


@router.post("/me/delete-account")
def delete_my_account(
    payload: DeleteAccountRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    saved_hash = user_repository.get_password_hash(user_id)
    if saved_hash is None:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    if not verify_password(payload.password, saved_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")
    if not user_repository.delete_user_and_related_data(user_id):
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"success": True, "message": "계정과 사용자 관련 데이터가 삭제되었습니다."}
