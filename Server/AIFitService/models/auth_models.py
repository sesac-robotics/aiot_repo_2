"""회원가입·로그인·프로필 API의 입력과 출력 DTO를 정의합니다. Pydantic 검증은 전송 형식의 유효성 검사이며 인증과 영속성은 서비스가 담당합니다."""

from typing import Literal

from pydantic import BaseModel, EmailStr, Field, model_validator


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    email_address: EmailStr
    nickname: str = Field(min_length=1, max_length=50)
    gender: Literal["male", "female", "other"]
    age: int = Field(ge=1, le=120)
    height: float = Field(gt=0, le=300)
    weight: float = Field(gt=0, le=500)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class ProfileUpdateRequest(BaseModel):
    email_address: EmailStr
    nickname: str = Field(min_length=1, max_length=50)
    gender: Literal["male", "female", "other"]
    age: int = Field(ge=1, le=120)
    height: float = Field(gt=0, le=300)
    weight: float = Field(gt=0, le=500)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
    confirm_new_password: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def validate_new_password(self):
        if self.new_password != self.confirm_new_password:
            raise ValueError("새 비밀번호가 일치하지 않습니다.")
        if self.current_password == self.new_password:
            raise ValueError("현재 비밀번호와 다른 비밀번호를 입력해 주세요.")
        return self


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)
