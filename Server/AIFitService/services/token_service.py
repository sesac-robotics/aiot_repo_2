"""Bearer JWT의 서명·발급 시각·만료와 사용자 존재를 검증합니다. 프레임워크 의존성으로 보호 API에 인증 사용자를 전달합니다."""

import logging

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from Server.AIFitService.config.settings import JWT_ALGORITHM, JWT_SECRET
from Server.AIFitService.repositories.user_repository import find_public_by_id

bearer_scheme = HTTPBearer(auto_error=False)

logger = logging.getLogger(__name__)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Validate Authorization: Bearer <token> and return the current DB user."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증이 필요하거나 토큰이 유효하지 않습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None or credentials.scheme.lower() != "bearer":
        logger.info("[AUTH] Protected API rejected")
        raise unauthorized

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp", "iat"]},
        )
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.isdigit():
            raise unauthorized

        user = find_public_by_id(int(subject))
        if user is None:
            raise unauthorized

        logger.info("[AUTH] JWT validated")
        return dict(user)
    except jwt.InvalidTokenError:
        logger.info("[AUTH] Protected API rejected")
        raise unauthorized
