import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import JWT_SECRET, JWT_ALGORITHM
from Server.AIFitService.repositories.user_repository import find_public_by_id

bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    """Validate Authorization: Bearer <token> and return the current DB user."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증이 필요하거나 토큰이 유효하지 않습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None or credentials.scheme.lower() != "bearer":
        print("[AUTH] Protected API rejected: missing Bearer token")
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

        print(f"[AUTH] JWT validated: user_id={subject}")
        return dict(user)
    except jwt.InvalidTokenError:
        print("[AUTH] Protected API rejected: invalid or expired JWT")
        raise unauthorized
