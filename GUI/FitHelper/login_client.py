"""인증 HTTP 어댑터입니다. 인증 상태는 QApplication 컨텍스트 한 곳에 저장합니다."""

import logging
import os

import requests

from GUI.FitHelper.application_context import get_context
from GUI.FitHelper.domain.auth_session import AuthSession

SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
LOGIN_URL = f"{SERVER_URL}/api/auth/login"
ME_URL = f"{SERVER_URL}/api/me"


class LoginClient:
    """상태와 전송 함수를 주입받아 로그인 및 GET /api/me 검증을 수행합니다."""

    def __init__(self, session: AuthSession, request=requests.request) -> None:
        self._session = session
        self._request = request

    def login(self, username: str, password: str) -> bool:
        """두 요청이 모두 성공했을 때만 세션을 설정합니다. 실패 시 이전 상태를 지웁니다.

        Returns:
            인증 성공 여부입니다. 자격 증명과 응답 본문은 로그에 기록하지 않습니다.
        """
        revision = self._session.begin_authentication()
        try:
            response = self._request(
                "POST",
                LOGIN_URL,
                json={"username": username, "password": password},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            token = body.get("access_token")
            if (
                not body.get("success")
                or not isinstance(token, str)
                or not token.strip()
            ):
                return False
            profile_response = self._request(
                "GET",
                ME_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            profile_response.raise_for_status()
            profile = profile_response.json()
            if not profile.get("success") or not isinstance(
                profile.get("user"), dict
            ):
                return False
            return self._session.establish(
                token, profile["user"], expected_revision=revision
            )
        except (requests.RequestException, ValueError, AttributeError):
            logging.getLogger(__name__).warning(
                "로그인 또는 인증 확인에 실패했습니다."
            )
            return False


def login(username: str, password: str) -> bool:
    """기존 화면 호출 경로를 유지합니다. 작업 스레드에서 호출해야 합니다."""
    return LoginClient(get_context().auth).login(username, password)


def get_access_token() -> str | None:
    """화면과 API 클라이언트가 공유하는 현재 토큰을 반환합니다."""
    return get_context().auth.token


def get_user() -> dict | None:
    """현재 사용자 정보의 복사본을 반환합니다."""
    return get_context().auth.user


def update_user(user: dict) -> None:
    """서버에서 저장한 프로필 정보를 단일 인증 상태에 반영합니다."""
    get_context().auth.update_user(user)


def logout() -> None:
    """토큰과 프로필을 동시에 제거합니다. 다른 클라이언트에도 즉시 반영됩니다."""
    get_context().auth.clear()
