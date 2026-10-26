"""Login client shared by the AIFit GUI and API clients."""

import os

import requests


SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://10.17.178.246:8000").rstrip("/")
LOGIN_URL = f"{SERVER_URL}/api/auth/login"
ME_URL = f"{SERVER_URL}/api/me"

_access_token = None
_user = None


def login(username: str, password: str) -> bool:
    """Log in against AIFit Server and cache the JWT for other GUI modules."""
    global _access_token, _user

    try:
        response = requests.post(
            LOGIN_URL,
            json={"username": username, "password": password},
            timeout=10,
        )
        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            detail = data.get("detail") or data.get("message") or f"HTTP {response.status_code}"
            print(f"[로그인 실패] {detail}", flush=True)
            _access_token, _user = None, None
            return False

        token = data.get("access_token")
        if not data.get("success") or not isinstance(token, str) or not token.strip():
            detail = data.get("detail") or data.get("message") or "로그인 응답에 access_token이 없습니다."
            print(f"[로그인 실패] {detail}", flush=True)
            _access_token, _user = None, None
            return False

        _access_token = token.strip()
        _user = data.get("user")

        # Keep the body-analysis client in sync when it is already loaded.
        import sys

        body_api = sys.modules.get("body_analysis_client")
        if body_api and hasattr(body_api, "set_access_token"):
            body_api.set_access_token(_access_token)

        return True

    except requests.RequestException as exc:
        _access_token, _user = None, None
        print(f"[로그인 통신 오류] {exc}", flush=True)
        return False


def logout() -> None:
    """Clear the cached JWT and remove it from the body-analysis client."""
    global _access_token, _user
    _access_token, _user = None, None

    try:
        import body_analysis_client

        body_analysis_client.set_access_token(None)
    except ImportError:
        pass
