"""이전 로그인 API를 시험하던 호환용 클라이언트입니다. 현재 GUI의 인증 진입점은 login_client와 login_window이며 이 파일은 과거 호출 계약 비교용으로 보존합니다."""

import os

import requests

AIFIT_SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
LOGIN_URL = f"{AIFIT_SERVER_URL}/api/login"


class LoginError(Exception):
    """Login error with a safe, user-facing message."""

    def __init__(self, message, user_message=None):
        super().__init__(message)
        self.user_message = user_message or message


def login(username: str, password: str) -> dict:
    """
    Authenticate against AIFit Server.

    Returns the successful JSON response, including access_token and user.
    Raises LoginError when authentication or communication fails.
    """
    payload = {
        "username": username,
        "password": password,
    }

    print(f"[로그인 요청] POST {LOGIN_URL}", flush=True)

    try:
        response = requests.post(
            LOGIN_URL,
            json=payload,
            timeout=10,
        )
    except requests.Timeout as exc:
        print("[로그인 통신 오류] 서버 응답 시간 초과", flush=True)
        raise LoginError(
            "Login request timed out",
            "서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
        ) from exc
    except requests.ConnectionError as exc:
        print("[로그인 통신 오류] AIFit Server 연결 실패", flush=True)
        raise LoginError(
            "Cannot connect to AIFit Server",
            "AIFit Server에 연결할 수 없습니다. 서버 주소와 네트워크를 확인해 주세요.",
        ) from exc
    except requests.RequestException as exc:
        print(f"[로그인 통신 오류] {type(exc).__name__}", flush=True)
        raise LoginError(
            "Login request failed",
            "로그인 요청 중 통신 오류가 발생했습니다.",
        ) from exc

    print(f"[로그인 응답] HTTP {response.status_code}", flush=True)

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code == 401:
        print("[로그인 실패] 아이디 또는 비밀번호 불일치", flush=True)
        raise LoginError(
            "Invalid username or password",
            "아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    if not response.ok:
        detail = body.get(
            "detail", "서버에서 로그인 요청을 처리하지 못했습니다."
        )
        print(
            f"[로그인 실패] HTTP {response.status_code}: {detail}", flush=True
        )
        raise LoginError(
            f"Login failed with HTTP {response.status_code}: {detail}",
            f"로그인에 실패했습니다. (HTTP {response.status_code})",
        )

    access_token = body.get("access_token")
    user = body.get("user")

    if (
        not body.get("success")
        or not access_token
        or not isinstance(user, dict)
    ):
        print(
            "[로그인 실패] 응답에 JWT 또는 사용자 정보가 없습니다.", flush=True
        )
        raise LoginError(
            "Malformed login response",
            "서버 응답에 인증 토큰이 없습니다. AIFit Server 코드를 확인해 주세요.",
        )

    # Never print the access token to the terminal.
    print(
        "[로그인 성공] JWT 수신 | "
        f"user_id={user.get('user_id', 'unknown')} | "
        f"username={user.get('username', 'unknown')}",
        flush=True,
    )
    return body
