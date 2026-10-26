"""Desktop client for the AIFit body-analysis API."""
import os
from pathlib import Path

import requests

AIFIT_SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://10.17.178.246:8000").rstrip("/")
TIMEOUT_SECONDS = 30
_access_token: str | None = None


class BodyAnalysisAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def set_access_token(token: str | None) -> None:
    """Call after login succeeds; pass None on logout."""
    global _access_token
    _access_token = token


def _request(method: str, path: str, **kwargs):
    token = _access_token
    if not token:
        try:
            import login_client
            token = getattr(login_client, "_access_token", None)
        except ImportError:
            token = None
    if not token:
        raise BodyAnalysisAPIError("로그인이 필요합니다. 로그인 후 다시 시도해 주세요.")
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.request(method, AIFIT_SERVER_URL + path,
                                    headers=headers, timeout=kwargs.pop("timeout", TIMEOUT_SECONDS), **kwargs)
    except requests.Timeout as exc:
        raise BodyAnalysisAPIError("AIFit Server 응답 시간이 초과되었습니다.") from exc
    except requests.RequestException as exc:
        raise BodyAnalysisAPIError(f"AIFit Server 통신 오류: {exc}") from exc
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        detail = data.get("detail", f"HTTP {response.status_code}")
        raise BodyAnalysisAPIError(str(detail), response.status_code)
    return data


def get_status():
    return _request("GET", "/body-analysis/status")


def upload_photos(photo_paths: dict[str, str]):
    sides = ("front", "left", "right", "back")
    missing = [side for side in sides if not photo_paths.get(side)]
    if missing:
        raise BodyAnalysisAPIError("사진을 모두 선택해 주세요. 누락: " + ", ".join(missing))
    files, opened = {}, []
    try:
        for side in sides:
            path = Path(photo_paths[side])
            handle = path.open("rb")
            opened.append(handle)
            mime = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
            files[f"{side}_photo"] = (path.name, handle, mime)
        return _request("POST", "/body-analysis", files=files)
    except OSError as exc:
        raise BodyAnalysisAPIError(f"사진 파일을 열 수 없습니다: {exc}") from exc
    finally:
        for handle in opened:
            handle.close()


def get_analysis(body_id: str):
    return _request("GET", f"/body-analysis/{body_id}")
