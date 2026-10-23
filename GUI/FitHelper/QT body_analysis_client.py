"""HTTP client for AIFit body-analysis endpoints; reuses the existing login JWT."""
import os
import requests
import login_client

SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://192.168.15.118:8000").rstrip("/")
TIMEOUT_SECONDS = 30


class BodyAnalysisAPIError(Exception):
    pass


def _headers():
    token = getattr(login_client, "_access_token", None)
    if not token:
        raise BodyAnalysisAPIError(
            "로그인 토큰을 찾을 수 없습니다. 로그인 후 login_client.py가 "
            "access_token을 저장하는지 확인해 주세요."
        )
    return {"Authorization": f"Bearer {token}"}


def _request(method, path, **kwargs):
    try:
        response = requests.request(
            method, f"{SERVER_URL}{path}",
            headers={**_headers(), **kwargs.pop("headers", {})},
            timeout=kwargs.pop("timeout", TIMEOUT_SECONDS),
            **kwargs,
        )
    except requests.Timeout as exc:
        raise BodyAnalysisAPIError("AIFit Server 응답 시간이 초과되었습니다.") from exc
    except requests.ConnectionError as exc:
        raise BodyAnalysisAPIError(f"AIFit Server에 연결할 수 없습니다: {SERVER_URL}") from exc
    except requests.RequestException as exc:
        raise BodyAnalysisAPIError(f"통신 오류: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        detail = data.get("detail") or data.get("message") or f"HTTP {response.status_code}"
        if isinstance(detail, list):
            detail = "; ".join(str(x.get("msg", x)) for x in detail)
        if response.status_code == 401:
            raise BodyAnalysisAPIError("인증 토큰이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요.")
        raise BodyAnalysisAPIError(f"서버 오류 (HTTP {response.status_code}): {detail}")
    return data


def get_status():
    return _request("GET", "/body-analysis/status")


def upload_photos(photo_paths):
    """Multipart keys must match server: front_photo, left_photo, right_photo, back_photo."""
    sides = ("front", "left", "right", "back")
    missing = [side for side in sides if not photo_paths.get(side)]
    if missing:
        raise BodyAnalysisAPIError("사진이 누락되었습니다: " + ", ".join(missing))

    files, opened = {}, []
    try:
        for side in sides:
            path = photo_paths[side]
            handle = open(path, "rb")
            opened.append(handle)
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            files[f"{side}_photo"] = (os.path.basename(path), handle, mime)
        return _request("POST", "/body-analysis", files=files)
    except OSError as exc:
        raise BodyAnalysisAPIError(f"사진 파일을 열 수 없습니다: {exc}") from exc
    finally:
        for handle in opened:
            handle.close()


def get_analysis(body_id):
    return _request("GET", f"/body-analysis/{body_id}")