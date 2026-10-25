"""AI 서버의 HTTP 통신을 업무 흐름에서 분리한 어댑터입니다.

AsyncClient를 주입하면 실제 AI 서버 없이 요청 경로·타임아웃·인증을 검증할 수
있습니다. 재시도는 수행하지 않습니다. start/finish 같은 상태 변경을 자동으로
재전송하면 중복 실행될 수 있기 때문입니다.
"""

from typing import Any, Protocol

import httpx


class GatewayError(RuntimeError):
    """AI 서버 미설정 또는 통신 실패를 표현합니다. 웹 계층이 응답으로 변환합니다."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.unavailable = unavailable


class AIGateway(Protocol):
    """업무 흐름에 필요한 최소 외부 통신 포트입니다."""

    async def request(
        self, method: str, path: str, payload: dict | None = None
    ) -> Any:
        """내부 API를 호출하고 JSON·텍스트·빈 응답 중 하나를 반환합니다."""
        ...


class HTTPAIGateway:
    """HTTPX를 사용하는 포트 구현입니다. URL과 비밀 키는 생성 시 주입합니다."""

    def __init__(
        self, base_url: str, api_key: str, client: httpx.AsyncClient
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client

    async def request(
        self, method: str, path: str, payload: dict | None = None
    ) -> Any:
        """상태 변경을 한 번만 요청하며 응답 본문이나 인증 정보를 로그에 남기지 않습니다.

        Raises:
            GatewayError: 설정이 없거나 연결·HTTP 응답이 실패했습니다.
        """
        if not self._base_url or not self._api_key:
            raise GatewayError(
                "AI Server 연결 설정이 없습니다.", unavailable=True
            )
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                headers={"X-Internal-API-Key": self._api_key},
                json=payload,
            )
        except httpx.RequestError as exc:
            raise GatewayError("AI Server에 연결할 수 없습니다.") from exc
        if response.is_error:
            raise GatewayError(
                f"AI Server 요청이 실패했습니다 (HTTP {response.status_code})."
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text
