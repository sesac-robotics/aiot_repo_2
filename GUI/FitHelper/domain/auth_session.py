"""로그인 상태의 단일 소유자입니다. Qt와 HTTP 라이브러리에 의존하지 않습니다."""

from copy import deepcopy
from threading import RLock
from typing import Any


class AuthSession:
    """토큰과 사용자 정보를 함께 갱신하고 외부에 복사본만 제공합니다.

    작업 스레드에서 인증을 수행해도 화면이 부분 갱신된 상태를 읽지 않도록
    동일한 잠금으로 보호합니다. 세션은 메모리에만 존재하며 디스크에 저장하지 않습니다.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._revision = 0
        self._token: str | None = None
        self._user: dict[str, Any] | None = None

    @property
    def token(self) -> str | None:
        """현재 Bearer 토큰을 반환합니다. 로그인하지 않았으면 None입니다."""
        with self._lock:
            return self._token

    @property
    def user(self) -> dict[str, Any] | None:
        """수정해도 내부 상태에 영향을 주지 않는 사용자 복사본을 반환합니다."""
        with self._lock:
            return deepcopy(self._user)

    def establish(
        self,
        token: str,
        user: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """검증된 인증 결과를 원자적으로 교체합니다. 빈 토큰은 허용하지 않습니다."""
        if not isinstance(token, str) or not token.strip():
            raise ValueError("인증 토큰이 비어 있습니다.")
        with self._lock:
            if (
                expected_revision is not None
                and expected_revision != self._revision
            ):
                return False
            self._revision += 1
            self._token = token.strip()
            self._user = deepcopy(user)
            return True

    def update_user(self, user: dict[str, Any]) -> None:
        """로그인 상태에서 서버가 확인한 프로필 필드만 합칩니다."""
        with self._lock:
            if self._user is None:
                raise ValueError("로그인이 필요합니다.")
            self._user.update(deepcopy(user))

    def begin_authentication(self) -> int:
        """이전 상태를 지우고 요청 세대를 발급합니다. 로그아웃 후 도착한 결과를 차단합니다."""
        with self._lock:
            self._revision += 1
            self._token = None
            self._user = None
            return self._revision

    def clear(self) -> None:
        """로그아웃 시 인증 상태를 지우고 진행 중 로그인 결과도 무효화합니다."""
        self.begin_authentication()
