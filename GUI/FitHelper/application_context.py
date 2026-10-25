"""QApplication이 소유하는 의존성을 조립합니다. 네트워크 연결은 만들지 않습니다."""

from dataclasses import dataclass, field

from PyQt6.QtWidgets import QApplication

from GUI.FitHelper.domain.auth_session import AuthSession


@dataclass
class ApplicationContext:
    """하나의 GUI 실행에서 공유하는 상태입니다. 전역 토큰 복제를 방지합니다."""

    auth: AuthSession = field(default_factory=AuthSession)


def get_context() -> ApplicationContext:
    """실행 진입점이 주입한 컨텍스트를 반환합니다.

    Raises:
        RuntimeError: main을 통하지 않아 컨텍스트가 구성되지 않았습니다.
    """
    app = QApplication.instance()
    context = getattr(app, "aifit_context", None)
    if not isinstance(context, ApplicationContext):
        raise RuntimeError(
            "python -m GUI.FitHelper.main으로 GUI를 실행해야 합니다."
        )
    return context
