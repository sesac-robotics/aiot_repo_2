"""통합 실행 수명을 소유합니다. 운동 반복 판정이나 카운터는 소유하지 않습니다."""

import enum


class SessionPhase(enum.Enum):
    """카메라·모델과 독립된 실행 상태입니다."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class SessionLifecycle:
    """단일 제어 스트림의 상태 전이만 소유하는 순수 도메인 객체입니다.

    application 서비스가 제어 잠금을 소유합니다. 중복 정지·일시정지는
    카운터를 바꾸거나 새로운 제어 이벤트를 만들지 않습니다.
    """

    def __init__(self) -> None:
        self._phase = SessionPhase.IDLE

    @property
    def phase(self) -> SessionPhase:
        """현재 실행 상태를 반환합니다."""
        return self._phase

    @property
    def enabled(self) -> bool:
        """실행 중이거나 일시정지된 세션인지 반환합니다."""
        return self._phase in (SessionPhase.RUNNING, SessionPhase.PAUSED)

    def start(self) -> None:
        """준비된 자원을 실행 상태로 반영합니다.

        Raises:
            ValueError: 기존 실행을 종료하지 않고 다시 시작한 경우.
        """
        if self.enabled:
            raise ValueError("기존 세션을 종료한 후 시작하십시오.")
        self._phase = SessionPhase.RUNNING

    def pause(self) -> bool:
        """첫 일시정지만 반영하며 중복 요청이면 False를 반환합니다.

        Raises:
            ValueError: 시작 전이거나 종료한 세션인 경우.
        """
        if self._phase is SessionPhase.PAUSED:
            return False
        if self._phase is not SessionPhase.RUNNING:
            raise ValueError("실행 중인 운동이 없습니다.")
        self._phase = SessionPhase.PAUSED
        return True

    def resume(self) -> bool:
        """일시정지된 세션만 재개하고 실제 전이 여부를 반환합니다."""
        if self._phase is not SessionPhase.PAUSED:
            return False
        self._phase = SessionPhase.RUNNING
        return True

    def stop(self) -> None:
        """반복 호출에도 추가 전이가 없는 종료를 반영합니다."""
        if self._phase is not SessionPhase.IDLE:
            self._phase = SessionPhase.STOPPED
