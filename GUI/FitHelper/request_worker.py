"""시간이 걸리는 요청을 Qt 스레드 풀에서 실행하고 결과만 화면으로 전달합니다."""

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot


class _Signals(QObject):
    completed = pyqtSignal(object, object)


class _Request(QRunnable):
    def __init__(self, task: Callable[[], Any], signals: _Signals) -> None:
        super().__init__()
        self._task = task
        self._signals = signals

    def run(self) -> None:
        try:
            value = self._task()
        except Exception as exc:
            # This is a thread boundary: forward failures to the GUI, never swallow.
            self._signals.completed.emit(None, exc)
        else:
            self._signals.completed.emit(value, None)


class RequestTask(QObject):
    """화면 생존 기간에 연결된 요청입니다. 콜백은 GUI 스레드에서만 호출됩니다.

    owner가 파괴되면 Qt가 신호 연결을 해제합니다. 진행 중 HTTP는 자체 timeout까지
    종료되며, QThread를 강제 종료하거나 이미 파괴된 화면을 갱신하지 않습니다.
    """

    def __init__(
        self,
        owner: QObject,
        task: Callable[[], Any],
        success: Callable[[Any], None],
        failure: Callable[[Exception], None],
    ) -> None:
        super().__init__(owner)
        self._cancelled = False
        self._success = success
        self._failure = failure
        # Keep signals alive independently until the runnable finishes.
        self._signals = _Signals()
        self._signals.completed.connect(self._deliver)
        self._request = _Request(task, self._signals)

    def start(self) -> None:
        """이벤트 루프를 차단하지 않고 요청을 한 번 제출합니다."""
        QThreadPool.globalInstance().start(self._request)

    def cancel(self) -> None:
        """결과 반영을 취소합니다. 네트워크 스레드는 자체 timeout으로 정리됩니다."""
        self._cancelled = True

    @pyqtSlot(object, object)
    def _deliver(self, value: Any, error: Exception | None) -> None:
        if self._cancelled:
            self.deleteLater()
            return
        if error is None:
            self._success(value)
        else:
            self._failure(error)
        self.deleteLater()
