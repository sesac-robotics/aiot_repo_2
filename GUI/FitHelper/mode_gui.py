"""원본 로그인·대시보드에 독립 실행 모드를 연결하는 확장입니다."""

import importlib
import json
from pathlib import Path

from PyQt6.QtCore import QByteArray, QUrl, pyqtSignal
from PyQt6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from GUI.FitHelper.application_context import ApplicationContext

try:
    from .direct_correction_window import DirectCorrectionWindow
    from .score_mode_window import ScoreModeWindow
except ImportError:
    from direct_correction_window import DirectCorrectionWindow
    from score_mode_window import ScoreModeWindow

EXERCISES = {
    "squats": "스쿼트",
    "pushups": "푸시업",
    "shoulder": "숄더 프레스",
    "plank": "엘보 플랭크",
    "ytw": "YTW",
}


class CorrectionModeWindow(DirectCorrectionWindow):
    closed = pyqtSignal()

    def closeEvent(self, event):
        confirmed = self.allow_close
        super().closeEvent(event)
        if confirmed:
            self.closed.emit()


class ModeChooser(QDialog):
    def __init__(
        self,
        original_root,
        parent=None,
        score_url="http://127.0.0.1:8790",
        correction_url="http://127.0.0.1:8782",
        integrated_url="http://127.0.0.1:8793",
    ):
        super().__init__(parent)
        self.original_root = Path(original_root)
        self.score_url, self.correction_url = score_url, correction_url
        self.integrated_url = integrated_url
        self.child = None
        self.busy = False
        self.network = QNetworkAccessManager(self)
        self.mode = QComboBox()
        self.mode.addItem("운동 교정", "correction")
        self.mode.addItem("자세 점수", "score")
        self.mode.addItem("운동 교정 + 자세 점수", "combined")
        self.exercise = QComboBox()
        for code, name in EXERCISES.items():
            self.exercise.addItem(name, code)
        self.webcam = QSpinBox()
        self.webcam.setRange(0, 32)
        self.open_button = QPushButton("선택한 모드 열기")
        self.back_button = QPushButton("대시보드로 돌아가기")
        self.notice = QLabel(
            "웹캠을 사용합니다. 한 번에 하나의 모드만 실행합니다."
        )
        self.notice.setWordWrap(True)
        self.setWindowTitle("AIFit · 운동 모드 선택")
        self.resize(510, 260)
        form = QFormLayout()
        form.addRow("실행 모드", self.mode)
        form.addRow("운동", self.exercise)
        form.addRow("웹캠 번호", self.webcam)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.notice)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.back_button)
        layout.addLayout(buttons)
        self.open_button.clicked.connect(self.open_mode)
        self.back_button.clicked.connect(self.close)

    def open_mode(self):
        if self.busy or self.child is not None:
            return
        self.busy = True
        self.open_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.mode.setEnabled(False)
        self.exercise.setEnabled(False)
        self.webcam.setEnabled(False)
        self.notice.setText("이전 측정의 중지를 확인합니다.")
        self.stop_next(
            [self.correction_url, self.score_url, self.integrated_url]
        )

    def stop_next(self, urls):
        if not urls:
            self.show_mode()
            return
        request = QNetworkRequest(QUrl(urls[0] + "/api/stop"))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        request.setTransferTimeout(20000)
        reply = self.network.post(request, QByteArray(b"{}"))

        def finished():
            try:
                if reply.error() != QNetworkReply.NetworkError.NoError:
                    raise ValueError(reply.errorString())
                if (
                    json.loads(bytes(reply.readAll())).get("running")
                    is not False
                ):
                    raise ValueError("이전 측정이 아직 실행 중입니다.")
                self.stop_next(urls[1:])
            except (ValueError, TypeError) as exc:
                self.notice.setText("모드 전환을 중단했습니다: " + str(exc))
                self.release()
            finally:
                reply.deleteLater()

        reply.finished.connect(finished)

    def show_mode(self):
        exercise = self.exercise.currentData()
        if self.mode.currentData() == "combined":
            try:
                from .integrated_window import IntegratedWindow
            except ImportError:
                from integrated_window import IntegratedWindow
            self.child = IntegratedWindow(
                self.original_root / "workout_window.ui",
                exercise,
                self.webcam.value(),
                self.integrated_url,
            )
        elif self.mode.currentData() == "score":
            self.child = ScoreModeWindow(
                self.original_root / "workout_window.ui",
                exercise,
                self.webcam.value(),
                self.score_url,
            )
        else:
            self.child = CorrectionModeWindow(self.correction_url)
            code = {"shoulder": "shoulder_press", "plank": "elbow_plank"}.get(
                exercise, exercise
            )
            self.child.exercise.setCurrentIndex(
                self.child.exercise.findData(code)
            )
            self.child.webcam.setValue(self.webcam.value())
        self.child.closed.connect(self.return_from_mode)
        self.child.show()
        self.hide()

    def release(self):
        self.busy = False
        for widget in (
            self.open_button,
            self.back_button,
            self.mode,
            self.exercise,
            self.webcam,
        ):
            widget.setEnabled(True)

    def return_from_mode(self):
        self.child = None
        self.release()
        self.notice.setText("측정을 중지했습니다. 실행 모드를 선택하십시오.")
        self.show()

    def closeEvent(self, event):
        if self.busy or self.child is not None:
            event.ignore()
            return
        if self.parentWidget():
            self.parentWidget().show()
        event.accept()


def create_login(
    original_root,
    score_url="http://127.0.0.1:8790",
    correction_url="http://127.0.0.1:8782",
    integrated_url="http://127.0.0.1:8793",
):
    """원본 파일을 유지하고 새 실행 프로세스에서만 하위 클래스를 연결합니다."""
    original_root = Path(original_root).resolve()
    if original_root != Path(__file__).resolve().parent:
        raise ValueError(
            "현재 저장소의 GUI/FitHelper 원본 경로를 사용하십시오."
        )
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication을 먼저 생성하십시오.")
    if not isinstance(getattr(app, "aifit_context", None), ApplicationContext):
        app.aifit_context = ApplicationContext()
    welcome_module = importlib.import_module("GUI.FitHelper.welcome_window")
    original_welcome = welcome_module.WelcomeWindow

    class ModeWelcomeWindow(original_welcome):
        def load_dashboard_ui(self):
            super().load_dashboard_ui()
            self.startWorkoutButton.setText(
                "운동 모드 선택 · 교정 / 점수 / 통합"
            )
            if not self.has_completed_analysis:
                self.todayDescLabel.setText(
                    "체형분석 없이 교정 또는 자세 점수를 선택할 수 있습니다."
                )

        def go_to_workout(self):
            self.mode_chooser = ModeChooser(
                original_root, self, score_url, correction_url, integrated_url
            )
            self.mode_chooser.show()
            self.hide()

    # The unchanged LoginWindow imports this class after successful authentication.
    welcome_module.WelcomeWindow = ModeWelcomeWindow
    login_module = importlib.import_module("GUI.FitHelper.login_window")
    return login_module.LoginWindow()
