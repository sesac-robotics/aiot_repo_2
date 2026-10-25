"""정면·좌측·우측·후면 사진 선택, 비동기 업로드, 분석 상태 폴링을 연결합니다. 작업 완료 신호를 받은 GUI 스레드에서 화면을 갱신합니다.

Photo selection, upload, and server-driven status polling for the GUI."""

import os
import sys
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMainWindow

from GUI.FitHelper import body_analysis_client as body_api
from GUI.FitHelper.body_analysis_cache import (
    area_groups,
    format_area_report,
    save_body_id,
)
from GUI.FitHelper.body_analysis_client import BodyAnalysisAPIError
from GUI.FitHelper.body_analysis_report import BodyAnalysisReportWindow

# The shared API client is saved beside the original AFS/QT files.


class _RequestThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function, *args, parent=None):
        super().__init__(parent)
        self.function, self.args = function, args

    def run(self):
        try:
            self.succeeded.emit(self.function(*self.args))
        except BodyAnalysisAPIError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"요청 처리 중 오류가 발생했습니다: {exc}")


class BodyAnalysisLoadingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "bodyAnalysisLoading.ui"),
            self,
        )
        self.analysisProgressBar.setRange(
            0, 0
        )  # Indeterminate until the server finishes.


class BodyUploadWindow(QMainWindow):
    SIDES = {1: "front", 2: "left", 3: "right", 4: "back"}
    LABELS = {1: "정면", 2: "왼쪽", 3: "오른쪽", 4: "뒷면"}

    def __init__(self, parent=None, access_token=None):
        super().__init__(parent)
        self.welcome_window = parent
        self.uploaded_photos = {1: None, 2: None, 3: None, 4: None}
        self.body_id = None
        self.loading_window = None
        self.report_window = None
        self._request = None
        self._polling = False
        self._switching_to_report = False

        # Pass the token from the successful login response when opening this window.
        if not access_token:
            try:
                from GUI.FitHelper import login_client

                access_token = login_client.get_access_token()
            except ImportError:
                access_token = None
        self.access_token = access_token

        uic.loadUi(
            os.path.join(os.path.dirname(__file__), "bodyUpload.ui"), self
        )
        for index in range(1, 5):
            button = getattr(self, f"uploadPhoto{index}Button")
            button.clicked.connect(
                lambda checked=False, i=index: self.select_photo(i)
            )
        self.submitAnalysisButton.clicked.connect(self.submit_analysis)
        self.backDashboardButton.clicked.connect(self.return_to_welcome)
        # Status lookup is intentionally omitted: the server does not provide
        # GET /body-analysis/status, and this screen does not need prior history.
        if not access_token:
            self.errorMsgLabel.setText(
                "로그인 서버 응답에서 인증 토큰을 받지 못했습니다. 다시 로그인해 주세요."
            )

    def select_photo(self, index):
        name, _ = QFileDialog.getOpenFileName(
            self,
            f"{self.LABELS[index]} 사진 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if name:
            self.uploaded_photos[index] = name
            getattr(self, f"uploadPhoto{index}Button").setText(
                f"✔ {self.LABELS[index]}\n({os.path.basename(name)[:18]})"
            )

    def submit_analysis(self):
        if not self.access_token:
            self.errorMsgLabel.setText(
                "로그인 토큰을 확인하지 못해 서버에 요청할 수 없습니다. 다시 로그인해 주세요."
            )
            return
        missing = [
            self.LABELS[i]
            for i, path in self.uploaded_photos.items()
            if not path
        ]
        if missing:
            self.errorMsgLabel.setText(
                "누락된 사진을 등록해 주세요: " + ", ".join(missing)
            )
            return
        self.errorMsgLabel.clear()
        self.submitAnalysisButton.setEnabled(False)
        self.loading_window = BodyAnalysisLoadingWindow(self.welcome_window)
        self.loading_window.analyzingStatusLabel.setText(
            "사진을 서버에 업로드하고 있습니다..."
        )
        self.loading_window.show()
        self.hide()
        paths = {
            self.SIDES[i]: path for i, path in self.uploaded_photos.items()
        }
        self._run_request(body_api.upload_photos, paths, self._upload_succeeded)

    def _run_request(self, function, *args_and_callback):
        callback = args_and_callback[-1]
        args = args_and_callback[:-1]
        self._request = _RequestThread(function, *args, parent=self)
        self._request.succeeded.connect(callback)
        self._request.failed.connect(self._request_failed)
        self._request.start()

    def _upload_succeeded(self, response):
        self.body_id = response.get("body_id")
        if not self.body_id or response.get("status") != "processing":
            self._request_failed(
                "서버 업로드 응답에 body_id 또는 processing 상태가 없습니다."
            )
            return
        self.loading_window.analyzingStatusLabel.setText(
            "AI가 사진을 분석하고 있습니다..."
        )
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2500)
        self.poll_timer.timeout.connect(self._poll_status)
        self.poll_timer.start()

    def _poll_status(self):
        if self._polling:
            return
        self._polling = True
        self.poll_timer.stop()
        self._run_request(
            body_api.get_analysis, self.body_id, self._poll_succeeded
        )

    def _poll_succeeded(self, response):
        self._polling = False
        state = response.get("status")
        if state == "completed":
            if hasattr(self, "poll_timer"):
                self.poll_timer.stop()
            result = response.get("result", {})
            data = dict(response)
            if isinstance(result, dict):
                data.update(result)
            _, _, undetermined_areas = area_groups(data)
            if len(undetermined_areas) >= 2:
                self._request_failed(
                    "판정 불가 부위가 2곳 이상입니다. "
                    "사진 방향과 인물이 잘 보이는지 확인한 뒤 다시 업로드해 주세요."
                )
                return
            self._show_report(result, response)
        elif state == "failed":
            self._request_failed(
                "AI 체형 분석에 실패했습니다. 사진을 확인한 뒤 다시 시도해 주세요."
            )
        else:
            self.loading_window.analyzingStatusLabel.setText(
                "AI가 체형 분석을 진행 중입니다..."
            )
            self.poll_timer.start()

    def _request_failed(self, message):
        self._polling = False
        if hasattr(self, "poll_timer"):
            self.poll_timer.stop()
        if self.loading_window:
            self.loading_window.close()
            self.loading_window = None
        self.show()
        self.submitAnalysisButton.setEnabled(True)
        self.errorMsgLabel.setText(message)

    def _show_report(self, result, response=None):
        if self.loading_window:
            self.loading_window.close()
            self.loading_window = None
        # Some API versions place recognition metadata beside `result`, while
        # others put it inside `result`; accept either response shape.
        data = dict(response or {})
        if isinstance(result, dict):
            data.update(result)

        self.report_window = BodyAnalysisReportWindow(
            self.welcome_window,
            analysis_data=data,
        )

        self.report_window.weaknessLabel.setText(format_area_report(data))
        self._switching_to_report = True
        try:
            save_body_id(self.welcome_window.login_window, self.body_id)
            self.welcome_window.display_completed_analysis(data, self.body_id)
        except (AttributeError, OSError, ValueError) as exc:
            print(f"[체형 분석 캐시 저장 실패] {exc}", flush=True)
        self.report_window.show()
        self.close()

    def return_to_welcome(self):
        if self.welcome_window:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if hasattr(self, "poll_timer"):
            self.poll_timer.stop()
        if self.welcome_window and not self._switching_to_report:
            self.welcome_window.show()
        event.accept()
