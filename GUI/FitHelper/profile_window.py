"""프로필 조회·변경, 비밀번호 변경, 탈퇴 확인 화면을 연결합니다. HTTP 요청은 작업 스레드에서 수행하고 완료 콜백에서 위젯과 인증 상태를 갱신합니다."""

import os

import requests
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow

from GUI.FitHelper import login_client
from GUI.FitHelper.request_worker import RequestTask

SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL",
    getattr(login_client, "SERVER_URL", "http://127.0.0.1:8000"),
).rstrip("/")


class ProfileAPIError(Exception):
    pass


class ProfileSettingsWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        self.delete_window = None

        ui_path = os.path.join(os.path.dirname(__file__), "profileSettings.ui")
        uic.loadUi(ui_path, self)

        self.saveButton.clicked.connect(self.save_profile_changes)
        self.changePasswordButton.clicked.connect(self.change_password)
        self.deleteAccountButton.clicked.connect(self.delete_account)
        self.backDashboardButton.clicked.connect(self.go_back_to_dashboard)
        self.load_profile()

    def _request(self, method, path, payload=None):
        token = login_client.get_access_token()
        if not token:
            raise ProfileAPIError(
                "로그인 정보가 만료되었습니다. 다시 로그인해 주세요."
            )

        try:
            response = requests.request(
                method,
                f"{SERVER_URL}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
        except requests.Timeout as exc:
            raise ProfileAPIError("서버 응답 시간이 초과되었습니다.") from exc
        except requests.RequestException as exc:
            raise ProfileAPIError(
                f"AIFit Server에 연결할 수 없습니다: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError:
            data = {}
        if not response.ok:
            detail = (
                data.get("detail")
                or data.get("message")
                or f"HTTP {response.status_code}"
            )
            if isinstance(detail, list):
                detail = "; ".join(
                    str(item.get("msg", item)) for item in detail
                )
            raise ProfileAPIError(str(detail))
        return data

    def _show_error(self, message, label=None):
        target = label or getattr(self, "errorMsgLabel", None)
        if target is not None:
            target.setText(f"⚠️ {message}")

    def _submit(self, method, path, payload, success, label=None):
        if getattr(self, "_request_pending", False):
            return
        self._request_pending = True
        for button in (
            self.saveButton,
            self.changePasswordButton,
            self.deleteAccountButton,
        ):
            button.setEnabled(False)

        def finish():
            self._request_pending = False
            for button in (
                self.saveButton,
                self.changePasswordButton,
                self.deleteAccountButton,
            ):
                button.setEnabled(True)

        def complete(data):
            finish()
            success(data)

        def failed(error):
            finish()
            if self.delete_window is not None:
                self.delete_window.confirmDeleteButton.setEnabled(True)
            self._show_error(
                "요청을 완료하지 못했습니다. 입력값과 서버 상태를 확인해 주세요.",
                label,
            )

        self._profile_task = RequestTask(
            self, lambda: self._request(method, path, payload), complete, failed
        )
        self._profile_task.start()

    def load_profile(self):
        self._submit("GET", "/api/me", None, self._profile_loaded)

    def _profile_loaded(self, data):
        user = data.get("user", {})
        self.editNickname.setText(str(user.get("nickname", "")))
        self.editEmail.setText(str(user.get("email_address", "")))
        self.editAge.setText(str(user.get("age", "")))
        self.editHeight.setText(str(user.get("height", "")))
        self.editWeight.setText(str(user.get("weight", "")))
        gender = str(user.get("gender", "")).lower()
        gender_index = self.comboGender.findText(gender.title())
        if gender_index >= 0:
            self.comboGender.setCurrentIndex(gender_index)

    def save_profile_changes(self):
        nickname = self.editNickname.text().strip()
        email = self.editEmail.text().strip()
        gender = self.comboGender.currentText().strip().lower()
        try:
            age = int(self.editAge.text().strip())
            height = float(self.editHeight.text().strip())
            weight = float(self.editWeight.text().strip())
        except ValueError:
            self._show_error("나이, 키, 몸무게를 숫자로 입력해 주세요.")
            return

        if not nickname:
            self._show_error("닉네임을 입력해 주세요.")
            return

        payload = {
            "email_address": email,
            "nickname": nickname,
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
        }
        self._submit("PUT", "/api/me", payload, self._profile_saved)

    def _profile_saved(self, data):
        user = data.get("user", {})
        if isinstance(login_client.get_user(), dict):
            login_client.update_user(user)
        self.errorMsgLabel.setText("프로필 변경사항을 저장했습니다.")

    def change_password(self):
        current_pw = self.editCurrentPassword.text()
        new_pw = self.editNewPassword.text()
        confirm_pw = self.editConfirmNewPassword.text()
        if not current_pw or not new_pw or not confirm_pw:
            self._show_error(
                "현재 비밀번호와 새 비밀번호를 모두 입력해 주세요."
            )
            return
        if new_pw != confirm_pw:
            self._show_error("새 비밀번호가 서로 일치하지 않습니다.")
            return
        if len(new_pw) < 8:
            self._show_error("새 비밀번호는 8자 이상이어야 합니다.")
            return

        self._submit(
            "POST",
            "/api/me/change-password",
            {
                "current_password": current_pw,
                "new_password": new_pw,
                "confirm_new_password": confirm_pw,
            },
            self._password_changed,
        )

    def _password_changed(self, data):
        self.editCurrentPassword.clear()
        self.editNewPassword.clear()
        self.editConfirmNewPassword.clear()
        self.errorMsgLabel.setText(
            "비밀번호를 변경했습니다. 다음 로그인부터 새 비밀번호를 사용하세요."
        )

    def delete_account(self):
        try:
            self.delete_window = QMainWindow()
            ui_path = os.path.join(
                os.path.dirname(__file__), "deleteAccount.ui"
            )
            uic.loadUi(ui_path, self.delete_window)
            self.delete_window.cancelButton.clicked.connect(
                self.cancel_delete_account
            )
            self.delete_window.confirmDeleteButton.clicked.connect(
                self.confirm_delete_account
            )
            self.delete_window.confirmPasswordEdit.clear()
            self.delete_window.errorMsgLabel.clear()
            self.delete_window.show()
            self.hide()
        except Exception as exc:
            self._show_error(f"탈퇴 확인 창을 열지 못했습니다: {exc}")

    def confirm_delete_account(self):
        password = self.delete_window.confirmPasswordEdit.text()
        if not password:
            self._show_error(
                "비밀번호를 입력해 주세요.", self.delete_window.errorMsgLabel
            )
            return

        self.delete_window.confirmDeleteButton.setEnabled(False)
        self._submit(
            "POST",
            "/api/me/delete-account",
            {"password": password},
            self._account_deleted,
            self.delete_window.errorMsgLabel,
        )

    def _account_deleted(self, data):
        login_window = getattr(self.welcome_window, "login_window", None)
        if hasattr(login_client, "logout"):
            login_client.logout()
        self.delete_window.close()
        self.welcome_window = None
        self.close()
        if login_window is not None:
            login_window.show()

    def cancel_delete_account(self):
        if self.delete_window is not None:
            self.delete_window.close()
        self.show()

    def go_back_to_dashboard(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        else:
            from GUI.FitHelper.welcome_window import WelcomeWindow

            self.welcome_window = WelcomeWindow()
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
