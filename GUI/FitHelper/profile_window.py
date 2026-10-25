import os

import requests
from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic

import login_client

SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL",
    getattr(login_client, "SERVER_URL", "http://192.168.15.148:8000"),
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
        token = getattr(login_client, "_access_token", None)
        if not token:
            raise ProfileAPIError("로그인 정보가 만료되었습니다. 다시 로그인해 주세요.")

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
            raise ProfileAPIError(f"AIFit Server에 연결할 수 없습니다: {exc}") from exc

        try:
            data = response.json()
        except ValueError:
            data = {}
        if not response.ok:
            detail = data.get("detail") or data.get("message") or f"HTTP {response.status_code}"
            if isinstance(detail, list):
                detail = "; ".join(str(item.get("msg", item)) for item in detail)
            raise ProfileAPIError(str(detail))
        return data

    def _show_error(self, message, label=None):
        target = label or getattr(self, "errorMsgLabel", None)
        if target is not None:
            target.setText(f"⚠️ {message}")

    def load_profile(self):
        try:
            data = self._request("GET", "/api/me")
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
        except ProfileAPIError as exc:
            self._show_error(f"프로필을 불러오지 못했습니다: {exc}")

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

        try:
            data = self._request("PUT", "/api/me", {
                "email_address": email,
                "nickname": nickname,
                "gender": gender,
                "age": age,
                "height": height,
                "weight": weight,
            })
        except ProfileAPIError as exc:
            self._show_error(f"프로필 저장 실패: {exc}")
            return

        user = data.get("user", {})
        if isinstance(getattr(login_client, "_user", None), dict):
            login_client._user.update(user)
        self.errorMsgLabel.setText("프로필 변경사항을 저장했습니다.")

    def change_password(self):
        current_pw = self.editCurrentPassword.text()
        new_pw = self.editNewPassword.text()
        confirm_pw = self.editConfirmNewPassword.text()
        if not current_pw or not new_pw or not confirm_pw:
            self._show_error("현재 비밀번호와 새 비밀번호를 모두 입력해 주세요.")
            return
        if new_pw != confirm_pw:
            self._show_error("새 비밀번호가 서로 일치하지 않습니다.")
            return
        if len(new_pw) < 8:
            self._show_error("새 비밀번호는 8자 이상이어야 합니다.")
            return

        try:
            self._request("POST", "/api/me/change-password", {
                "current_password": current_pw,
                "new_password": new_pw,
                "confirm_new_password": confirm_pw,
            })
        except ProfileAPIError as exc:
            self._show_error(f"비밀번호 변경 실패: {exc}")
            return

        self.editCurrentPassword.clear()
        self.editNewPassword.clear()
        self.editConfirmNewPassword.clear()
        self.errorMsgLabel.setText("비밀번호를 변경했습니다. 다음 로그인부터 새 비밀번호를 사용하세요.")

    def delete_account(self):
        try:
            self.delete_window = QMainWindow()
            ui_path = os.path.join(os.path.dirname(__file__), "deleteAccount.ui")
            uic.loadUi(ui_path, self.delete_window)
            self.delete_window.cancelButton.clicked.connect(self.cancel_delete_account)
            self.delete_window.confirmDeleteButton.clicked.connect(self.confirm_delete_account)
            self.delete_window.confirmPasswordEdit.clear()
            self.delete_window.errorMsgLabel.clear()
            self.delete_window.show()
            self.hide()
        except Exception as exc:
            self._show_error(f"탈퇴 확인 창을 열지 못했습니다: {exc}")

    def confirm_delete_account(self):
        password = self.delete_window.confirmPasswordEdit.text()
        if not password:
            self._show_error("비밀번호를 입력해 주세요.", self.delete_window.errorMsgLabel)
            return

        self.delete_window.confirmDeleteButton.setEnabled(False)
        try:
            self._request("POST", "/api/me/delete-account", {"password": password})
        except ProfileAPIError as exc:
            self.delete_window.confirmDeleteButton.setEnabled(True)
            self._show_error(f"계정 삭제 실패: {exc}", self.delete_window.errorMsgLabel)
            return

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
            from welcome_window import WelcomeWindow
            self.welcome_window = WelcomeWindow()
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
