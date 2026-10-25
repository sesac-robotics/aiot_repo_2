"""로그인·회원가입 입력을 검증하고 요청 작업을 제출합니다. 인증 성공 후 대시보드로 이동하며 요청 중에는 중복 제출을 막습니다."""

import os
import re

import requests
from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow

from GUI.FitHelper import login_client
from GUI.FitHelper.request_worker import RequestTask

# ============================================================
# AIFit Server 설정
# ============================================================

AIFIT_SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL", "http://127.0.0.1:8000"
).rstrip("/")

REGISTER_URL = f"{AIFIT_SERVER_URL}/api/auth/register"


# ============================================================
# 회원가입 완료 화면
# ============================================================


class RegistrationCompleteWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.login_window = parent

        ui_path = os.path.join(
            os.path.dirname(__file__), "registrationComplete.ui"
        )
        uic.loadUi(ui_path, self)

        if hasattr(self, "goToLoginButton"):
            self.goToLoginButton.clicked.connect(self.return_to_login)

    def return_to_login(self):
        if self.login_window is not None:
            self.login_window.show()

        self.close()

    def closeEvent(self, event):
        if self.login_window is not None:
            self.login_window.show()

        event.accept()


# ============================================================
# 회원가입 화면
# ============================================================


class RegistrationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.login_window = parent
        self.complete_window = None

        ui_path = os.path.join(os.path.dirname(__file__), "registration.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, "backButton"):
            self.backButton.clicked.connect(self.return_to_login)

        if hasattr(self, "registerButton"):
            self.registerButton.clicked.connect(self.handle_registration)

    # --------------------------------------------------------
    # 오류 메시지 출력
    # --------------------------------------------------------

    def show_error(self, message):
        if hasattr(self, "errorMsgLabel"):
            self.errorMsgLabel.setText(message)

        print(f"[회원가입 오류] {message}", flush=True)

    # --------------------------------------------------------
    # 회원가입 처리
    # --------------------------------------------------------

    def handle_registration(self):
        # UI 입력값 읽기
        username = (
            self.regUsernameEdit.text().strip()
            if hasattr(self, "regUsernameEdit")
            else ""
        )

        email = (
            self.regEmailEdit.text().strip()
            if hasattr(self, "regEmailEdit")
            else ""
        )

        password = (
            self.regPasswordEdit.text()
            if hasattr(self, "regPasswordEdit")
            else ""
        )

        re_password = (
            self.regPasswordEdit_2.text()
            if hasattr(self, "regPasswordEdit_2")
            else ""
        )

        nickname = (
            self.editNickname.text().strip()
            if hasattr(self, "editNickname")
            else ""
        )

        gender = (
            self.comboGender.currentText().strip()
            if hasattr(self, "comboGender")
            else ""
        )

        age_text = (
            self.editAge.text().strip() if hasattr(self, "editAge") else ""
        )

        height_text = (
            self.editHeight.text().strip()
            if hasattr(self, "editHeight")
            else ""
        )

        weight_text = (
            self.editWeight.text().strip()
            if hasattr(self, "editWeight")
            else ""
        )

        if hasattr(self, "errorMsgLabel"):
            self.errorMsgLabel.clear()

        # 1. 필수 입력값 확인
        required_fields = {
            "아이디": username,
            "이메일": email,
            "비밀번호": password,
            "비밀번호 확인": re_password,
            "닉네임": nickname,
            "나이": age_text,
            "키": height_text,
            "몸무게": weight_text,
        }

        missing_fields = [
            name for name, value in required_fields.items() if not value
        ]

        if missing_fields:
            self.show_error(
                "다음 항목을 입력해 주세요: " + ", ".join(missing_fields)
            )
            return

        # 2. 이메일 기본 형식 확인
        email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

        if not re.match(email_regex, email):
            self.show_error("올바른 이메일 주소를 입력해 주세요.")
            return

        # 3. 비밀번호 정책 확인
        password_regex = (
            r"^(?=.*[A-Za-z])"
            r"(?=.*\d)"
            r"(?=.*[@$!%*#?&])"
            r"[A-Za-z\d@$!%*#?&]{8,}$"
        )

        if not re.match(password_regex, password):
            self.show_error(
                "비밀번호는 영문자, 숫자, 특수문자를 포함한 "
                "8자 이상이어야 합니다."
            )
            return

        # 4. 비밀번호 일치 여부 확인
        if password != re_password:
            self.show_error("비밀번호가 일치하지 않습니다.")
            return

        # 5. 숫자 입력값 검증
        try:
            age = int(age_text)
            height = float(height_text)
            weight = float(weight_text)

        except ValueError:
            self.show_error("나이는 정수, 키와 몸무게는 숫자로 입력해 주세요.")
            return

        if age <= 0:
            self.show_error("나이는 1 이상의 정수여야 합니다.")
            return

        if height <= 0 or weight <= 0:
            self.show_error("키와 몸무게는 0보다 큰 숫자여야 합니다.")
            return

        # NaN 및 무한대 입력 방지
        import math

        if not math.isfinite(height) or not math.isfinite(weight):
            self.show_error("키와 몸무게를 올바르게 입력해 주세요.")
            return

        # ----------------------------------------------------
        # 6. 서버에 전달할 JSON 데이터
        # ----------------------------------------------------

        payload = {
            "username": username,
            "email_address": email,
            "password": password,
            "nickname": nickname,
            "gender": gender.lower(),
            "age": age,
            "height": height,
            "weight": weight,
        }

        self.registerButton.setEnabled(False)
        self._registration_task = RequestTask(
            self,
            lambda: self._register_request(payload),
            self._registration_completed,
            self._registration_failed,
        )
        self._registration_task.start()

    @staticmethod
    def _register_request(payload):
        response = requests.post(REGISTER_URL, json=payload, timeout=10)
        response.raise_for_status()
        return True

    def _registration_failed(self, error):
        self.registerButton.setEnabled(True)
        self.show_error(
            "회원가입에 실패했습니다. 입력값과 서버 상태를 확인해 주세요."
        )

    def _registration_completed(self, result):
        self.registerButton.setEnabled(True)
        parent_login = self.login_window

        self.login_window = None
        self.close()

        self.complete_window = RegistrationCompleteWindow(parent_login)
        self.complete_window.show()

    # --------------------------------------------------------
    # 로그인 화면으로 돌아가기
    # --------------------------------------------------------

    def return_to_login(self):
        if self.login_window is not None:
            self.login_window.show()

        self.close()

    def closeEvent(self, event):
        if self.login_window is not None:
            self.login_window.show()

        event.accept()


# ============================================================
# 로그인 화면
# ============================================================


class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.welcome_window = None
        self.reg_window = None

        ui_path = os.path.join(os.path.dirname(__file__), "mainLoginWindow.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, "loginButton"):
            self.loginButton.clicked.connect(self.handle_login)

        if hasattr(self, "registerLinkButton"):
            self.registerLinkButton.clicked.connect(self.go_to_registration)

    # --------------------------------------------------------
    # 로그인 처리
    # --------------------------------------------------------

    def closeEvent(self, event):
        task = getattr(self, "_login_task", None)
        if task is not None:
            task.cancel()
        login_client.logout()
        event.accept()

    def handle_login(self):
        username = (
            self.usernameEdit.text().strip()
            if hasattr(self, "usernameEdit")
            else ""
        )

        password = (
            self.passwordEdit.text() if hasattr(self, "passwordEdit") else ""
        )

        if hasattr(self, "errorMsgLabel"):
            self.errorMsgLabel.clear()

        if not username or not password:
            if hasattr(self, "errorMsgLabel"):
                self.errorMsgLabel.setText("아이디와 비밀번호를 입력해 주세요.")
            return

        self.loginButton.setEnabled(False)
        self._login_task = RequestTask(
            self,
            lambda: login_client.login(username, password),
            self._login_completed,
            self._login_failed,
        )
        self._login_task.start()

    def _login_failed(self, error):
        self.loginButton.setEnabled(True)
        self.errorMsgLabel.setText("로그인 요청을 완료하지 못했습니다.")

    def _login_completed(self, success):
        self.loginButton.setEnabled(True)
        if not success:
            self.errorMsgLabel.setText("로그인 또는 인증 확인에 실패했습니다.")
            return
        self.errorMsgLabel.clear()
        from GUI.FitHelper.welcome_window import WelcomeWindow

        self.welcome_window = WelcomeWindow(self)
        self.welcome_window.show()
        self.hide()

    # --------------------------------------------------------
    # 회원가입 화면으로 이동
    # --------------------------------------------------------

    def go_to_registration(self):
        self.reg_window = RegistrationWindow(self)
        self.reg_window.show()
        self.hide()
