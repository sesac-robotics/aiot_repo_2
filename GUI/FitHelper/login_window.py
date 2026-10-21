
import os
import re
import requests

from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic

import login_client


# ============================================================
# AIFit Server 설정
# ============================================================

AIFIT_SERVER_URL = os.getenv(
    "AIFIT_SERVER_URL",
    "http://192.168.15.118:8000"
).rstrip("/")

REGISTER_URL = f"{AIFIT_SERVER_URL}/api/register"


# ============================================================
# 회원가입 완료 화면
# ============================================================

class RegistrationCompleteWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.login_window = parent

        ui_path = os.path.join(
            os.path.dirname(__file__),
            "registrationComplete.ui"
        )
        uic.loadUi(ui_path, self)

        if hasattr(self, "goToLoginButton"):
            self.goToLoginButton.clicked.connect(
                self.return_to_login
            )

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

        ui_path = os.path.join(
            os.path.dirname(__file__),
            "registration.ui"
        )
        uic.loadUi(ui_path, self)

        if hasattr(self, "backButton"):
            self.backButton.clicked.connect(
                self.return_to_login
            )

        if hasattr(self, "registerButton"):
            self.registerButton.clicked.connect(
                self.handle_registration
            )

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
            self.editAge.text().strip()
            if hasattr(self, "editAge")
            else ""
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
            name
            for name, value in required_fields.items()
            if not value
        ]

        if missing_fields:
            self.show_error(
                "다음 항목을 입력해 주세요: "
                + ", ".join(missing_fields)
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
            self.show_error(
                "나이는 정수, 키와 몸무게는 숫자로 입력해 주세요."
            )
            return

        if age <= 0:
            self.show_error("나이는 1 이상의 정수여야 합니다.")
            return

        if height <= 0 or weight <= 0:
            self.show_error(
                "키와 몸무게는 0보다 큰 숫자여야 합니다."
            )
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
            "userID": username,
            "email": email,
            "password": password,
            "confirmPassword": re_password,
            "nickname": nickname,
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
        }

        print(
            f"[회원가입 요청] 서버 주소: {REGISTER_URL}",
            flush=True
        )
        print(
            f"[회원가입 요청] 사용자 ID: {username}",
            flush=True
        )

        # ----------------------------------------------------
        # 7. AIFit Server에 회원가입 요청
        # ----------------------------------------------------

        try:
            response = requests.post(
                REGISTER_URL,
                json=payload,
                timeout=10
            )

            print(
                f"[회원가입 응답] HTTP {response.status_code}",
                flush=True
            )

            # HTTP 오류 응답 처리
            if not response.ok:
                error_message = (
                    f"서버에서 회원가입을 거부했습니다. "
                    f"(HTTP {response.status_code})"
                )

                try:
                    result = response.json()

                    # FastAPI의 일반적인 오류 응답 형식
                    if isinstance(result, dict):
                        detail = result.get("detail")

                        if detail:
                            error_message = (
                                f"회원가입 실패: {detail}"
                            )

                except ValueError:
                    pass

                self.show_error(error_message)

                print(
                    f"[회원가입 실패] 응답 내용: "
                    f"{response.text[:500]}",
                    flush=True
                )
                return

            # 성공 응답 본문은 서버 응답 형식에 따라 확인
            print(
                f"[회원가입 성공] 서버가 HTTP "
                f"{response.status_code} 응답을 반환했습니다.",
                flush=True
            )

        except requests.Timeout:
            self.show_error(
                "서버 응답 시간이 초과되었습니다. "
                "서버 상태와 네트워크를 확인해 주세요."
            )
            return

        except requests.ConnectionError as e:
            self.show_error(
                "서버에 연결할 수 없습니다. "
                "AIFit Server의 IP 주소와 포트를 확인해 주세요."
            )
            print(
                f"[회원가입 연결 오류] {e}",
                flush=True
            )
            return

        except requests.RequestException as e:
            self.show_error(
                "회원가입 요청 중 통신 오류가 발생했습니다."
            )
            print(
                f"[회원가입 통신 오류] {e}",
                flush=True
            )
            return

        # ----------------------------------------------------
        # 8. 서버 응답이 성공인 경우에만 완료 화면 표시
        # ----------------------------------------------------

        parent_login = self.login_window

        self.login_window = None
        self.close()

        self.complete_window = RegistrationCompleteWindow(
            parent_login
        )
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

        ui_path = os.path.join(
            os.path.dirname(__file__),
            "mainLoginWindow.ui"
        )
        uic.loadUi(ui_path, self)

        if hasattr(self, "loginButton"):
            self.loginButton.clicked.connect(
                self.handle_login
            )

        if hasattr(self, "registerLinkButton"):
            self.registerLinkButton.clicked.connect(
                self.go_to_registration
            )

    # --------------------------------------------------------
    # 로그인 처리
    # --------------------------------------------------------

    def handle_login(self):
        username = (
            self.usernameEdit.text().strip()
            if hasattr(self, "usernameEdit")
            else ""
        )

        password = (
            self.passwordEdit.text()
            if hasattr(self, "passwordEdit")
            else ""
        )

        if hasattr(self, "errorMsgLabel"):
            self.errorMsgLabel.clear()

        if not username or not password:
            if hasattr(self, "errorMsgLabel"):
                self.errorMsgLabel.setText(
                    "아이디와 비밀번호를 입력해 주세요."
                )
            return

        print(
            f"[로그인 요청] 사용자 ID: {username}",
            flush=True
        )

        try:
            # login_client.py에서 서버 로그인 API 호출
            is_success = login_client.login(
                username,
                password
            )

        except Exception as e:
            print(
                f"[로그인 오류] {type(e).__name__}: {e}",
                flush=True
            )

            if hasattr(self, "errorMsgLabel"):
                self.errorMsgLabel.setText(
                    "로그인 중 오류가 발생했습니다. "
                    "서버 연결 및 클라이언트 코드를 확인해 주세요."
                )
            return

        if is_success:
            print("[로그인 성공] 인증 완료", flush=True)

            if hasattr(self, "errorMsgLabel"):
                self.errorMsgLabel.clear()

            from welcome_window import WelcomeWindow

            self.welcome_window = WelcomeWindow(self)
            self.welcome_window.show()
            self.hide()

        else:
            print(
                "[로그인 실패] 인증에 실패했습니다.",
                flush=True
            )

            if hasattr(self, "errorMsgLabel"):
                self.errorMsgLabel.setText(
                    "아이디 또는 비밀번호가 올바르지 않습니다."
                )

    # --------------------------------------------------------
    # 회원가입 화면으로 이동
    # --------------------------------------------------------

    def go_to_registration(self):
        self.reg_window = RegistrationWindow(self)
        self.reg_window.show()
        self.hide()
