import json
import os
import urllib.error
import urllib.request

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QFrame, QComboBox,
    QSpinBox, QDoubleSpinBox
)

SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://192.168.15.118:8000").rstrip("/")
REQUEST_TIMEOUT = 8

def request_json(method, path, payload=None, token=None):
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        f"{SERVER_URL}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            result = json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            result = {"detail": f"HTTP 오류 ({exc.code})"}
        return exc.code, result
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(
            f"AIFit Server와 통신할 수 없습니다.\n주소: {SERVER_URL}\n"
            "서버 실행 상태, IP 주소, 포트 및 방화벽을 확인해 주세요."
        ) from exc

def response_message(body):
    detail = body.get("detail", "요청을 처리하지 못했습니다.")
    if isinstance(detail, list):
        return "\n".join(f"- {item.get('msg', '입력값 오류')}" for item in detail)
    return str(detail)

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Fitness Hub - Login & Registration")
        self.resize(760, 820)
        self.access_token = None
        self.user_info = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.login_page = self._build_login_page()
        self.register_page = self._build_register_page()
        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.register_page)
        self.stack.setCurrentWidget(self.login_page)

        self.setStyleSheet("""
            QWidget { font-size: 14px; }
            QMainWindow { background: #f2f4f7; }
            QFrame#panel { background: white; border: 1px solid #dce2ea; border-radius: 16px; }
            QLabel#heading { font-size: 26px; font-weight: 700; color: #078ac5; }
            QLabel#subtitle { color: #526477; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 10px; border: 1px solid #d4dce6; border-radius: 8px;
                background: #fbfcfe; min-height: 22px;
            }
            QPushButton { padding: 12px; border-radius: 9px; font-weight: 600; }
            QPushButton#primary { background: #078fc9; color: white; border: none; }
            QPushButton#secondary { background: white; color: #536579; border: 1px solid #d4dce6; }
        """)

    def _build_login_page(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(45, 35, 45, 35)
        root.addStretch()
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 30, 32, 30)

        title = QLabel("🔥  AI Fitness Hub")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Login")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Username")
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Username", self.login_username)
        form.addRow("Password", self.login_password)
        layout.addLayout(form)

        self.login_status = QLabel("")
        self.login_status.setWordWrap(True)
        layout.addWidget(self.login_status)

        login_button = QPushButton("Login")
        login_button.setObjectName("primary")
        login_button.clicked.connect(self.handle_login)
        register_button = QPushButton("Create Account & Profile")
        register_button.setObjectName("secondary")
        register_button.clicked.connect(lambda: self.stack.setCurrentWidget(self.register_page))
        layout.addWidget(login_button)
        layout.addWidget(register_button)
        root.addWidget(panel)
        root.addStretch()
        return page

    def _build_register_page(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(28, 18, 28, 18)
        title = QLabel("🔥  Create Account & Profile")
        title.setObjectName("heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(28, 24, 28, 24)
        form = QFormLayout()
        form.setVerticalSpacing(10)

        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("Enter username")
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("name@example.com")
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password (at least 8 characters)")
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_confirm_password = QLineEdit()
        self.reg_confirm_password.setPlaceholderText("Confirm password")
        self.reg_confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_nickname = QLineEdit()
        self.reg_nickname.setPlaceholderText("Nickname")
        self.reg_gender = QComboBox()
        self.reg_gender.addItem("Male", "male")
        self.reg_gender.addItem("Female", "female")
        self.reg_gender.addItem("Other", "other")
        self.reg_age = QSpinBox()
        self.reg_age.setRange(1, 120)
        self.reg_age.setValue(25)
        self.reg_height = QDoubleSpinBox()
        self.reg_height.setRange(0.1, 300.0)
        self.reg_height.setDecimals(1)
        self.reg_height.setValue(170.0)
        self.reg_height.setSuffix(" cm")
        self.reg_weight = QDoubleSpinBox()
        self.reg_weight.setRange(0.1, 500.0)
        self.reg_weight.setDecimals(1)
        self.reg_weight.setValue(65.0)
        self.reg_weight.setSuffix(" kg")

        for label, widget in [
            ("Username (Login ID)", self.reg_username),
            ("Email Address", self.reg_email),
            ("Password", self.reg_password),
            ("Re-Enter Password", self.reg_confirm_password),
            ("Nickname", self.reg_nickname), ("Gender", self.reg_gender),
            ("Age", self.reg_age), ("Height", self.reg_height), ("Weight", self.reg_weight)
        ]:
            form.addRow(label, widget)
        layout.addLayout(form)

        self.register_status = QLabel("")
        self.register_status.setWordWrap(True)
        layout.addWidget(self.register_status)
        submit_button = QPushButton("✨ Complete Registration")
        submit_button.setObjectName("primary")
        submit_button.clicked.connect(self.handle_register)
        back_button = QPushButton("Back to Login")
        back_button.setObjectName("secondary")
        back_button.clicked.connect(self.back_to_login)
        layout.addWidget(submit_button)
        layout.addWidget(back_button)
        root.addWidget(panel)
        return page

    def handle_register(self):
        username = self.reg_username.text().strip()
        email = self.reg_email.text().strip()
        password = self.reg_password.text()
        confirm = self.reg_confirm_password.text()
        nickname = self.reg_nickname.text().strip()

        missing = [label for label, value in [
            ("Username", username), ("Email Address", email), ("Password", password),
            ("Re-Enter Password", confirm), ("Nickname", nickname)
        ] if not value]
        if missing:
            QMessageBox.warning(self, "Input Required", "다음 항목을 입력해 주세요:\n" + "\n".join(missing))
            return
        if password != confirm:
            QMessageBox.warning(self, "Password Mismatch", "비밀번호가 일치하지 않습니다. 다시 입력해 주세요.")
            self.reg_confirm_password.clear()
            self.reg_confirm_password.setFocus()
            return
        if len(password) < 8:
            QMessageBox.warning(self, "Password Too Short", "비밀번호는 8자 이상 입력해 주세요.")
            return
        if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            QMessageBox.warning(self, "Invalid Email", "올바른 이메일 주소를 입력해 주세요.")
            self.reg_email.setFocus()
            return

        payload = {
            "username": username,
            "password": password,
            "email_address": email,
            "nickname": nickname,
            "gender": self.reg_gender.currentData(),
            "age": self.reg_age.value(),
            "height": self.reg_height.value(),
            "weight": self.reg_weight.value(),
        }
        try:
            code, body = request_json("POST", "/api/auth/register", payload)
        except ConnectionError as exc:
            self.register_status.setText(str(exc))
            QMessageBox.critical(self, "Connection Error", str(exc))
            return

        if code == 201 and body.get("success"):
            QMessageBox.information(self, "Registration Complete",
                f"회원가입이 완료되었습니다.\n사용자 ID: {body['user_id']}\n로그인해 주세요.")
            self.login_username.setText(username)
            self.login_password.clear()
            self.login_status.setText("회원가입 성공. 로그인해 주세요.")
            self.stack.setCurrentWidget(self.login_page)
        else:
            message = response_message(body)
            self.register_status.setText(message)
            QMessageBox.warning(self, "Registration Failed", message)

    def handle_login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text()
        if not username or not password:
            QMessageBox.warning(self, "Input Required", "username과 password를 모두 입력해 주세요.")
            return

        try:
            code, body = request_json("POST", "/api/auth/login", {
                "username": username, "password": password
            })
        except ConnectionError as exc:
            self.login_status.setText(str(exc))
            QMessageBox.critical(self, "Connection Error", str(exc))
            return

        if code != 200 or not body.get("success"):
            message = response_message(body)
            self.login_status.setText(message)
            QMessageBox.warning(self, "Login Failed", message)
            return

        self.access_token = body["access_token"]
        self.user_info = body["user"]

        # Demonstrate the full JWT flow: call a protected API with Bearer token.
        try:
            profile_code, profile_body = request_json(
                "GET", "/api/me", token=self.access_token
            )
        except ConnectionError as exc:
            QMessageBox.critical(self, "Connection Error", str(exc))
            self.login_status.setText(str(exc))
            return

        if profile_code == 200 and profile_body.get("success"):
            user = profile_body["user"]
            self.login_status.setText(
                f"로그인 성공: {user['nickname']}\n"
                "JWT 검증 성공: 보호된 API 접근 허용"
            )
            QMessageBox.information(
                self, "Login Success",
                f"{user['nickname']}님, 로그인에 성공했습니다.\n"
                "JWT 발급 및 보호된 API의 토큰 검증까지 성공했습니다."
            )
            print("[AUTH] Login successful; JWT sent as Authorization: Bearer.")
            print("[AUTH] Protected GET /api/me returned 200.")
        else:
            message = response_message(profile_body)
            self.login_status.setText("로그인은 성공했지만 JWT 검증에 실패했습니다.\n" + message)
            QMessageBox.warning(self, "JWT Validation Failed", message)
            if profile_code == 401:
                self.access_token = None

    def back_to_login(self):
        self.stack.setCurrentWidget(self.login_page)
