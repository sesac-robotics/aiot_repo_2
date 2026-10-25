"""QApplication, 인증 세션 컨텍스트와 로그인 화면을 조립하는 GUI 진입점입니다. 컴포넌트는 실행 위치와 무관한 패키지 경로로 가져옵니다."""

import sys

from PyQt6.QtWidgets import QApplication

from GUI.FitHelper.application_context import ApplicationContext
from GUI.FitHelper.login_window import LoginWindow


def main():
    app = QApplication(sys.argv)
    app.aifit_context = ApplicationContext()
    window = LoginWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
