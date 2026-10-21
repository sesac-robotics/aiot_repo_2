import os
from PyQt6.QtWidgets import QMainWindow, QPushButton
from PyQt6 import uic

class ProfileSettingsWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.welcome_window = parent
        
        # profileSettings.ui 파일 로드
        ui_path = os.path.join(os.path.dirname(__file__), "profileSettings.ui")
        uic.loadUi(ui_path, self)

        # 버튼 시그널 연결
        if hasattr(self, 'saveButton'):
            self.saveButton.clicked.connect(self.save_profile_changes)

        if hasattr(self, 'changePasswordButton'):
            self.changePasswordButton.clicked.connect(self.change_password)

        if hasattr(self, 'deleteAccountButton'):
            self.deleteAccountButton.clicked.connect(self.delete_account)

        if hasattr(self, 'backDashboardButton'):
            self.backDashboardButton.clicked.connect(self.go_back_to_dashboard)

    def save_profile_changes(self):
        """프로필 변경사항 저장 로직"""
        nickname = self.editNickname.text().strip() if hasattr(self, 'editNickname') else ""
        if not nickname:
            if hasattr(self, 'errorMsgLabel'):
                self.errorMsgLabel.setText("⚠️ Nickname cannot be empty.")
            return

        if hasattr(self, 'errorMsgLabel'):
            self.errorMsgLabel.setText("")
        print("Profile changes saved successfully.")

    def change_password(self):
        """비밀번호 변경 로직"""
        current_pw = self.editCurrentPassword.text() if hasattr(self, 'editCurrentPassword') else ""
        new_pw = self.editNewPassword.text() if hasattr(self, 'editNewPassword') else ""

        if not current_pw or not new_pw:
            if hasattr(self, 'errorMsgLabel'):
                self.errorMsgLabel.setText("⚠️ Please fill in both password fields.")
            return

        if hasattr(self, 'errorMsgLabel'):
            self.errorMsgLabel.setText("")
        print("Password changed successfully.")

    def delete_account(self):
        """회원 탈퇴 창(deleteAccount.ui)을 띄우는 로직"""
        try:
            self.delete_window = QMainWindow()
            ui_path = os.path.join(os.path.dirname(__file__), "deleteAccount.ui")
            uic.loadUi(ui_path, self.delete_window)
            
            # deleteAccount.ui 내의 Cancel 버튼(또는 취소 버튼)을 찾아 클릭 시 팝업을 닫고 profileSettings 복귀
            for btn in self.delete_window.findChildren(QPushButton):
                text = btn.text().lower()
                # 오브젝트 이름이 cancelButton이거나 텍스트에 cancel/취소가 포함된 경우
                if btn.objectName() == 'cancelButton' or 'cancel' in text or '취소' in text:
                    btn.clicked.connect(self.cancel_delete_account)
                    break

            self.delete_window.show()
            self.hide() # 프로필 설정창 숨기기
        except Exception as e:
            if hasattr(self, 'errorMsgLabel'):
                self.errorMsgLabel.setText("⚠️ Failed to open delete account window.")
            print(f"Error opening deleteAccount.ui: {e}")

    def cancel_delete_account(self):
        """회원 탈퇴 취소 시 팝업을 닫고 프로필 설정 창(self)을 다시 표시"""
        if hasattr(self, 'delete_window'):
            self.delete_window.close()
        self.show()

    def go_back_to_dashboard(self):
        """대시보드(WelcomeWindow)로 복귀"""
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