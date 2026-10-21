import os
import datetime
from PyQt6.QtWidgets import QMainWindow, QPushButton, QFileDialog
from PyQt6 import uic

class WelcomeWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.login_window = parent
        
        # 최초 대시보드 UI 로드
        self.load_dashboard_ui()

    def load_dashboard_ui(self):
        """대시보드 UI(welcomeBack.ui)를 현재 창에 로드하고 시그널을 연결합니다."""
        ui_path = os.path.join(os.path.dirname(__file__), "welcomeBack.ui")
        uic.loadUi(ui_path, self)

        # 1. 오늘의 요일에 맞는 루틴 동적 설정 ("요일: 운동" 형식)
        self.setup_today_routine_text()

        # 2. 7일 전체 보기 토글 버튼 초기화 및 시그널 연결
        if hasattr(self, 'toggleRoutineButton'):
            self.toggleRoutineButton.clicked.connect(self.on_toggle_routine_view)
            # 초기 상태는 꺼짐 (오늘 운동만 보임)
            self.set_routine_view_mode(is_all_visible=False)

        if hasattr(self, 'startWorkoutButton'):
            self.startWorkoutButton.clicked.connect(self.go_to_workout)

        for analyze_btn in ['analyzeNowButton', 'analyzeButton', 'bodyAnalysisButton', 'startAnalysisBannerButton', 'btn_analyze']:
            if hasattr(self, analyze_btn):
                getattr(self, analyze_btn).clicked.connect(self.switch_to_body_upload)
                break
        
        for btn in self.findChildren(QPushButton):
            if 'analyze' in btn.text().lower() or '분석' in btn.text():
                btn.clicked.connect(self.switch_to_body_upload)
                break

        if hasattr(self, 'historyButton'):
            self.historyButton.clicked.connect(self.go_to_history_detail)

        # 📊 My Body Analysis 버튼 클릭 시 리포트 창 오픈 연결
        if hasattr(self, 'myAnalysisButton'):
            self.myAnalysisButton.clicked.connect(self.go_to_body_analysis_report)

        for profile_btn in ['profileButton', 'btn_profile']:
            if hasattr(self, profile_btn):
                getattr(self, profile_btn).clicked.connect(self.go_to_profile)
                break

        for logout_btn in ['logoutButton', 'btn_logout']:
            if hasattr(self, logout_btn):
                getattr(self, logout_btn).clicked.connect(self.logout)
                break
        
        for btn in self.findChildren(QPushButton):
            if 'logout' in btn.text().lower() or '로그아웃' in btn.text():
                btn.clicked.connect(self.logout)
                break

    def setup_today_routine_text(self):
        """오늘 요일에 맞춰 todayDescLabel에 '요일: 운동' 형태로 텍스트를 구성합니다."""
        if not hasattr(self, 'todayDescLabel'):
            return

        routine_schedule = {
            'Wednesday': 'Wednesday: [Day 1] Shoulder Mobility & Release',
            'Thursday': 'Thursday: [Day 2] Core Stability & Balance',
            'Friday': 'Friday: [Day 3] Lower Body Strength & Posture',
            'Saturday': 'Saturday: [Day 4] Active Recovery & Stretching',
            'Sunday': 'Sunday: [Day 5] Upper Body Posture Correction',
            'Monday': 'Monday: [Day 6] Full Body HIIT & Cardio',
            'Tuesday': 'Tuesday: [Day 7] Weekly Review & Flexibility Test'
        }

        current_day_str = datetime.datetime.now().strftime('%A')
        today_text = routine_schedule.get(current_day_str, 'Wednesday: [Day 1] Shoulder Mobility & Release')
        self.todayDescLabel.setText(today_text)

    def on_toggle_routine_view(self):
        """토글 버튼이 클릭되었을 때 호출되어 7일 전체 보기 모드를 켜거나 끕니다."""
        is_checked = self.toggleRoutineButton.isChecked()
        if is_checked:
            self.toggleRoutineButton.setText("📅 View All Days (On)")
            self.set_routine_view_mode(is_all_visible=True)
        else:
            self.toggleRoutineButton.setText("📅 View All Days (Off)")
            self.set_routine_view_mode(is_all_visible=False)

    def set_routine_view_mode(self, is_all_visible: bool):
        """토글 상태에 따라 오늘 요일 운동 / 7일 전체 운동 목록의 가시성, 카드 높이, 하단 버튼 위치를 동적으로 조절합니다."""
        day_items = ['day1Item', 'day2Item', 'day3Item', 'day4Item', 'day5Item', 'day6Item', 'day7Item']
        
        if is_all_visible:
            # 토글 켜짐: 카드 높이 310, 전체 목록 표시, 하단 버튼들을 스크린샷 원본 위치(Y=510, Y=562)로 배치
            if hasattr(self, 'todayRoutineCard'):
                self.todayRoutineCard.setFixedHeight(310)
            if hasattr(self, 'allDaysContainerLabel'):
                self.allDaysContainerLabel.show()
            for item_name in day_items:
                if hasattr(self, item_name):
                    getattr(self, item_name).show()
            
            if hasattr(self, 'historyButton'):
                self.historyButton.setGeometry(30, 510, 460, 44)
            if hasattr(self, 'myAnalysisButton'):
                self.myAnalysisButton.setGeometry(30, 562, 460, 44)
        else:
            # 토글 꺼짐: 카드 높이 140, 전체 목록 숨김, 하단 버튼들을 오늘의 운동 카드 바로 밑(Y=340, Y=394)으로 밀착
            if hasattr(self, 'todayRoutineCard'):
                self.todayRoutineCard.setFixedHeight(140)
            if hasattr(self, 'allDaysContainerLabel'):
                self.allDaysContainerLabel.hide()
            for item_name in day_items:
                if hasattr(self, item_name):
                    getattr(self, item_name).hide()
            
            if hasattr(self, 'historyButton'):
                self.historyButton.setGeometry(30, 340, 460, 44)
            if hasattr(self, 'myAnalysisButton'):
                self.myAnalysisButton.setGeometry(30, 394, 460, 44)

    def switch_to_body_upload(self):
        ui_path = os.path.join(os.path.dirname(__file__), "bodyUpload.ui")
        uic.loadUi(ui_path, self)

        self.uploaded_photos = {1: None, 2: None, 3: None, 4: None}

        if hasattr(self, 'uploadPhoto1Button'):
            self.uploadPhoto1Button.clicked.connect(lambda: self.select_photo(1, self.uploadPhoto1Button))
        if hasattr(self, 'uploadPhoto2Button'):
            self.uploadPhoto2Button.clicked.connect(lambda: self.select_photo(2, self.uploadPhoto2Button))
        if hasattr(self, 'uploadPhoto3Button'):
            self.uploadPhoto3Button.clicked.connect(lambda: self.select_photo(3, self.uploadPhoto3Button))
        if hasattr(self, 'uploadPhoto4Button'):
            self.uploadPhoto4Button.clicked.connect(lambda: self.select_photo(4, self.uploadPhoto4Button))

        if hasattr(self, 'submitAnalysisButton'):
            self.submitAnalysisButton.clicked.connect(self.submit_analysis)

        if hasattr(self, 'backDashboardButton'):
            self.backDashboardButton.clicked.connect(self.load_dashboard_ui)

    def select_photo(self, photo_index, button_widget):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Body Photo", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.uploaded_photos[photo_index] = file_name
            base_name = os.path.basename(file_name)
            if photo_index == 1:
                button_widget.setText(f"✔ Front\n({base_name[:10]}...)")
            elif photo_index == 2:
                button_widget.setText(f"✔ Left Side\n({base_name[:10]}...)")
            elif photo_index == 3:
                button_widget.setText(f"✔ Right Side\n({base_name[:10]}...)")
            elif photo_index == 4:
                button_widget.setText(f"✔ Back\n({base_name[:10]}...)")

    def submit_analysis(self):
        missing_photos = [idx for idx, path in self.uploaded_photos.items() if path is None]
        
        if missing_photos:
            if hasattr(self, 'errorMsgLabel'):
                self.errorMsgLabel.setText("⚠️ Please upload all 4 photos for precise AI assessment.")
            return

        if hasattr(self, 'errorMsgLabel'):
            self.errorMsgLabel.setText("")
        
        # 🚀 수정 완료: 단순히 ui 로드만 하던 것을 BodyAnalysisLoadingWindow 창을 띄우도록 연동합니다.
        from bodyAnalysisLoading import BodyAnalysisLoadingWindow
        self.loading_window = BodyAnalysisLoadingWindow(self)
        self.loading_window.show()
        self.hide()

    def go_to_body_analysis_report(self):
        from body_analysis_report import BodyAnalysisReportWindow
        self.report_window = BodyAnalysisReportWindow(self)
        self.report_window.show()
        self.hide()

    def go_to_workout(self):
        from workout_window import WorkoutWindow
        self.workout_window = WorkoutWindow(self)
        self.workout_window.show()
        self.hide()

    def go_to_history_detail(self):
        from history_window import HistoryDetailWindow
        self.history_detail_window = HistoryDetailWindow(self)
        self.history_detail_window.show()
        self.hide()

    def go_to_profile(self):
        from profile_window import ProfileSettingsWindow
        self.profile_window = ProfileSettingsWindow(self)
        self.profile_window.show()
        self.hide()

    def logout(self):
        if self.login_window is not None:
            self.login_window.show()
        else:
            from login_window import LoginWindow
            self.new_login = LoginWindow()
            self.new_login.show()
        self.close()

    def closeEvent(self, event):
        if self.login_window is not None:
            self.login_window.close()
        event.accept()
