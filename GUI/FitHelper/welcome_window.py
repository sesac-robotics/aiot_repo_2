"""로그인 후 대시보드, 체형 분석 요약과 하위 화면 이동을 담당합니다. 분석 ID 캐시와 서버의 분석 완료 상태를 구분합니다."""

import datetime
import os
import sys
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QPushButton

# The shared authenticated API client is stored beside this project folder.
from GUI.FitHelper import body_analysis_client as body_api
from GUI.FitHelper.body_analysis_cache import (
    clear_body_id,
    format_area_summary,
    load_body_id,
    save_body_id,
)
from GUI.FitHelper.body_analysis_client import BodyAnalysisAPIError


class _LatestAnalysisThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, object)

    def __init__(self, body_id, parent=None):
        super().__init__(parent)
        self.body_id = body_id

    def run(self):
        try:
            self.succeeded.emit(body_api.get_analysis(self.body_id))
        except BodyAnalysisAPIError as exc:
            self.failed.emit(str(exc), exc.status_code)
        except Exception as exc:
            self.failed.emit(str(exc), None)


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
        self.completed_body_id = load_body_id(self.login_window)
        self.latest_routine = None
        # Show the analysis card only after the server confirms a completed record.
        self.has_completed_analysis = False
        self._latest_analysis_thread = None
        self.analysisNudgeCard.show()
        self.bodyAnalysisSummaryCard.hide()

        # 1. 오늘의 요일에 맞는 루틴 동적 설정 ("요일: 운동" 형식)
        self.setup_today_routine_text()

        # 2. 7일 전체 보기 토글 버튼 초기화 및 시그널 연결
        if hasattr(self, "toggleRoutineButton"):
            self.toggleRoutineButton.clicked.connect(
                self.on_toggle_routine_view
            )
            # 초기 상태는 꺼짐 (오늘 운동만 보임)
            self.set_routine_view_mode(is_all_visible=False)

        if hasattr(self, "startWorkoutButton"):
            self.startWorkoutButton.clicked.connect(self.go_to_workout)

        for analyze_btn in [
            "analyzeNowButton",
            "analyzeButton",
            "bodyAnalysisButton",
            "startAnalysisBannerButton",
            "btn_analyze",
        ]:
            if hasattr(self, analyze_btn):
                getattr(self, analyze_btn).clicked.connect(
                    self.switch_to_body_upload
                )
                break

        if hasattr(self, "historyButton"):
            self.historyButton.clicked.connect(self.go_to_history_detail)

        for profile_btn in ["profileButton", "btn_profile"]:
            if hasattr(self, profile_btn):
                getattr(self, profile_btn).clicked.connect(self.go_to_profile)
                break

        for logout_btn in ["logoutButton", "btn_logout"]:
            if hasattr(self, logout_btn):
                getattr(self, logout_btn).clicked.connect(self.logout)
                break

        for btn in self.findChildren(QPushButton):
            if "logout" in btn.text().lower() or "로그아웃" in btn.text():
                btn.clicked.connect(self.logout)
                break

        if self.completed_body_id:
            self._latest_analysis_thread = _LatestAnalysisThread(
                self.completed_body_id, self
            )
            self._latest_analysis_thread.succeeded.connect(
                lambda response: self.display_completed_analysis(
                    response, self.completed_body_id
                )
            )
            self._latest_analysis_thread.failed.connect(
                self._latest_analysis_failed
            )
            self._latest_analysis_thread.start()

    def setup_today_routine_text(self):
        """Display today's entry from the most recently fetched server routine."""
        if not hasattr(self, "todayDescLabel"):
            return
        today_no = datetime.date.today().isoweekday()
        self.todayDescLabel.setText(self._format_routine_day(today_no))

    def _format_routine_day(self, day_no):
        day_names = {
            1: "월요일",
            2: "화요일",
            3: "수요일",
            4: "목요일",
            5: "금요일",
            6: "토요일",
            7: "일요일",
        }
        routine = self.latest_routine
        days = routine.get("days") if isinstance(routine, dict) else None
        day = next(
            (
                item
                for item in days or []
                if isinstance(item, dict) and item.get("day_no") == day_no
            ),
            None,
        )
        if day is None:
            return f"{day_names[day_no]} · 서버 루틴 정보 없음"
        exercises = day.get("exercises", [])
        if not exercises:
            return f"{day_names[day_no]} · 휴식"
        entries = []
        for exercise in exercises:
            if not isinstance(exercise, dict):
                continue
            name = exercise.get("exercise_name") or "운동"
            sets = exercise.get("sets")
            reps = exercise.get("reps")
            duration = exercise.get("target_duration_sec")
            details = []
            if sets is not None:
                details.append(f"{sets}세트")
            if duration is not None:
                details.append(f"{duration}초 유지")
            elif reps is not None:
                details.append(f"{reps}회")
            entries.append(
                str(name) + (" · " + " · ".join(details) if details else "")
            )
        return f"{day_names[day_no]} · " + (
            " / ".join(entries) if entries else "운동 정보 없음"
        )

    def _apply_routine_to_dashboard(self):
        for day_no in range(1, 8):
            label = getattr(self, f"day{day_no}Item", None)
            if label is not None:
                label.setText(self._format_routine_day(day_no))
        self.setup_today_routine_text()

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
        """Show routine controls only after a completed body analysis exists."""
        day_items = [
            "day1Item",
            "day2Item",
            "day3Item",
            "day4Item",
            "day5Item",
            "day6Item",
            "day7Item",
        ]
        dashboard_top = 90 if self.has_completed_analysis else 185

        if hasattr(self, "toggleRoutineButton"):
            self.toggleRoutineButton.setVisible(self.has_completed_analysis)
        if not self.has_completed_analysis:
            is_all_visible = False
            if hasattr(self, "toggleRoutineButton"):
                self.toggleRoutineButton.setChecked(False)

        if is_all_visible:
            self.toggleRoutineButton.setText("📅 View All Days (On)")
            self.todayRoutineCard.setFixedHeight(310)
            self.todayRoutineCard.move(30, dashboard_top)
            self.allDaysContainerLabel.show()
            for item_name in day_items:
                getattr(self, item_name).show()
            history_y = 415 if self.has_completed_analysis else 510
            summary_y = 467 if self.has_completed_analysis else 562
        else:
            self.toggleRoutineButton.setText("📅 View All Days (Off)")
            self.todayRoutineCard.setFixedHeight(140)
            self.todayRoutineCard.move(30, dashboard_top)
            self.allDaysContainerLabel.hide()
            for item_name in day_items:
                getattr(self, item_name).hide()
            history_y = 245 if self.has_completed_analysis else 340
            summary_y = 297 if self.has_completed_analysis else 394

        self.historyButton.move(30, history_y)
        self.bodyAnalysisSummaryCard.setGeometry(30, summary_y, 460, 120)
        if self.has_completed_analysis:
            self.bodyAnalysisSummaryCard.show()
        else:
            self.bodyAnalysisSummaryCard.hide()

    def display_completed_analysis(self, response, body_id=None):
        if isinstance(response, dict) and isinstance(
            response.get("result"), dict
        ):
            data = dict(response)
            data.update(response["result"])
        else:
            data = response if isinstance(response, dict) else {}
        if data.get("status") != "completed":
            clear_body_id(self.login_window, body_id)
            self.completed_body_id = None
            self.has_completed_analysis = False
            self.latest_routine = None
            self._apply_routine_to_dashboard()
            self.bodyAnalysisSummaryCard.hide()
            self.analysisNudgeCard.show()
            self.set_routine_view_mode(self.toggleRoutineButton.isChecked())
            return
        if body_id:
            self.completed_body_id = body_id
            self.has_completed_analysis = True
            try:
                save_body_id(self.login_window, body_id)
            except (OSError, ValueError) as exc:
                print(f"[체형 분석 캐시 저장 실패] {exc}", flush=True)
        self.latest_routine = (
            data.get("routine")
            if isinstance(data.get("routine"), dict)
            else None
        )
        self._apply_routine_to_dashboard()
        self.analysisNudgeCard.hide()
        self.bodyAnalysisSummaryLabel.setText(format_area_summary(data))
        self.set_routine_view_mode(self.toggleRoutineButton.isChecked())

    def _latest_analysis_failed(self, message, status_code=None):
        if status_code == 404:
            clear_body_id(self.login_window, self.completed_body_id)
            self.completed_body_id = None
            self.has_completed_analysis = False
            self.bodyAnalysisSummaryCard.hide()
            self.analysisNudgeCard.show()
            self.set_routine_view_mode(self.toggleRoutineButton.isChecked())
            return
        self.bodyAnalysisSummaryLabel.setText(
            "최근 체형 분석 결과를 불러오지 못했습니다."
        )
        print(f"[최근 체형 분석 조회 실패] {message}", flush=True)

    def switch_to_body_upload(self):
        from GUI.FitHelper.body_upload_window import BodyUploadWindow

        self.body_upload_window = BodyUploadWindow(self)
        self.body_upload_window.show()
        self.hide()

    def go_to_workout(self):
        from GUI.FitHelper.workout_window import WorkoutWindow

        self.workout_window = WorkoutWindow(self)
        self.workout_window.show()
        self.hide()

    def go_to_history_detail(self):
        from GUI.FitHelper.history_window import HistoryDetailWindow

        self.history_detail_window = HistoryDetailWindow(self)
        self.history_detail_window.show()
        self.hide()

    def go_to_profile(self):
        from GUI.FitHelper.profile_window import ProfileSettingsWindow

        self.profile_window = ProfileSettingsWindow(self)
        self.profile_window.show()
        self.hide()

    def logout(self):
        try:
            from GUI.FitHelper import login_client

            login_client.logout()
        except (ImportError, AttributeError):
            pass
        if self.login_window is not None:
            self.login_window.show()
        else:
            from GUI.FitHelper.login_window import LoginWindow

            self.new_login = LoginWindow()
            self.new_login.show()
        self.close()

    def closeEvent(self, event):
        if self.login_window is not None:
            self.login_window.close()
        event.accept()
