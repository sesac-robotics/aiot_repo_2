"""Workout history calendar backed by the AIFit exercise-report API."""

import os
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PyQt6 import uic
from PyQt6.QtCore import QDate, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QDialog,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://10.17.178.246:8000").rstrip("/")


class _RecordCalendarWidget(QCalendarWidget):
    """Calendar that draws a red dot on dates known to have workout reports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recorded_dates = set()

    def paintCell(self, painter, rect, date):
        super().paintCell(painter, rect, date)
        if date.toString("yyyy-MM-dd") not in self.recorded_dates:
            return
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0284C7"))
        painter.drawEllipse(rect.center().x() - 3, rect.bottom() - 9, 6, 6)
        painter.restore()


class _MonthRecordsThread(QThread):
    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, year, month, token, parent=None):
        super().__init__(parent)
        self.year = year
        self.month = month
        self.month_key = f"{year:04d}-{month:02d}"
        self.token = token

    @staticmethod
    def _check_date(workout_date, token):
        response = requests.get(
            f"{SERVER_URL}/exercise-report",
            params={"date": workout_date},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code == 404:
            return None
        if response.status_code == 401:
            raise RuntimeError("로그인 토큰이 만료되었습니다. 다시 로그인해 주세요.")
        if not response.ok:
            try:
                detail = response.json().get("detail", f"HTTP {response.status_code}")
            except ValueError:
                detail = f"HTTP {response.status_code}"
            raise RuntimeError(str(detail))
        try:
            payload = response.json()
        except ValueError:
            return None
        return workout_date if isinstance(payload.get("report_json"), dict) else None

    def run(self):
        last_day = monthrange(self.year, self.month)[1]
        dates = [f"{self.year:04d}-{self.month:02d}-{day:02d}" for day in range(1, last_day + 1)]
        found = set()
        try:
            # This server exposes reports by date, so check the current month in
            # the background, with a small concurrency limit to keep the UI snappy.
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._check_date, workout_date, self.token): workout_date
                    for workout_date in dates
                }
                for future in as_completed(futures):
                    workout_date = future.result()
                    if workout_date:
                        found.add(workout_date)
            self.succeeded.emit(self.month_key, found)
        except requests.RequestException as exc:
            self.failed.emit(self.month_key, f"운동 기록 날짜를 확인하지 못했습니다: {exc}")
        except RuntimeError as exc:
            self.failed.emit(self.month_key, f"운동 기록 날짜를 확인하지 못했습니다: {exc}")


def _format_report_for_display(report_json):
    """Turn report_json into a readable, exercise-by-exercise list."""
    if not isinstance(report_json, dict):
        return str(report_json)

    lines = []
    routine_name = report_json.get("routine_name") or report_json.get("routine")
    day_no = report_json.get("day_no")
    if routine_name or day_no is not None:
        heading = str(routine_name) if routine_name else "운동 세션"
        if day_no is not None:
            heading += f" · Day {day_no}"
        lines.append(heading)
        lines.append("")

    exercises = next(
        (report_json.get(key) for key in ("exercises", "exercise_list", "workouts")
         if isinstance(report_json.get(key), list)),
        None,
    )
    if exercises is not None:
        if not exercises:
            lines.append("저장된 개별 운동 기록이 없습니다.")
        for index, exercise in enumerate(exercises, start=1):
            if not isinstance(exercise, dict):
                lines.extend([f"{index}. {exercise}", ""])
                continue

            name = exercise.get("exercise_name") or exercise.get("name") or exercise.get("title") or f"운동 {index}"
            lines.append(f"{index}. {name}")
            sets = exercise.get("sets")
            reps = exercise.get("reps")
            if sets is not None or reps is not None:
                details = []
                if sets is not None:
                    details.append(f"목표 세트 {sets}세트")
                if reps is not None:
                    details.append(f"목표 반복 {reps}회")
                lines.append("   " + " · ".join(details))

            records = exercise.get("records") or exercise.get("results") or []
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    lines.append(f"   - {record}")
                    continue
                set_no = record.get("set_no")
                rep_no = record.get("rep_no")
                label = "   -"
                if set_no is not None:
                    label += f" {set_no}세트"
                if rep_no is not None:
                    label += f" {rep_no}회"
                state = record.get("pose_state")
                score = record.get("score")
                feedback = record.get("feedback_text")
                parts = [str(value) for value in (state, f"점수 {score}" if score is not None else None, feedback) if value]
                if parts:
                    label += " · " + " · ".join(parts)
                lines.append(label)
            lines.append("")

        if len(lines) > 0:
            return "\n".join(lines).rstrip()

    # Keep useful information visible even if the JSON uses a different shape.
    for key, value in report_json.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}:")
            if isinstance(value, list):
                for item in value:
                    lines.append(f"  • {item}")
            else:
                for subkey, subvalue in value.items():
                    lines.append(f"  • {subkey}: {subvalue}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) or "표시할 운동 리포트 내용이 없습니다."


class _ReportRequestThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, workout_date, token, parent=None):
        super().__init__(parent)
        self.workout_date = workout_date
        self.token = token

    def run(self):
        try:
            response = requests.get(
                f"{SERVER_URL}/exercise-report",
                params={"date": self.workout_date},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=20,
            )
            try:
                payload = response.json()
            except ValueError:
                payload = {}

            if response.status_code == 404:
                raise RuntimeError(f"{self.workout_date} 날짜에 저장된 운동 리포트가 없습니다.")
            if response.status_code == 401:
                raise RuntimeError("로그인 토큰이 만료되었습니다. 다시 로그인해 주세요.")
            if not response.ok:
                raise RuntimeError(str(payload.get("detail", f"HTTP {response.status_code}")))
            if not isinstance(payload.get("report_json"), dict):
                raise RuntimeError("서버 응답에 올바른 report_json이 없습니다.")
            self.succeeded.emit(payload)
        except requests.Timeout:
            self.failed.emit("서버 응답 시간이 초과되었습니다.")
        except requests.ConnectionError:
            self.failed.emit(f"AIFit Server에 연결할 수 없습니다: {SERVER_URL}")
        except requests.RequestException as exc:
            self.failed.emit(f"서버 통신 오류: {exc}")
        except RuntimeError as exc:
            self.failed.emit(str(exc))


class HistoryDetailWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.welcome_window = parent
        self._report_thread = None
        self.calendar_dialog = None
        self._recorded_dates_by_month = {}
        self._month_threads = {}
        ui_path = os.path.join(os.path.dirname(__file__), "history_detail.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, "backDashboardButton"):
            self.backDashboardButton.clicked.connect(self.return_to_welcome)
        if hasattr(self, "openCalendarButton"):
            self.openCalendarButton.clicked.connect(self.open_calendar_view)

    @staticmethod
    def _get_access_token():
        try:
            import login_client
            return getattr(login_client, "_access_token", None)
        except ImportError:
            return None

    def open_calendar_view(self):
        if self.calendar_dialog is not None and self.calendar_dialog.isVisible():
            self.calendar_dialog.raise_()
            self.calendar_dialog.activateWindow()
            return

        # Refresh each time the calendar is opened so newly saved workouts appear.
        self._recorded_dates_by_month.clear()
        dialog = QDialog(self)
        self.calendar_dialog = dialog
        dialog.setWindowTitle("운동 리포트 달력")
        dialog.resize(430, 560)
        layout = QVBoxLayout(dialog)

        calendar = _RecordCalendarWidget(dialog)
        calendar.setGridVisible(True)
        dialog.selected_date = None
        status = QLabel(
            "운동 기록이 있는 날짜에는 빨간 점이 표시됩니다. 조회할 날짜를 선택한 뒤 확인을 눌러주세요.",
            dialog,
        )
        status.setWordWrap(True)
        button_row = QVBoxLayout()
        confirm_button = QPushButton("확인", dialog)
        cancel_button = QPushButton("취소", dialog)
        confirm_button.clicked.connect(
            lambda: self._confirm_selected_date(dialog, status)
        )
        cancel_button.clicked.connect(dialog.reject)

        layout.addWidget(calendar)
        layout.addWidget(status)
        button_row.addWidget(confirm_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)
        calendar.clicked.connect(lambda date: self._select_date(dialog, date, status))
        calendar.currentPageChanged.connect(
            lambda year, month: self._load_month_record_dates(
                year, month, calendar, status, dialog
            )
        )
        dialog.finished.connect(lambda _result: setattr(self, "calendar_dialog", None))
        dialog.open()
        self._load_month_record_dates(
            calendar.yearShown(), calendar.monthShown(), calendar, status, dialog
        )

    def _load_month_record_dates(self, year, month, calendar, status, dialog):
        month_key = f"{year:04d}-{month:02d}"
        if month_key in self._recorded_dates_by_month:
            calendar.recorded_dates = self._recorded_dates_by_month[month_key]
            calendar.updateCells()
            return
        if month_key in self._month_threads:
            return

        token = self._get_access_token()
        if not token:
            status.setText("로그인 토큰을 찾지 못했습니다. 다시 로그인해 주세요.")
            return

        thread = _MonthRecordsThread(year, month, token, self)
        self._month_threads[month_key] = thread
        thread.succeeded.connect(
            lambda key, dates: self._month_records_loaded(key, dates, calendar, dialog)
        )
        thread.failed.connect(
            lambda key, message: self._month_records_failed(
                key, message, calendar, status, dialog
            )
        )
        thread.finished.connect(lambda key=month_key: self._month_threads.pop(key, None))
        thread.start()

    def _month_records_loaded(self, month_key, dates, calendar, dialog):
        self._recorded_dates_by_month[month_key] = set(dates)
        if self.calendar_dialog is not dialog:
            return
        visible_month = f"{calendar.yearShown():04d}-{calendar.monthShown():02d}"
        if visible_month == month_key:
            calendar.recorded_dates = set(dates)
            calendar.updateCells()

    @staticmethod
    def _month_records_failed(month_key, message, calendar, status, dialog):
        if dialog.isVisible() and f"{calendar.yearShown():04d}-{calendar.monthShown():02d}" == month_key:
            calendar.setToolTip(message)
            status.setText(message)

    @staticmethod
    def _select_date(dialog, selected_date, status):
        dialog.selected_date = selected_date
        status.setText(
            f"선택한 날짜: {selected_date.toString('yyyy-MM-dd')} · 확인을 누르면 리포트를 조회합니다."
        )

    def _confirm_selected_date(self, dialog, status):
        selected_date = dialog.selected_date
        if selected_date is None:
            status.setText("먼저 조회할 날짜를 선택해 주세요.")
            return
        workout_date = selected_date.toString("yyyy-MM-dd")
        dialog.accept()
        self.load_report_for_date(workout_date)

    def load_report_for_date(self, workout_date):
        """Load and display a report for a date without opening the calendar."""
        token = self._get_access_token()
        if not token:
            self._show_error("로그인 토큰을 찾지 못했습니다. 다시 로그인해 주세요.", workout_date)
            return

        if hasattr(self, "dateEdit"):
            selected_date = QDate.fromString(workout_date, "yyyy-MM-dd")
            if selected_date.isValid():
                self.dateEdit.setDate(selected_date)
        self.recordLabel.setText(f"⏳ {workout_date} 운동 리포트를 불러오는 중...")
        self.analysisContent.setPlainText("운동 리포트를 불러오는 중입니다...")

        thread = _ReportRequestThread(workout_date, token, self)
        self._report_thread = thread
        thread.succeeded.connect(lambda payload: self._show_report(payload, workout_date))
        thread.failed.connect(lambda message: self._show_error(message, workout_date))
        thread.start()

    def _show_report(self, payload, workout_date):
        report_json = payload["report_json"]
        pretty_report = _format_report_for_display(report_json)

        score = payload.get("avg_score")
        score_text = f"평균 점수: {score}" if score is not None else "평균 점수 없음"
        response_date = payload.get("date", workout_date)
        if hasattr(self, "recordLabel"):
            self.recordLabel.setText(f"✅ {response_date} 운동 리포트 조회 완료 · {score_text}")
        if hasattr(self, "analysisContent"):
            self.analysisContent.setPlainText(pretty_report)

    def _show_error(self, message, workout_date):
        self.recordLabel.setText(f"❌ {workout_date} 운동 리포트를 불러오지 못했습니다.")
        self.analysisContent.setPlainText(message)

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.calendar_dialog is not None:
            self.calendar_dialog.close()
            self.calendar_dialog = None
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
