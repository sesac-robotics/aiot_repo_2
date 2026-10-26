"""Workout completion report shown immediately after a session ends."""

import os
from datetime import datetime

import requests
from PyQt6 import uic
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QTableWidgetItem


SERVER_URL = os.getenv("AIFIT_SERVER_URL", "http://10.17.178.246:8000").rstrip("/")
MAX_REPORT_FETCH_ATTEMPTS = 6
REPORT_RETRY_DELAY_MS = 1000


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _records_from(report_json):
    exercises = next(
        (report_json.get(key) for key in ("exercises", "exercise_list", "workouts")
         if isinstance(report_json.get(key), list)),
        [],
    )
    records = []
    names = []
    for exercise_index, exercise in enumerate(exercises, start=1):
        if not isinstance(exercise, dict):
            names.append(str(exercise))
            continue
        name = (exercise.get("exercise_name") or exercise.get("name")
                or exercise.get("title") or f"운동 {exercise_index}")
        names.append(str(name))
        items = exercise.get("records") or exercise.get("results") or []
        if isinstance(items, list):
            for record in items:
                if isinstance(record, dict):
                    item = dict(record)
                    item.setdefault("exercise_name", name)
                    records.append(item)
    return names, records


def _feedback_from(report_json):
    """Extract improvement feedback that the server actually returned."""
    if not isinstance(report_json, dict):
        return []

    feedback = []
    for key in ("improvement_feedback", "bad_feedback", "weaknesses", "feedback"):
        value = report_json.get(key)
        if isinstance(value, str) and value.strip():
            feedback.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    message = (item.get("feedback_text") or item.get("message")
                               or item.get("text") or item.get("description"))
                    if message:
                        feedback.append(str(message).strip())
                elif item:
                    feedback.append(str(item).strip())
        elif isinstance(value, dict):
            bad_items = value.get("bad") or value.get("improvement") or value.get("weaknesses") or []
            if isinstance(bad_items, str):
                feedback.append(bad_items.strip())
            elif isinstance(bad_items, list):
                for item in bad_items:
                    if isinstance(item, dict):
                        message = (item.get("feedback_text") or item.get("message")
                                   or item.get("text") or item.get("description"))
                        if message:
                            feedback.append(str(message).strip())
                    elif item:
                        feedback.append(str(item).strip())

    _, records = _records_from(report_json)
    for record in records:
        message = (record.get("feedback_text") or record.get("feedback")
                   or record.get("message"))
        if not isinstance(message, str) or not message.strip():
            continue

        # Prefer explicit server classification when available. Otherwise keep
        # only feedback attached to a record that indicates an error.
        state = str(record.get("pose_state") or record.get("state")
                    or record.get("status") or "").upper()
        errors = record.get("error_codes") or record.get("errors")
        is_bad = (
            state in {"BAD", "INCORRECT", "WRONG", "ERROR", "POOR", "NEEDS_IMPROVEMENT"}
            or bool(errors)
        )
        if is_bad:
            rep_no = record.get("rep_no")
            prefix = f"Reps: {rep_no} · " if rep_no is not None else ""
            feedback.append(prefix + message.strip())

    # Preserve server order while avoiding duplicate messages.
    return list(dict.fromkeys(feedback))


class _ReportFetchThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, bool)

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
            if not response.ok:
                message = str(payload.get("detail") or f"HTTP {response.status_code}")
                self.failed.emit(message, response.status_code in {404, 502, 503, 504})
                return
            if not isinstance(payload.get("report_json"), dict):
                self.failed.emit("서버 응답에 운동 기록이 없습니다.", True)
                return
            self.succeeded.emit(payload)
        except requests.RequestException as exc:
            self.failed.emit(f"저장된 운동 기록을 불러오지 못했습니다: {exc}", True)


class WorkoutReportWindow(QMainWindow):
    def __init__(self, parent=None, session_info=None):
        super().__init__(parent)
        self.welcome_window = parent
        self.session_info = session_info if isinstance(session_info, dict) else {}
        self._fetch_thread = None
        self._report_attempt = 0
        uic.loadUi(os.path.join(os.path.dirname(__file__), "workout_report.ui"), self)
        self.backDashboardButton.clicked.connect(self.return_to_welcome)
        self._prepare_report_view()
        self._load_saved_report()

    @staticmethod
    def _access_token():
        try:
            import login_client
            return getattr(login_client, "_access_token", None)
        except ImportError:
            return None

    @staticmethod
    def _parse_time(value):
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
        except (TypeError, ValueError):
            return None

    def _prepare_report_view(self):
        """Keep the report empty until the authenticated server response arrives."""
        self.dateValue.setText("서버 응답 대기 중")
        self.timeValue.setText("서버 응답 대기 중")
        self.exerciseValue.setText("AIFit Server에서 운동 리포트를 불러오는 중입니다.")
        self._render_records([])
        self.pageSubtitle.setText("AIFit Server에서 운동 분석 결과를 불러오는 중입니다.")

    def _render_records(self, records):
        table = self.repScoreTable
        table.setRowCount(0)
        scores = []
        for record in records:
            if not isinstance(record, dict):
                continue
            row = table.rowCount()
            table.insertRow(row)
            values = (
                str(record.get("set_no", "—")),
                str(record.get("rep_no", "—")),
                f"{_as_float(record.get('score')):g}" if _as_float(record.get("score")) is not None else "—",
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
            score = _as_float(record.get("score"))
            if score is not None:
                scores.append(score)
        if scores:
            self.averageScoreValue.setText(f"{sum(scores) / len(scores):.1f}점")
        else:
            self.averageScoreValue.setText("—")
        self.badFeedbackText.clear()
        table.resizeColumnsToContents()

    def _load_saved_report(self):
        token = self._access_token()
        if not token:
            self._show_report_error("로그인이 필요합니다. 다시 로그인해 주세요.")
            return
        self._report_attempt += 1
        report_time = (
            self._parse_time(self.session_info.get("started_at"))
            or self._parse_time(self.session_info.get("ended_at"))
            or datetime.now()
        )
        thread = _ReportFetchThread(report_time.date().isoformat(), token, self)
        self._fetch_thread = thread
        thread.succeeded.connect(self._report_loaded)
        thread.failed.connect(self._report_fetch_failed)
        thread.start()

    def _report_fetch_failed(self, message, retryable):
        print(f"[WORKOUT_REPORT] {message}", flush=True)
        if retryable and self._report_attempt < MAX_REPORT_FETCH_ATTEMPTS:
            QTimer.singleShot(REPORT_RETRY_DELAY_MS, self._load_saved_report)
            return
        self._show_report_error(f"서버 운동 리포트를 불러오지 못했습니다: {message}")

    def _show_report_error(self, message):
        self.pageSubtitle.setText(message)
        self.dateValue.setText("—")
        self.timeValue.setText("—")
        self.exerciseValue.setText("서버 리포트를 표시할 수 없습니다.")
        self._render_records([])
        self.badFeedbackText.clear()

    def _report_loaded(self, payload):
        report = payload.get("report_json") or {}
        names, records = _records_from(report)
        response_date = payload.get("date") or report.get("date")
        self.dateValue.setText(str(response_date) if response_date else "—")

        started = self._parse_time(payload.get("started_at") or report.get("started_at"))
        ended = self._parse_time(payload.get("ended_at") or report.get("ended_at"))
        if started and ended:
            elapsed = max(0, int((ended - started).total_seconds()))
            self.timeValue.setText(
                f"{started.strftime('%H:%M:%S')} – {ended.strftime('%H:%M:%S')}  · "
                f"{elapsed // 60}분 {elapsed % 60}초"
            )
        else:
            self.timeValue.setText("—")
        self.exerciseValue.setText("\n".join(dict.fromkeys(names)) or "서버 응답에 운동 종목 정보가 없습니다.")
        self._render_records(records)
        self.badFeedbackText.setPlainText("\n".join(_feedback_from(report)))
        average = _as_float(payload.get("avg_score"))
        if average is None:
            average = _as_float(report.get("avg_score"))
        self.averageScoreValue.setText(f"{average:.1f}점" if average is not None else "—")
        self.pageSubtitle.setText("AIFit Server에서 받은 운동 분석 결과입니다.")

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
