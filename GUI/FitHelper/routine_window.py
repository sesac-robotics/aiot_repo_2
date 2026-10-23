"""Display the 7-day routine returned by the authenticated AIFit API."""
import os
import sys
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow

# The shared authenticated API client and per-account analysis cache live above
# this GUI package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import body_analysis_client as body_api
from body_analysis_cache import load_body_id
from body_analysis_client import BodyAnalysisAPIError


DAY_NAMES = {
    1: "월요일",
    2: "화요일",
    3: "수요일",
    4: "목요일",
    5: "금요일",
    6: "토요일",
    7: "일요일",
}


def _extract_routine(payload):
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        routine = result.get("routine")
        if isinstance(routine, dict):
            return routine
        if isinstance(result.get("days"), list):
            return result
    routine = payload.get("routine")
    if isinstance(routine, dict):
        return routine
    return None


def _exercise_text(exercise):
    if not isinstance(exercise, dict):
        return "운동 정보 없음"
    name = str(exercise.get("exercise_name") or "운동")
    sets = exercise.get("sets")
    reps = exercise.get("reps")
    duration = exercise.get("target_duration_sec")
    details = []
    if sets is not None:
        details.append(f"{sets}세트")
    if duration is not None:
        details.append(f"{duration:g}초 유지" if isinstance(duration, (int, float)) else f"{duration}초 유지")
    elif reps is not None:
        details.append(f"{reps}회")
    return name + (" · " + " · ".join(details) if details else "")


class _RoutineFetchThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, body_id, parent=None):
        super().__init__(parent)
        self.body_id = body_id

    def run(self):
        try:
            response = body_api.get_analysis(self.body_id)
            if response.get("status") != "completed":
                raise BodyAnalysisAPIError("체형 분석이 아직 완료되지 않았습니다.")
            self.succeeded.emit(response)
        except BodyAnalysisAPIError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"루틴 조회 중 오류가 발생했습니다: {exc}")


class FullRoutineWindow(QMainWindow):
    def __init__(self, parent=None, analysis_data=None):
        super().__init__()
        self.welcome_window = parent
        self.fetch_thread = None
        ui_path = os.path.join(os.path.dirname(__file__), "full_routine.ui")
        uic.loadUi(ui_path, self)

        if hasattr(self, "closeRoutineButton"):
            self.closeRoutineButton.clicked.connect(self.return_to_welcome)

        routine = _extract_routine(analysis_data)
        if routine is not None:
            self.display_routine(routine)
            return

        self._show_message("서버에서 7일 루틴을 불러오는 중입니다.")
        body_id = analysis_data.get("body_id") if isinstance(analysis_data, dict) else None
        if not body_id:
            login_window = getattr(parent, "login_window", None)
            body_id = load_body_id(login_window)
        if body_id:
            self.fetch_thread = _RoutineFetchThread(body_id, self)
            self.fetch_thread.succeeded.connect(self._fetch_succeeded)
            self.fetch_thread.failed.connect(self._fetch_failed)
            self.fetch_thread.start()
        else:
            self._show_message("저장된 체형 분석 기록이 없습니다.")

    def _show_message(self, message):
        if hasattr(self, "routineHeaderTitle"):
            self.routineHeaderTitle.setText(message)
        for day_no in range(1, 8):
            label = getattr(self, f"day{day_no}Item", None)
            if label is not None:
                label.setText(f"{DAY_NAMES[day_no]} · {message}")

    def _fetch_succeeded(self, response):
        routine = _extract_routine(response)
        if routine is None:
            self._show_message("서버 응답에 7일 루틴 정보가 없습니다.")
            return
        self.display_routine(routine)

    def _fetch_failed(self, message):
        self._show_message(f"루틴을 불러오지 못했습니다: {message}")

    def display_routine(self, routine):
        days = routine.get("days")
        if not isinstance(days, list):
            self._show_message("서버 루틴의 날짜 정보가 올바르지 않습니다.")
            return

        days_by_number = {
            day.get("day_no"): day
            for day in days
            if isinstance(day, dict) and isinstance(day.get("day_no"), int)
        }
        title = routine.get("routine_name") or "7일 맞춤 운동 루틴"
        target = routine.get("target_body_part")
        if target:
            title = f"{title} · 목표 부위: {target}"
        if hasattr(self, "routineHeaderTitle"):
            self.routineHeaderTitle.setText(title)

        for day_no, day_name in DAY_NAMES.items():
            label = getattr(self, f"day{day_no}Item", None)
            if label is None:
                continue
            day = days_by_number.get(day_no)
            exercises = day.get("exercises", []) if isinstance(day, dict) else None
            if not isinstance(exercises, list):
                text = f"{day_name} · 루틴 정보 없음"
            elif not exercises:
                text = f"{day_name} · 휴식"
            else:
                text = f"{day_name} · " + " / ".join(_exercise_text(item) for item in exercises)
            label.setText(text)

    def return_to_welcome(self):
        if self.welcome_window is not None:
            self.welcome_window.show()
        self.close()

    def closeEvent(self, event):
        if self.welcome_window is not None:
            self.welcome_window.show()
        event.accept()
