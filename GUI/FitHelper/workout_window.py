"""Workout UI connected to AIFit Server session orchestration and video stream."""
import os
import re
from datetime import datetime

import requests
from PyQt6 import uic
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow

from video_receiver import VideoReceiver

AIFIT_SERVER_URL = "http://10.17.178.246:8000"
STREAM_URL = f"{AIFIT_SERVER_URL}/api/video/stream"
REQUEST_TIMEOUT = 15


class _ApiRequestThread(QThread):
    completed = pyqtSignal(str, object, object)

    def __init__(self, operation, method, path, access_token, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.method = method
        self.path = path
        self.access_token = access_token

    def run(self):
        try:
            response = requests.request(
                self.method,
                f"{AIFIT_SERVER_URL}{self.path}",
                headers={"Authorization": f"Bearer {self.access_token}"},
                timeout=REQUEST_TIMEOUT,
            )
            try:
                body = response.json()
            except ValueError:
                body = {}
            if not response.ok:
                detail = body.get("detail") or f"HTTP {response.status_code}"
                raise RuntimeError(str(detail))
            self.completed.emit(self.operation, body, None)
        except requests.RequestException as exc:
            self.completed.emit(self.operation, None, f"AIFit Server 통신 오류: {exc}")
        except Exception as exc:
            self.completed.emit(self.operation, None, str(exc))


class WorkoutWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        ui_path = os.path.join(os.path.dirname(__file__), "workout_window.ui")
        uic.loadUi(ui_path, self)

        self.receiver = None
        self.is_paused = False
        self.session_id = None
        self.exercise_id = None
        self.session_status = None
        self.session_complete = False
        self._opening_report = False
        self._report_window = None
        self._active_operation = None
        self._finish_requested = False
        self._request_threads = set()
        self._last_feedback_signature = None
        self._session_started_at = None
        self._session_exercise_names = []
        self._session_records = {}
        self._target_reps = None
        self._auto_finish_for_report = False

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(1000)
        self._live_timer.timeout.connect(self.fetch_live_status)

        if hasattr(self, "endWorkoutButton"):
            self.endWorkoutButton.clicked.connect(self.return_to_welcome)
        if hasattr(self, "pauseResumeButton"):
            self.pauseResumeButton.setText("연결 중...")
            self.pauseResumeButton.setEnabled(False)
            self.pauseResumeButton.clicked.connect(self.toggle_pause)

        # 세션과 첫 운동을 서버에서 받은 뒤에 영상 표시를 시작합니다.
        self.create_session()

    def _access_token(self):
        try:
            import login_client
            return getattr(login_client, "_access_token", None)
        except ImportError:
            return None

    def _send_api(self, operation, method, path):
        if self._active_operation is not None:
            return
        token = self._access_token()
        if not token:
            self.update_status("로그인이 필요합니다. 다시 로그인해 주세요.")
            return

        self._active_operation = operation
        print(f"[WORKOUT][REQUEST] {method} {path}", flush=True)
        thread = _ApiRequestThread(operation, method, path, token, self)
        self._request_threads.add(thread)
        thread.completed.connect(self._handle_api_result)
        thread.finished.connect(lambda thread=thread: self._request_threads.discard(thread))
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def create_session(self):
        """Ask AIFit to create today's session and start its first AI exercise."""
        if self.session_id is not None:
            return
        self.update_status("오늘의 운동 세션을 준비하고 있습니다...")
        self._send_api("create_session", "POST", "/api/workout-sessions")

    def _handle_api_result(self, operation, result, error):
        if self._active_operation == operation:
            self._active_operation = None

        if error:
            print(f"[WORKOUT][ERROR] operation={operation} detail={error}", flush=True)
            self.update_status(f"요청 실패: {error}")
            if operation == "create_session" and hasattr(self, "pauseResumeButton"):
                self.pauseResumeButton.setText("다시 시도")
                self.pauseResumeButton.setEnabled(True)
            if operation == "finish" and hasattr(self, "endWorkoutButton"):
                self.endWorkoutButton.setEnabled(True)
            if operation == "finish" and self._auto_finish_for_report:
                self._auto_finish_for_report = False
                self.session_complete = True
                self._open_completion_report()
                return
            if self._finish_requested and operation != "finish":
                self._send_pending_finish()
            return

        print(f"[WORKOUT][SUCCESS] operation={operation}", flush=True)
        if operation == "create_session":
            self._session_started_at = result.get("started_at") or datetime.now().astimezone().isoformat()
            self.session_id = result.get("session_id")
            exercise = result.get("exercise") or {}
            self.exercise_id = exercise.get("exercise_id")
            self._target_reps = exercise.get("reps")
            if exercise.get("exercise_name"):
                self._session_exercise_names.append(exercise["exercise_name"])
            self.session_status = result.get("session_status", "IN_PROGRESS")
            if not self.session_id or not self.exercise_id:
                self.update_status("세션 응답에 session_id 또는 exercise_id가 없습니다.")
                return
            print(
                f"[WORKOUT][SESSION] session_id={self.session_id} "
                f"exercise_id={self.exercise_id} "
                f"exercise={exercise.get('exercise_name')} "
                f"target_reps={exercise.get('reps')} "
                f"target_duration_sec={exercise.get('target_duration_sec')}",
                flush=True,
            )
            self._display_exercise(exercise, result.get("current_set_no", 1))
            if hasattr(self, "pauseResumeButton"):
                self.pauseResumeButton.setText("⏸ Pause")
                self.pauseResumeButton.setEnabled(True)
            self.start_video()
            self._live_timer.start()
        elif operation == "live":
            self._apply_live_status(result)
        elif operation in {"pause", "resume"}:
            self.session_status = result.get("session_status", self.session_status)
            if result.get("exercise_id"):
                self.exercise_id = result["exercise_id"]
            if operation == "resume":
                self.is_paused = False
            else:
                self.is_paused = True
            self.fetch_live_status()
        elif operation == "finish":
            self.session_complete = True
            if self._auto_finish_for_report:
                self._auto_finish_for_report = False
                self._open_completion_report()
            else:
                self._return_to_dashboard()

        if self._finish_requested and operation != "finish":
            self._send_pending_finish()

    def _display_exercise(self, exercise, set_no):
        name = exercise.get("exercise_name", "오늘의 운동")
        sets = exercise.get("sets", "?")
        reps = exercise.get("reps")
        duration = exercise.get("target_duration_sec")
        if duration is not None:
            goal = f"{duration}초 유지"
        elif reps is not None:
            goal = f"{reps}회"
        else:
            goal = "목표 확인 중"
        if hasattr(self, "feedbackLabel"):
            self.feedbackLabel.setText(f"💡 {name} · {set_no}/{sets}세트 · 목표 {goal}")
        if hasattr(self, "counterLabel"):
            current = "0초" if duration is not None else "0"
            target = f"{duration}초" if duration is not None else str(reps or "?")
            self.counterLabel.setText(f"🔥 {current} / {target}")

    def start_video(self):
        if self.receiver is not None and self.receiver.isRunning():
            return
        access_token = self._access_token()
        if not access_token:
            self.update_status("로그인이 필요합니다")
            return

        self.receiver = VideoReceiver(
            STREAM_URL,
            parent=self,
            access_token=access_token,
        )
        self.receiver.frame_received.connect(self.update_frame)
        self.receiver.status_changed.connect(self.update_status)
        self.receiver.start()

    def fetch_live_status(self):
        if self.session_id is None or self._active_operation is not None or self.session_complete:
            return
        self._send_api(
            "live",
            "GET",
            f"/api/workout-sessions/{self.session_id}/live",
        )

    def _apply_live_status(self, data):
        was_complete = self.session_complete
        self.session_status = data.get("session_status", self.session_status)
        self.session_complete = bool(data.get("session_complete"))
        self.exercise_id = data.get("current_exercise_id", self.exercise_id)
        current_exercise = data.get("current_exercise") or data.get("exercise") or {}
        if isinstance(current_exercise, dict) and current_exercise.get("exercise_name"):
            self._session_exercise_names.append(current_exercise["exercise_name"])
        current_set = data.get("current_set_no", 1)
        snapshot = data.get("ai_server") or {}
        ai_exercise = snapshot.get("exercise") if isinstance(snapshot, dict) else None
        ai_exercise = ai_exercise if isinstance(ai_exercise, dict) else {}
        rep_count = ai_exercise.get("rep_count")
        hold_seconds = ai_exercise.get("current_hold_sec")
        target_reps = data.get("target_reps") or self._target_reps
        if target_reps is not None:
            try:
                target_reps = int(target_reps)
                self._target_reps = target_reps
            except (TypeError, ValueError):
                target_reps = self._target_reps
        target_duration = data.get("target_duration_sec")
        record = data.get("latest_record") or {}
        record_rep_no = record.get("rep_no")
        if rep_count is not None:
            try:
                rep_count = int(rep_count)
            except (TypeError, ValueError):
                pass
        if record_rep_no is not None:
            try:
                record_rep_no = int(record_rep_no)
            except (TypeError, ValueError):
                pass
        observed_rep_count = rep_count if rep_count is not None else record_rep_no
        score = data.get("live_score")
        if score is None and record:
            score = record.get("score")
        feedback = data.get("live_feedback") or record.get("feedback_text")

        if record_rep_no is not None:
            record_key = (record.get("set_no", current_set), record_rep_no)
            self._session_records[record_key] = {
                "set_no": record.get("set_no", current_set),
                "rep_no": record_rep_no,
                "score": score,
                "feedback_code": record.get("feedback_code"),
                "feedback_text": feedback,
                "pose_state": record.get("pose_state"),
            }

        paused = self.session_status == "PAUSED"
        self.is_paused = paused
        if hasattr(self, "counterLabel"):
            if target_duration is not None:
                current = int(hold_seconds or 0)
                self.counterLabel.setText(f"🔥 {current}초 / {target_duration}초")
            elif rep_count is not None:
                target = target_reps if target_reps is not None else "?"
                self.counterLabel.setText(f"🔥 Reps: {rep_count} / {target}")
            elif record_rep_no is not None:
                target = target_reps if target_reps is not None else "?"
                self.counterLabel.setText(f"🔥 Reps: {record_rep_no} / {target}")

        # Match the exact rep count visible in the workout window.
        rep_goal_reached = False
        if target_duration is None and hasattr(self, "counterLabel"):
            match = re.search(r"Reps:\s*(\d+)\s*/\s*(\d+)", self.counterLabel.text(), re.IGNORECASE)
            if match:
                rep_goal_reached = int(match.group(1)) >= int(match.group(2))
        if (
            not rep_goal_reached
            and not paused
            and target_duration is None
            and target_reps is not None
            and observed_rep_count is not None
        ):
            try:
                rep_goal_reached = int(observed_rep_count) >= int(target_reps)
            except (TypeError, ValueError):
                rep_goal_reached = False

        if score is not None or feedback:
            feedback = feedback or "피드백 없음"
            if hasattr(self, "liveScoreLabel"):
                if score is not None:
                    self.liveScoreLabel.setText(f"⭐ {score:g}")
            if hasattr(self, "feedbackLabel"):
                self.feedbackLabel.setText(f"💡 {feedback}")
            signature = (
                record.get("set_no", current_set), record_rep_no, score,
                record.get("feedback_code"), feedback,
            )
            if signature != self._last_feedback_signature:
                self._last_feedback_signature = signature
                print(
                    "[WORKOUT][FEEDBACK] "
                    f"set={record.get('set_no', current_set)} rep={record_rep_no} "
                    f"pose_state={record.get('pose_state')} score={score} "
                    f"error_codes={record.get('error_codes')} "
                    f"feedback_code={record.get('feedback_code')} "
                    f"feedback_text={feedback}",
                    flush=True,
                )

        waiting = bool(data.get("awaiting_start"))
        if hasattr(self, "pauseResumeButton"):
            if self.session_complete:
                self.pauseResumeButton.setText("운동 완료")
                self.pauseResumeButton.setEnabled(False)
            elif paused:
                self.pauseResumeButton.setText("▶ Start")
                self.pauseResumeButton.setEnabled(True)
            else:
                self.pauseResumeButton.setText("⏸ Pause")
                self.pauseResumeButton.setEnabled(True)

        if self.session_complete:
            self._live_timer.stop()
            self.update_status("오늘 루틴을 모두 완료했습니다.")
        elif waiting:
            self.update_status("세트가 완료되었습니다. Start를 눌러 다음 세트를 시작하세요.")

        if self.session_complete and not was_complete:
            self._open_completion_report()
        elif not self.session_complete and rep_goal_reached:
            print(
                "[WORKOUT][AUTO_REPORT] rep target reached; opening report now "
                f"(shown={getattr(self, 'counterLabel', None).text() if hasattr(self, 'counterLabel') else 'data'})",
                flush=True,
            )
            self._finish_after_target_reps()

    def _finish_after_target_reps(self):
        """Finalize the session and show its report when the rep goal is met."""
        if self._auto_finish_for_report or self._report_window is not None:
            return
        self._auto_finish_for_report = True
        self._live_timer.stop()
        self.update_status("목표 반복을 완료했습니다. 결과 리포트를 준비하고 있습니다...")
        if self.session_id is None:
            self._auto_finish_for_report = False
            self.session_complete = True
            self._open_completion_report()
            return
        self._send_api(
            "finish",
            "POST",
            f"/api/workout-sessions/{self.session_id}/finish",
        )
        if self._active_operation != "finish":
            # Still show the local report if finalizing could not be started.
            self._auto_finish_for_report = False
        # Show the report immediately; finalizing can finish in parallel.
        self.session_complete = True
        self._open_completion_report()

    def _open_completion_report(self):
        """Close the live workout view and show its completion report."""
        if self._report_window is not None:
            return

        self._live_timer.stop()
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(1000)

        from workout_report_window import WorkoutReportWindow

        session_info = {
            "started_at": self._session_started_at,
            "ended_at": datetime.now().astimezone().isoformat(),
            "exercise_names": self._session_exercise_names,
            "records": list(self._session_records.values()),
        }
        # The dashboard is hidden while a workout is active.  Do not make
        # the report its Qt child: Qt can hide child windows with that
        # hidden parent.  Retain the dashboard separately for the back action.
        try:
            report_window = WorkoutReportWindow(None, session_info)
            report_window.welcome_window = self.parent_window
            report_window.setWindowFlag(Qt.WindowType.Window, True)
            self._report_window = report_window
            if self.parent_window is not None:
                self.parent_window.workout_report_window = report_window
            self._opening_report = True
            report_window.show()
            report_window.raise_()
            report_window.activateWindow()
            print("[WORKOUT][AUTO_REPORT] report window shown", flush=True)
            QTimer.singleShot(0, self.close)
        except Exception as exc:
            # Keep the workout window available if report construction fails.
            self._opening_report = False
            self._report_window = None
            print(f"[WORKOUT][AUTO_REPORT][ERROR] {type(exc).__name__}: {exc}", flush=True)
            self.update_status(f"결과 창을 열지 못했습니다: {exc}")

    def update_frame(self, image):
        # 일시정지 중에도 영상은 계속 수신하지만 화면 표시만 유지합니다.
        if hasattr(self, "videoLabel"):
            pixmap = QPixmap.fromImage(image)
            self.videoLabel.setPixmap(
                pixmap.scaled(
                    self.videoLabel.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

    def update_status(self, message):
        if hasattr(self, "feedbackLabel") and self.session_status != "PAUSED":
            self.feedbackLabel.setText(f"💡 AI Coach: {message}")

    def toggle_pause(self):
        if self._active_operation is not None:
            return
        if self.session_id is None:
            self.create_session()
            return
        if self.session_complete:
            return
        if self.session_status == "PAUSED":
            self._send_api(
                "resume",
                "POST",
                f"/api/workout-sessions/{self.session_id}/start",
            )
        else:
            self._send_api(
                "pause",
                "POST",
                f"/api/workout-sessions/{self.session_id}/pause",
            )

    def return_to_welcome(self):
        self._live_timer.stop()
        if self._active_operation is not None:
            # A live poll may already be in flight. Queue End instead of
            # dropping the click while another request owns the API slot.
            self._finish_requested = True
            if hasattr(self, "endWorkoutButton"):
                self.endWorkoutButton.setEnabled(False)
            self.update_status("운동 종료 요청을 준비하고 있습니다...")
            return
        self._send_pending_finish()

    def _send_pending_finish(self):
        self._finish_requested = False
        if self.session_id and not self.session_complete:
            if hasattr(self, "endWorkoutButton"):
                self.endWorkoutButton.setEnabled(False)
            self._send_api(
                "finish",
                "POST",
                f"/api/workout-sessions/{self.session_id}/finish",
            )
        else:
            self._return_to_dashboard()

    def _return_to_dashboard(self):
        self._live_timer.stop()
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(1000)
        if self.parent_window is not None:
            self.parent_window.show()
        self.close()

    def closeEvent(self, event):
        self._live_timer.stop()
        if self.receiver is not None and self.receiver.isRunning():
            self.receiver.stop()
            self.receiver.wait(1000)
        if self.parent_window is not None and not self._opening_report:
            self.parent_window.show()
        event.accept()
