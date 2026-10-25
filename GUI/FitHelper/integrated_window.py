"""팀 기본 운동 UI에서 교정 안내와 자세 점수를 함께 표시합니다."""

from PyQt6.QtWidgets import QLabel, QPushButton

try:
    from .score_mode_window import ScoreModeWindow
except ImportError:
    from score_mode_window import ScoreModeWindow


class IntegratedWindow(ScoreModeWindow):
    def __init__(
        self,
        ui_path,
        exercise="squats",
        webcam=0,
        base_url="http://127.0.0.1:8793",
    ):
        super().__init__(ui_path, exercise, webcam, base_url)
        self.setWindowTitle("AIFit · 운동 교정 + 자세 점수")
        self.correction_notice = QLabel("교정 모델 준비 대기 중")
        self.correction_notice.setWordWrap(True)
        layout = self.centralWidget().layout()
        # Keep the original video's top edge and existing overlay layout.
        self.notice.hide()
        self.quality_notice.hide()
        self.correction_notice.hide()
        self.saved_notice.hide()
        self.session_pause = QPushButton("일시정지")
        self.session_pause.clicked.connect(self.pause_resume)
        layout.addWidget(self.session_pause)
        self.pauseResumeButton.setText("통합 측정 시작")

    def toggle(self):
        if self.last_state.get("running"):
            self.control("/api/stop", {})
        else:
            self.control(
                "/api/start",
                dict(
                    mode="combined",
                    exercise=self.exercise,
                    camera="webcam",
                    webcam=str(self.webcam),
                    view="front",
                ),
            )

    def pause_resume(self):
        if self.last_state.get("running"):
            self.control(
                (
                    "/api/resume"
                    if self.last_state.get("paused")
                    else "/api/pause"
                ),
                {},
            )

    def render_state(self, state):
        super().render_state(state)
        if not hasattr(self, "correction_notice"):
            return
        correction = state.get("correction", {})
        result = correction.get("result") or {}
        feedbacks = result.get("feedbacks", [])
        text = (
            " / ".join(feedbacks)
            if feedbacks
            else correction.get("message", "교정 입력을 기다립니다.")
        )
        if state.get("paused"):
            text = "일시정지 · 완료 횟수와 점수는 유지됩니다."
        if correction.get("error"):
            text = "교정 오류: " + correction["error"]
        self.correction_notice.setText("운동 교정 · " + text)
        self.feedbackLabel.setText(text)
        self.feedbackLabel.setWordWrap(True)
        self.session_pause.setText(
            "재개" if state.get("paused") else "일시정지"
        )
        self.session_pause.setEnabled(bool(state.get("running")))
        self.notice.setText(
            "운동 교정 + 자세 점수 · MediaPipe · 단일 웹캠 · 얼굴 모자이크"
        )
        if not state.get("running"):
            summary = state.get("integrated_summary")
            if summary:
                self.saved_notice.setText(
                    "교정·점수·각 관절 기록 연결 저장 완료 · "
                    + summary["session_id"]
                )
