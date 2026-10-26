"""팀 기본 운동 화면에 기존 자세 점수만 연결합니다. 교정 모델은 실행하지 않습니다."""

import json
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QByteArray, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from Server.ExerciseEvaluation.paths import DATA_ROOT

try:
    from .correction_session_store import atomic_json
except ImportError:
    from correction_session_store import atomic_json


def item_lines(score):
    return [
        f"{item.get('title', item['id'])}: AI {item.get('ai_points') if item.get('ai_points') is not None else '미평가'} / 규칙 {item.get('rule_points') if item.get('rule_points') is not None else '미평가'}"
        for item in score.get("items", [])
    ]


class ScoreModeWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(
        self,
        ui_path,
        exercise="squats",
        webcam=0,
        base_url="http://127.0.0.1:8790",
        sessions_root=None,
    ):
        super().__init__()
        uic.loadUi(str(ui_path), self)
        self.videoLabel.setText("웹캠 · MediaPipe Pose Landmarker")
        self.base_url = base_url
        self.exercise = exercise
        self.webcam = webcam
        self.sessions_root = (
            Path(sessions_root)
            if sessions_root
            else DATA_ROOT / "score-mode-sessions"
        )
        self.network = QNetworkAccessManager(self)
        self.pending = set()
        self.controlling = False
        self.closing = False
        self.allow_close = False
        self.generation = 0
        self.last_state = {}
        self.saved_path = None
        self.frames_received = 0
        self.setWindowTitle("AIFit · 자세 점수 모드")
        # Keep the team's original workout UI as a fixed video/overlay panel.
        original = self.takeCentralWidget()
        original.setFixedSize(640, 480)
        self.result = QTextBrowser()
        self.result.setMinimumWidth(300)
        self.notice = QLabel("자세 점수 준비 중 · 웹캠")
        self.notice.setWordWrap(True)
        self.saved_notice = QLabel(
            "중지하거나 돌아가면 완료 횟수·점수를 파일로 저장합니다."
        )
        self.saved_notice.setWordWrap(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.notice)
        self.quality_notice = QLabel(
            "전체 항목을 채점할 수 있는 준비 자세를 확인합니다."
        )
        self.quality_notice.setWordWrap(True)
        layout.addWidget(self.quality_notice)
        row = QHBoxLayout()
        row.addWidget(original)
        row.addWidget(self.result)
        layout.addLayout(row)
        layout.addWidget(self.saved_notice)
        self.history_button = QPushButton("관절·결과 기록 열기")
        self.history_button.clicked.connect(self.open_recordings)
        layout.addWidget(self.history_button)
        self.setCentralWidget(container)
        self.resize(1000, 570)
        self.pauseResumeButton.setText("측정 시작")
        self.endWorkoutButton.setText("모드 선택으로 돌아가기")
        self.counterLabel.setText("검출 0회")
        self.liveScoreLabel.setText("미평가")
        self.pauseResumeButton.clicked.connect(self.toggle)
        self.endWorkoutButton.clicked.connect(self.close)
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.request("/api/state"))
        self.timer.start(250)
        self.request("/api/state")

    def toggle(self):
        if self.last_state.get("running"):
            self.control("/api/stop", {})
        else:
            self.control(
                "/api/start",
                dict(
                    exercise=self.exercise,
                    mode="score",
                    camera="webcam",
                    webcam=str(self.webcam),
                    view="front",
                ),
            )

    def control(self, path, payload):
        if self.controlling:
            return
        self.controlling = True
        self.generation += 1
        self.pauseResumeButton.setEnabled(False)
        self.endWorkoutButton.setEnabled(False)
        self.request(path, payload)

    def request(self, path, payload=None):
        if path in self.pending:
            return
        self.pending.add(path)
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setTransferTimeout(20000)
        if payload is None:
            reply = self.network.get(request)
        else:
            request.setHeader(
                QNetworkRequest.KnownHeaders.ContentTypeHeader,
                "application/json",
            )
            reply = self.network.post(
                request, QByteArray(json.dumps(payload).encode())
            )
        generation = self.generation
        reply.finished.connect(
            lambda: self.receive(path, payload is not None, generation, reply)
        )

    def receive(self, path, control, generation, reply):
        self.pending.discard(path)
        success = False
        try:
            if generation != self.generation:
                return
            if reply.error() != QNetworkReply.NetworkError.NoError:
                raise ValueError(reply.errorString())
            data = bytes(reply.readAll())
            if path == "/api/frame.jpg":
                image = QImage.fromData(data)
                if not image.isNull():
                    self.frames_received += 1
                    self.videoLabel.setPixmap(
                        QPixmap.fromImage(image).scaled(
                            self.videoLabel.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    )
            else:
                self.render_state(json.loads(data))
                if path == "/api/stop":
                    if self.last_state.get("running"):
                        raise ValueError(
                            "점수 측정이 아직 중지되지 않았습니다."
                        )
                    self.save_result()
            success = True
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.notice.setText("자세 점수 연결·저장 확인: " + str(exc))
            if self.closing:
                self.closing = False
                self.timer.start(250)
        finally:
            reply.deleteLater()
            if control:
                self.controlling = False
                self.pauseResumeButton.setEnabled(True)
                self.endWorkoutButton.setEnabled(True)
                if self.closing and success:
                    if path == "/api/stop":
                        self.allow_close = True
                        self.close()
                    else:
                        self.control("/api/stop", {})

    def render_state(self, state):
        self.last_state = state
        score = state.get("score", {})
        movement = score.get("movement", {})
        count = movement.get("count", 0)
        self.counterLabel.setText(f"검출 {count}회")
        self.pauseResumeButton.setText(
            "중지·저장" if state.get("running") else "측정 시작"
        )
        observation = movement.get("observation") or {}
        summary = score.get("assessment_summary") or {}
        scored = summary.get(
            "scored_repetitions",
            sum(
                all(
                    r.get("score_100", {}).get(m, {}).get("total") is not None
                    for m in ("ai", "rule")
                )
                for r in score.get("repetitions", [])
            ),
        )
        readiness = score.get("scoring_readiness") or {}
        quality = score.get("joint_quality") or {}
        ready = readiness.get("prepared", False)
        current = readiness.get("current_all_items_observed", False)
        if state.get("running"):
            if not ready:
                message = f"채점 준비 {readiness.get('ready_frames',0)}/{readiness.get('required_frames',6)} · 양쪽 무릎·골반·어깨를 함께 확인합니다."
            elif not current:
                message = (
                    "추적 복구 중 · 몸 전체가 화면 안쪽에 보이도록 서십시오."
                )
            else:
                message = "전체 항목 관절 관측 중 · 서 있는 준비 자세를 유지한 뒤 운동하십시오."
            knee = []
            for j, title in [("25", "왼쪽 무릎"), ("26", "오른쪽 무릎")]:
                item = quality.get("joints", {}).get(j, {})
                visibility = item.get("visibility")
                presence = item.get("presence")
                why = item.get("reasons", [])
                detail = (
                    "화면 안쪽으로 이동"
                    if any(x in why for x in ("outside_frame", "frame_border"))
                    else "추적 복구 대기" if why else "관측됨"
                )
                knee.append(
                    f"{title}: {detail}"
                    + (
                        f" (가시성 {visibility:.0%} / 추적 {presence:.0%})"
                        if visibility is not None and presence is not None
                        else ""
                    )
                )
            self.quality_notice.setText(message + "\n" + " · ".join(knee))
            self.feedbackLabel.setText(
                message
                if not ready or not current
                else observation.get("message")
                or movement.get("phase", "측정 대기")
            )
        else:
            self.feedbackLabel.setText(
                f"측정 종료 · 검출 {count}회 / 전체 채점 확정 {scored}회"
            )
            self.quality_notice.setText(
                "이전 프레임의 관측 경고는 종료된 운동 전체의 판정이 아닙니다."
            )
        self.notice.setText(
            f"자세 점수 · MediaPipe · {self.exercise} · 웹캠 {self.webcam} · {state.get('capture_fps', 0):.1f} fps · 얼굴 모자이크 켜짐"
        )
        lines = [f"동작 검출 {count}회 / 전체 항목 채점 확정 {scored}회"]
        if score.get("error"):
            lines.append(score["error"])
        repetitions = score.get("repetitions", [])
        for rep in repetitions:
            complete = rep.get(
                "assessment_complete",
                all(
                    rep.get("score_100", {}).get(m, {}).get("total") is not None
                    for m in ("ai", "rule")
                ),
            )
            lines.append(
                f"\n{rep['number']}회 · "
                + ("전체 채점 확정" if complete else "추적·관측 재확인 필요")
            )
            if not complete:
                lines.append(
                    "전체 항목이 확보된 채점 완료 결과가 아닙니다. 준비 자세와 구도를 확인하고 다시 수행하십시오."
                )
            if rep.get("error"):
                lines.append("평가 보류: " + rep["error"])
            else:
                assessment = rep.get("score_100", {})
                for method, title in [("ai", "AI"), ("rule", "규칙")]:
                    value = assessment.get(method, {}).get("total")
                    lines.append(
                        f"{title} 총점: {value if value is not None else '미평가'}"
                    )
                lines.extend(item_lines(assessment))
        current = score.get("live_score_100")
        if current:
            lines.append("\n진행 중 · 잠정 항목")
            lines.extend(item_lines(current))
        last = repetitions[-1].get("score_100", {}) if repetitions else {}
        total = last.get("ai", {}).get("total")
        self.liveScoreLabel.setText(
            f"{total}점"
            if total is not None
            else "재확인" if count else "준비 중"
        )
        self.result.setPlainText("\n".join(lines))
        if state.get("running"):
            self.request("/api/frame.jpg")

    def open_recordings(self):
        try:
            from .score_recording_viewer import ScoreRecordingViewer
        except ImportError:
            from score_recording_viewer import ScoreRecordingViewer
        self.history = ScoreRecordingViewer(self.base_url, self)
        self.history.show()

    def save_result(self):
        session = self.last_state.get("session_id")
        if not session:
            return
        if not isinstance(session, str) or not all(
            c in "0123456789abcdef" for c in session
        ):
            raise ValueError("잘못된 점수 세션 ID입니다.")
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.saved_path = self.sessions_root / (session + ".json")
        atomic_json(self.saved_path, self.last_state)
        self.saved_notice.setText(
            "횟수·점수 저장 완료: " + str(self.saved_path)
        )

    def closeEvent(self, event):
        if self.allow_close:
            self.timer.stop()
            self.closed.emit()
            event.accept()
            return
        event.ignore()
        self.closing = True
        self.timer.stop()
        if not self.controlling:
            self.control("/api/stop", {})
