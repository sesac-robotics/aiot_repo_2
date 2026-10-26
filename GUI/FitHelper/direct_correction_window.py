"""체형분석·회원 DB 없이 팀원의 교정 HTTP 엔진을 사용하는 별도 화면입니다."""

import argparse
import json
import sys

from PyQt6.QtCore import QByteArray, Qt, QTimer, QUrl
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

EXERCISES = {
    "squats": "스쿼트",
    "pushups": "푸시업",
    "elbow_plank": "엘보 플랭크",
    "ytw": "YTW",
    "shoulder_press": "숄더 프레스",
}
STATUS = {
    "idle": "대기",
    "loading_model": "모델 준비 중",
    "connecting": "웹캠 연결 중",
    "warming_up": "연속 자세 수집 중",
    "pose_not_visible": "필요 관절 관측 부족",
    "predicting": "교정 분석 중",
    "error": "실행 오류",
}


class DirectCorrectionWindow(QMainWindow):
    """웹캠은 서버 한 곳에서 열고 GUI는 비동기로 상태와 JPEG만 수신합니다."""

    def __init__(self, base_url="http://127.0.0.1:8782"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.network = QNetworkAccessManager(self)
        self.pending = set()
        self.controlling = False
        self.closing = False
        self.allow_close = False
        self.last_state = {}
        self.frames_received = 0
        self.last_frame_sequence = -1
        self.generation = 0
        self.setWindowTitle("AIFit · 운동 선택 · 웹캠 교정")
        self.resize(1080, 760)
        self.exercise = QComboBox()
        for code, title in EXERCISES.items():
            self.exercise.addItem(title, code)
        self.webcam = QSpinBox()
        self.webcam.setRange(0, 32)
        self.start_button = QPushButton("선택한 운동 시작")
        self.stop_button = QPushButton("중지·저장")
        self.record = QCheckBox("결과 자동저장 (시작 시 적용)")
        self.record.setChecked(True)
        self.history_button = QPushButton("저장 결과 열기")
        self.history_button.clicked.connect(self.open_history)
        self.start_button.clicked.connect(self.start_correction)
        self.stop_button.clicked.connect(lambda: self.control("/api/stop", {}))
        controls = QHBoxLayout()
        for widget in (
            QLabel("운동"),
            self.exercise,
            QLabel("웹캠 번호"),
            self.webcam,
            self.start_button,
            self.stop_button,
        ):
            controls.addWidget(widget)
        self.notice = QLabel("교정 서버 연결 대기 중")
        self.notice.setWordWrap(True)
        self.image = QLabel("웹캠 영상 대기 중")
        self.image.setMinimumSize(600, 400)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet("background: #18212c; color: white;")
        self.result = QTextBrowser()
        self.result.setMinimumWidth(260)
        content = QHBoxLayout()
        content.addWidget(self.image, 3)
        content.addWidget(self.result, 2)
        layout = QVBoxLayout()
        layout.addWidget(
            QLabel(
                "체형분석 없이 운동을 선택해 교정합니다. 카메라: 웹캠 / 영상 녹화: 꺼짐"
            )
        )
        layout.addLayout(controls)
        saving = QHBoxLayout()
        saving.addWidget(self.record)
        saving.addWidget(self.history_button)
        layout.addLayout(saving)
        self.saved_notice = QLabel(
            "저장 대기: 관절·교정 결과를 자동 저장합니다."
        )
        self.saved_notice.setWordWrap(True)
        layout.addWidget(self.saved_notice)
        layout.addWidget(self.notice)
        layout.addLayout(content)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(250)
        self.poll()

    def poll(self):
        if not self.closing:
            self.request("/api/state")

    def start_correction(self):
        if not self.last_state.get("full_pose_recording"):
            self.notice.setText(
                "관절 저장 서버에 연결되지 않았습니다. 새 실행기로 서버를 실행하십시오."
            )
            return
        self.control(
            "/api/start",
            {
                "exercise": self.exercise.currentData(),
                "source": str(self.webcam.value()),
                "record": self.record.isChecked(),
            },
        )

    def open_history(self):
        try:
            from .correction_session_viewer import CorrectionSessionViewer
        except ImportError:
            from correction_session_viewer import CorrectionSessionViewer

        self.history = CorrectionSessionViewer(self.base_url, self)
        self.history.show()

    def control(self, path, payload):
        if self.controlling:
            return
        self.controlling = True
        self.generation += 1
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.exercise.setEnabled(False)
        self.webcam.setEnabled(False)
        self.record.setEnabled(False)
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
            lambda: self.receive(path, payload is not None, reply, generation)
        )

    def receive(self, path, control, reply, generation):
        self.pending.discard(path)
        try:
            if generation != self.generation:
                return
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.notice.setText(
                    "교정 서버 연결 실패: " + reply.errorString()
                )
                self.last_state = {}
                self.image.clear()
                self.image.setText("최신 영상 수신 중단")
                self.result.setPlainText("최신 교정 결과를 받을 수 없습니다.")
                if self.closing:
                    self.closing = False
                    self.timer.start(250)
                    self.notice.setText(
                        "중지 확인에 실패했습니다. 서버 연결을 확인한 뒤 다시 닫으십시오."
                    )
                return
            data = bytes(reply.readAll())
            if path == "/api/frame.jpg":
                image = QImage.fromData(data, "JPG")
                if not image.isNull():
                    self.frames_received += 1
                    self.image.setPixmap(
                        QPixmap.fromImage(image).scaled(
                            self.image.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
            else:
                state = json.loads(data)
                self.last_state = state
                self.render(state)
                if not self.closing:
                    self.request("/api/frame.jpg")
                if (
                    self.closing
                    and path == "/api/stop"
                    and not state.get("running")
                ):
                    self.allow_close = True
                    self.close()
        except (ValueError, TypeError) as exc:
            self.notice.setText("서버 응답을 해석할 수 없습니다: " + str(exc))
        finally:
            if control:
                self.controlling = False
                for widget in (
                    self.start_button,
                    self.stop_button,
                    self.exercise,
                    self.webcam,
                    self.record,
                ):
                    widget.setEnabled(True)
            reply.deleteLater()

    def render(self, state):
        if state.get("session_id"):
            finalized = state.get("session_finalized")
            marker = (
                "저장 완료"
                if finalized
                else "저장 중" if state.get("record") else "저장 미확정"
            )
            self.saved_notice.setText(
                f"{marker} · {state.get('saved_frames', 0)}프레임 · {state['session_path']}"
            )
        if state.get("recording_error"):
            self.saved_notice.setText("저장 실패: " + state["recording_error"])
        status = state.get("status", "idle")
        exercise = EXERCISES.get(state.get("exercise"), "선택 대기")
        self.notice.setText(
            f"{exercise} · {STATUS.get(status, status)} · "
            f"카메라 {state.get('capture_fps', 0):.1f} fps · "
            f"처리 {state.get('processing_fps', 0):.1f} fps · "
            f"관측 {state.get('buffer_frames', 0)}/{state.get('required_frames', 32)}"
        )
        prediction = state.get(
            "result"
        )  # 과거 last_result를 현재 판정으로 표시하지 않습니다.
        lines = []
        if status == "predicting" and prediction:
            errors = prediction.get("detected_errors", [])
            lines.append(
                "감지된 오류: "
                + (", ".join(errors) if errors else "선택 오류 미검출")
            )
            lines.extend(prediction.get("feedbacks", []))
            lines.append(
                "선택 오류 미검출은 모든 자세가 정상이라는 의미가 아닙니다."
            )
        else:
            lines.append("현재 교정 판정 대기 중입니다.")
            lines.append(
                "전신과 해당 운동에 필요한 관절이 화면에 보이도록 위치를 조정하십시오."
            )
        if state.get("error"):
            lines.append(state["error"])
        self.result.setPlainText("\n\n".join(lines))

    def closeEvent(self, event):
        if self.allow_close:
            self.timer.stop()
            event.accept()
            return
        event.ignore()
        if not self.controlling and not self.closing:
            self.closing = True
            self.timer.stop()
            self.control("/api/stop", {})


def main():
    parser = argparse.ArgumentParser(description="체형분석 없는 웹캠 교정 GUI")
    parser.add_argument("--server", default="http://127.0.0.1:8782")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = DirectCorrectionWindow(args.server)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
