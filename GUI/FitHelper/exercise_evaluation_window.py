"""팀 PyQt6 환경에서 ESP32·웹캠 교정·점수 실행기를 조작합니다.

기존 Device/ImageStreamer/ESP32Cam.py의 QLabel·QImage·QPixmap 표시
방식을 사용하되 수신·평가는 이미 실행 중인 공통 평가 엔진에 맡깁니다.
카메라를 중복으로 열지 않으며 기존 로그인·운동 화면을 수정하지 않습니다.
"""

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
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from Server.ExerciseEvaluation.contracts import EXERCISES

STATUS = {
    "idle": "측정 대기",
    "running": "측정 중",
    "connecting": "카메라 연결 중",
    "loading": "모델 준비 중",
    "warming_up": "연속 관측 수집 중",
    "pose_not_visible": "관절 관측 부족",
    "collecting": "반복 관측 중",
    "predicting": "교정 결과",
    "not_requested": "선택하지 않은 평가",
    "stale": "최신 관측 대기",
    "unavailable": "실행 불가",
    "camera_unavailable": "카메라 연결 대기",
    "stopped": "측정 종료",
}


class ExerciseEvaluationWindow(QMainWindow):
    """비동기 HTTP로 화면을 갱신하며 UI 스레드에서 모델을 추론하지 않습니다."""

    def __init__(self, base_url="http://127.0.0.1:8790"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.network = QNetworkAccessManager(self)
        self.pending = set()
        self.initialized = False
        self.closing = False
        self.allow_close = False
        self.setWindowTitle("AIFit · ESP32 / 웹캠 운동 교정·점수")
        self.resize(1180, 780)
        self.exercise = QComboBox()
        for code, title in EXERCISES.items():
            self.exercise.addItem(title, code)
        self.mode = QComboBox()
        for title, code in [
            ("교정 + 점수", "combined"),
            ("교정만", "correction"),
            ("점수만", "score"),
        ]:
            self.mode.addItem(title, code)
        self.camera = QComboBox()
        for title, code in [
            ("웹캠", "webcam"),
            ("ESP32", "esp32"),
            ("ESP32 우선 · 실패 시 웹캠", "auto"),
        ]:
            self.camera.addItem(title, code)
        self.webcam = QSpinBox()
        self.webcam.setRange(0, 32)
        self.esp32 = QLineEdit("http://192.168.15.208:81/stream")
        form = QFormLayout()
        for title, field in [
            ("운동", self.exercise),
            ("평가 방식", self.mode),
            ("카메라", self.camera),
            ("웹캠 번호", self.webcam),
            ("ESP32 주소", self.esp32),
        ]:
            form.addRow(title, field)
        self.start_button = QPushButton("선택한 설정으로 시작")
        self.stop_button = QPushButton("중지")
        self.start_button.clicked.connect(self.start_evaluation)
        self.stop_button.clicked.connect(lambda: self.request("/api/stop", {}))
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        self.notice = QLabel("평가 엔진 연결 대기 중입니다.")
        self.notice.setWordWrap(True)
        self.image = QLabel("웹캠 영상 대기 중")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 360)
        self.image.setStyleSheet("background: #18212c; color: white")
        self.result = QTextBrowser()
        content = QHBoxLayout()
        content.addWidget(self.image, 1)
        content.addWidget(self.result, 1)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.notice)
        layout.addLayout(content, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.request("/api/state"))
        self.timer.start(400)
        self.request("/api/state")

    def start_evaluation(self):
        """선택을 엔진에 전달하며 소스·모드 변경은 새 세션을 시작합니다."""
        self.request(
            "/api/start",
            {
                "exercise": self.exercise.currentData(),
                "mode": self.mode.currentData(),
                "camera": self.camera.currentData(),
                "webcam": str(self.webcam.value()),
                "esp32": self.esp32.text().strip(),
                "view": "front",
            },
        )

    def request(self, path, body=None):
        """경로별 요청을 하나로 제한하고 응답은 Qt 이벤트 루프에서 처리합니다."""
        if path in self.pending:
            return
        self.pending.add(path)
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setTransferTimeout(8000)
        if body is None:
            reply = self.network.get(request)
        else:
            request.setHeader(
                QNetworkRequest.KnownHeaders.ContentTypeHeader,
                "application/json",
            )
            reply = self.network.post(
                request, QByteArray(json.dumps(body).encode())
            )
        reply.finished.connect(lambda: self.finished(path, reply))

    def finished(self, path, reply):
        """오류와 미평가를 구분해 표시하며 이미지 버퍼의 수명을 복사로 분리합니다."""
        self.pending.discard(path)
        try:
            payload = bytes(reply.readAll())
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.notice.setText(
                    f"평가 엔진 연결 확인: {reply.errorString()} · {self.base_url}"
                )
                return
            if path == "/api/frame.jpg":
                image = QImage.fromData(payload)
                if not image.isNull():
                    self.image.setPixmap(
                        QPixmap.fromImage(image.copy()).scaled(
                            self.image.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation,
                        )
                    )
            else:
                self.render_state(json.loads(payload))
        except (ValueError, TypeError, KeyError) as error:
            self.notice.setText(f"응답 확인이 필요합니다: {error}")
        finally:
            reply.deleteLater()
            if self.closing and path == "/api/stop":
                self.finish_close()

    def render_state(self, state):
        """완료 점수와 잠정 항목을 분리하고 미평가 값을 그대로 표시합니다."""
        options = state.get("options", {})
        if not self.initialized and options:
            for field, key in [
                (self.exercise, "exercise"),
                (self.mode, "mode"),
                (self.camera, "camera"),
            ]:
                index = field.findData(options.get(key))
                if index >= 0:
                    field.setCurrentIndex(index)
            self.webcam.setValue(int(options.get("webcam", 0)))
            self.esp32.setText(options.get("esp32", self.esp32.text()))
            self.initialized = True
        fps = state.get("capture_fps", 0)
        self.notice.setText(
            f"{STATUS.get(state.get('status'), '대기')} · {state.get('camera', '웹캠')} · {fps:.1f} fps · {state.get('camera_error', '')}"
        )
        correction, score = state.get("correction", {}), state.get("score", {})
        lines = [
            "운동 교정",
            f"상태: {STATUS.get(correction.get('status'), '대기')}",
            "선택 오류 미검출은 전체 정상 판정이 아닙니다.",
        ]
        lines.extend((correction.get("result") or {}).get("feedbacks", []))
        if correction.get("error"):
            lines.append(correction["error"])
        lines.extend(
            [
                "",
                "자세 점수",
                f"상태: {STATUS.get(score.get('status'), '대기')}",
            ]
        )
        movement = score.get("movement", {})
        lines.append(
            f"완료 {movement.get('count', 0)}회 · {movement.get('phase', '')}"
        )
        if score.get("error"):
            lines.append(score["error"])
        live = score.get("live_score_100")
        if live:
            lines.append("진행 중 · 잠정 항목")
            lines.extend(item_lines(live))
        for rep in score.get("repetitions", []):
            lines.append(f"\n{rep['number']}회")
            if rep.get("error"):
                lines.append(f"평가 보류: {rep['error']}")
            else:
                lines.extend(item_lines(rep.get("score_100", {})))
        self.result.setPlainText("\n".join(lines))
        if state.get("running") and state.get("status") == "running":
            self.request("/api/frame.jpg")
        else:
            self.image.clear()
            self.image.setText("웹캠 영상 대기 중")

    def closeEvent(self, event):
        """닫기 요청에서 측정을 중지하며 서버 응답 지연에는 종료 제한을 둡니다."""
        if self.allow_close:
            event.accept()
            return
        event.ignore()
        self.closing = True
        self.timer.stop()
        self.request("/api/stop", {})
        QTimer.singleShot(2000, self.finish_close)

    def finish_close(self):
        """이미 시작한 종료를 한 번만 마감합니다."""
        self.allow_close = True
        self.close()


def item_lines(score):
    """미평가를 0점으로 바꾸지 않고 관측 항목의 점수를 표시합니다."""
    return [
        f"{item.get('title', item['id'])}: AI {item.get('ai_points') if item.get('ai_points') is not None else '미평가'} / 규칙 {item.get('rule_points') if item.get('rule_points') is not None else '미평가'}"
        for item in score.get("items", [])
    ]


def main():
    """실행 중인 공통 평가 엔진을 팀 PyQt6 화면에 연결합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8790")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = ExerciseEvaluationWindow(args.url)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
