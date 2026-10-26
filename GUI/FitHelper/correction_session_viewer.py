"""저장 세션의 요약·무결성·프레임별 33개 관절과 교정 출력을 재조회합니다."""

import json
from urllib.parse import quote

from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)


class CorrectionSessionViewer(QDialog):
    def __init__(self, base_url, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.setWindowTitle("저장 운동 결과 · 관절·교정 재조회")
        self.resize(900, 720)
        self.network = QNetworkAccessManager(self)
        self.sessions = QComboBox()
        self.refresh_button = QPushButton("목록 새로고침")
        self.open_button = QPushButton("선택 세션 열기")
        self.frame = QSpinBox()
        self.frame.setRange(0, 0)
        self.notice = QLabel("저장 목록 조회 중")
        self.notice.setWordWrap(True)
        self.details = QTextBrowser()
        self.overview = QTextBrowser()
        self.overview.setMaximumHeight(180)
        self.joints = QTableWidget(0, 5)
        self.joints.setHorizontalHeaderLabels(
            ["관절 번호", "x", "y", "z", "신뢰도"]
        )
        self.joints.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.last_response = None
        self.sequence = 0
        self.summary = {}
        top = QHBoxLayout()
        for widget in (self.sessions, self.refresh_button, self.open_button):
            top.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.notice)
        row = QHBoxLayout()
        row.addWidget(QLabel("분석 프레임 번호 (0부터)"))
        row.addWidget(self.frame)
        layout.addLayout(row)
        tabs = QTabWidget()
        from PyQt6.QtWidgets import QWidget

        readable = QWidget()
        readable_layout = QVBoxLayout(readable)
        readable_layout.addWidget(self.overview)
        readable_layout.addWidget(self.joints)
        tabs.addTab(readable, "관절·교정 결과")
        tabs.addTab(self.details, "원시 기록")
        layout.addWidget(tabs)
        self.refresh_button.clicked.connect(self.refresh)
        self.open_button.clicked.connect(self.open_session)
        self.frame.valueChanged.connect(lambda: self.load_frame(False))
        self.refresh()

    def get(self, path, callback):
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setTransferTimeout(15000)
        reply = self.network.get(request)

        def finished():
            try:
                if reply.error() != QNetworkReply.NetworkError.NoError:
                    self.notice.setText("재조회 실패: " + reply.errorString())
                    return
                callback(json.loads(bytes(reply.readAll())))
            except (ValueError, KeyError) as exc:
                self.notice.setText("저장 응답 오류: " + str(exc))
            finally:
                reply.deleteLater()

        reply.finished.connect(finished)

    def refresh(self):
        self.get("/api/sessions", self.show_list)

    def show_list(self, data):
        self.sessions.clear()
        for info in data["sessions"]:
            self.sessions.addItem(
                f"{info['session_id']} · {info['exercise']} · {info['status']}",
                info["session_id"],
            )
        self.notice.setText(f"저장 세션 {self.sessions.count()}개")
        if self.sessions.count():
            self.open_session()

    def open_session(self):
        self.frame.blockSignals(True)
        self.frame.setValue(0)
        self.frame.blockSignals(False)
        self.load_frame(True)

    def load_frame(self, verify):
        session = self.sessions.currentData()
        if not session:
            return
        self.sequence += 1
        sequence = self.sequence
        index = self.frame.value()

        def display(data):
            if sequence != self.sequence:
                return
            self.last_response = data
            self.summary = data["summary"]
            info = self.summary
            self.frame.blockSignals(True)
            self.frame.setMaximum(max(0, info.get("frame_count", 1) - 1))
            self.frame.blockSignals(False)
            if verify:
                integrity = data["integrity_matches"]
                text = (
                    "일치"
                    if integrity is True
                    else "불일치" if integrity is False else "미확정 세션"
                )
                self.notice.setText(
                    f"{info['status']} · 분석 프레임 {info.get('frame_count', '?')}개 · SHA-256 {text}"
                )
            self.details.setPlainText(
                json.dumps(data, ensure_ascii=False, indent=2)
            )
            rows = data["frames"]
            current = rows[0] if rows else {}
            prediction = current.get("prediction")
            text = [
                f"운동: {info['exercise']} / 전체 분석: {info.get('frame_count', '?')}프레임",
                f"관절 관측: {info.get('observed_pose_frames', '?')} / 관절 누락: {info.get('missing_pose_frames', '?')}",
                f"교정 판정: {info.get('prediction_count', '?')}회 (운동 반복 횟수와 다릅니다.)",
                f"현재 프레임: {current.get('index', '-')} / 상태: {current.get('status', '-')}",
            ]
            if prediction:
                text.append(
                    "감지 오류: "
                    + (
                        ", ".join(prediction.get("detected_errors", []))
                        or "선택 오류 미검출"
                    )
                )
                text.extend(prediction.get("feedbacks", []))
            else:
                text.append("이 프레임에는 교정 판정이 없습니다.")
            self.overview.setPlainText("\n".join(text))
            points = current.get("landmarks") or []
            self.joints.setRowCount(len(points))
            for index, values in enumerate(points):
                for column, value in enumerate([index] + values):
                    self.joints.setItem(
                        index,
                        column,
                        QTableWidgetItem(
                            str(index) if column == 0 else f"{value:.6f}"
                        ),
                    )

        self.get(
            f"/api/sessions/{quote(session)}?offset={index}&limit=1&verify={int(verify)}",
            display,
        )
