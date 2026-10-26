"""점수 모드의 관절·신뢰도·제외 사유와 완료 결과를 재조회합니다."""

import json

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
    QTextBrowser,
    QVBoxLayout,
)


class ScoreRecordingViewer(QDialog):
    def __init__(self, base_url, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.network = QNetworkAccessManager(self)
        self.setWindowTitle("자세 점수 · 관절 및 채점 기록")
        self.resize(1050, 760)
        self.sessions = QComboBox()
        self.refresh_button = QPushButton("목록 새로고침")
        self.open_button = QPushButton("기록 열기")
        self.frame = QSpinBox()
        self.frame.setRange(0, 0)
        self.notice = QLabel("저장 목록을 확인합니다.")
        self.notice.setWordWrap(True)
        self.details = QTextBrowser()
        self.details.setMaximumHeight(180)
        self.joints = QTableWidget(0, 7)
        self.joints.setHorizontalHeaderLabels(
            ["관절", "x", "y", "z", "가시성", "추적", "제외 사유"]
        )
        self.joints.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.last_response = None
        self.sequence = 0
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        for widget in (
            self.sessions,
            self.refresh_button,
            self.open_button,
            self.frame,
        ):
            row.addWidget(widget)
        layout.addLayout(row)
        layout.addWidget(self.notice)
        layout.addWidget(self.details)
        layout.addWidget(self.joints)
        self.refresh_button.clicked.connect(self.refresh)
        self.open_button.clicked.connect(lambda: self.read(True))
        self.frame.valueChanged.connect(lambda: self.read(False))
        self.refresh()

    def get(self, path, callback):
        request = QNetworkRequest(QUrl(self.base_url + path))
        request.setTransferTimeout(20000)
        reply = self.network.get(request)

        def finished():
            try:
                if reply.error() != QNetworkReply.NetworkError.NoError:
                    raise ValueError(reply.errorString())
                callback(json.loads(bytes(reply.readAll())))
            except (ValueError, KeyError) as exc:
                self.notice.setText("기록 조회 실패: " + str(exc))
            finally:
                reply.deleteLater()

        reply.finished.connect(finished)

    def refresh(self):
        def show(data):
            self.sessions.clear()
            for record in data["recordings"]:
                self.sessions.addItem(
                    f"{record['recording_id']} · {record.get('status')}",
                    record["recording_id"],
                )
            if self.sessions.count():
                self.frame.blockSignals(True)
                self.frame.setValue(0)
                self.frame.blockSignals(False)
                self.read(True)

        self.get("/api/recordings", show)

    def read(self, verify):
        recording = self.sessions.currentData()
        if not recording:
            return
        self.sequence += 1
        sequence = self.sequence

        def show(data):
            if sequence != self.sequence:
                return
            self.last_response = data
            summary = data["summary"]
            row = data.get("frame") or {}
            self.frame.blockSignals(True)
            self.frame.setMaximum(max(0, summary.get("frame_count", 1) - 1))
            self.frame.blockSignals(False)
            score = summary.get("score", {})
            counts = score.get("assessment_summary", {})
            if verify:
                integrity = (
                    "일치"
                    if data["integrity_matches"] is True
                    else (
                        "불일치"
                        if data["integrity_matches"] is False
                        else "종료 미확정"
                    )
                )
                self.notice.setText(
                    f"관절 {summary.get('frame_count','?')}프레임 · 기록 검증 {integrity} · 검출 {counts.get('detected_repetitions',0)}회 / 전체 채점 확정 {counts.get('scored_repetitions',0)}회"
                )
            self.details.setPlainText(
                json.dumps(
                    dict(
                        tick=row.get("tick"),
                        extraction=row.get("extraction_source"),
                        readiness=row.get("scoring_readiness"),
                        assessment=counts,
                        repetitions=score.get("repetitions", []),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raw = row.get("raw_selected_landmarks") or []
            reasons = row.get("quality", {}).get(
                "joint_rejection_reasons", [[] for _ in range(33)]
            )
            self.joints.setRowCount(len(raw))
            for j, values in enumerate(raw):
                text = (
                    [str(j)]
                    + ["미검출" if v is None else f"{v:.5f}" for v in values]
                    + [", ".join(reasons[j])]
                )
                for col, value in enumerate(text):
                    self.joints.setItem(j, col, QTableWidgetItem(value))

        self.get(
            f"/api/recordings/{recording}?offset={self.frame.value()}&verify={int(verify)}",
            show,
        )
