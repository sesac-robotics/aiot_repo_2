"""교정·점수 실행을 조정하는 application 서비스입니다.

완료 횟수는 기존 점수 추적기가 소유합니다. 이 서비스는 별도 카운터를
증가시키지 않고 실행 수명과 두 분석 기록의 연결만 관리합니다.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from Server.ExerciseEvaluation.application.ports import (
    CorrectionEnginePort,
    ScoreCoordinatorPort,
    SessionRepository,
)
from Server.ExerciseEvaluation.contracts import CORRECTION_CODES, validate
from Server.ExerciseEvaluation.domain.lifecycle import (
    SessionLifecycle,
    SessionPhase,
)

LOGGER = logging.getLogger(__name__)


class IntegratedSession:
    """단일 제어 잠금으로 교정·점수 자원의 수명과 저장 순서를 조정합니다.

    Args:
        engine: 기존 교정 엔진을 감싼 카메라·기록 어댑터입니다.
        sessions_root: 기록 연결의 저장 루트이며 실제 I/O는 저장소가 담당합니다.
        replay_source: 검증용 기존 영상 경로이며 None이면 선택 장치입니다.
        repository: bootstrap에서 주입하는 통합 요약 저장 포트입니다.
            구현 선택은 bootstrap 또는 호환 진입점이 담당합니다.
        score_factory: 단일 카메라를 사용하는 점수 실행 포트의 생성자입니다.
    """

    fixed_mode = "combined"

    def __init__(
        self,
        engine: CorrectionEnginePort,
        sessions_root: Path,
        repository: SessionRepository,
        score_factory: Callable[["IntegratedSession"], ScoreCoordinatorPort],
        replay_source: str | None = None,
    ) -> None:
        self.engine = engine
        self.sessions_root = Path(sessions_root)
        self.repository = repository
        self.replay_source = replay_source
        self.lock = threading.RLock()
        self.control = threading.RLock()
        self.lifecycle = SessionLifecycle()
        self.session_id = None
        self.started = None
        self.events = []
        self.last_summary = None

        self.score = score_factory(self)

    @property
    def enabled(self) -> bool:
        """어댑터가 참조할 실행 여부를 단일 상태 소유자에서 읽습니다."""
        return self.lifecycle.enabled

    @property
    def paused(self) -> bool:
        """일시정지 여부를 별도 가변 플래그 없이 반환합니다."""
        return self.lifecycle.phase is SessionPhase.PAUSED

    @property
    def jpeg(self):
        return self.engine.image() if self.session_id else None

    def snapshot(self) -> dict[str, object]:
        """표시용 상태를 복사합니다. 모델 판정과 완료 횟수를 재계산하지 않습니다.

        Returns:
            공통 세션 ID·실행 상태·각 분석 결과·저장 요약입니다. 일시정지
            중에는 완료 점수를 유지하고 진행 중 잠정 점수만 숨깁니다.
        """
        state = self.score.snapshot()
        correction = self.engine.snapshot()
        if self.paused:
            state["score"] = dict(
                self.score.retained_score(),
                live_score_100=None,
                paused=True,
            )
            correction = dict(correction, result=None, status="paused")
        return dict(
            state,
            session_id=self.session_id,
            mode="combined",
            running=state.get("running", False),
            paused=self.paused,
            status="paused" if self.paused else state.get("status"),
            correction=correction,
            integrated_summary=self.last_summary,
            recording_scope="동일 카메라의 교정·점수 입력을 각각 저장하며 공통 세션으로 연결합니다.",
        )

    def start(self, options: Mapping[str, object]) -> None:
        """설정을 검증한 뒤 이전 실행을 종료하고 새 세션을 준비합니다.

        Args:
            options: 운동 코드·입력 종류·웹캠 번호 또는 ESP32 주소입니다.
                촬영 방향은 현재 정면만 지원합니다.

        Raises:
            ValueError: 모드·운동·입력 설정이 잘못된 경우.
            OSError: 원본 어댑터의 시작·이전 기록 마감이 실패한 경우.
            RuntimeError: 기존 엔진 또는 Worker 준비가 실패한 경우.
        """
        if options.get("mode", "combined") != "combined":
            raise ValueError("이 서버는 운동 교정·자세 점수 통합 전용입니다.")
        checked = validate(dict(options, mode="score"), "score")
        if checked["camera"] == "auto":
            raise ValueError("통합 모드는 webcam 또는 esp32를 명시하십시오.")
        with self.control:
            self.stop()
            self.session_id = uuid.uuid4().hex
            self.started = time.time()
            self.events = [
                {
                    "event": "start",
                    "monotonic": time.monotonic(),
                    "unix": self.started,
                }
            ]
            self.last_summary = None
            self.engine.integration_session_id = self.session_id
            camera = checked["camera"]
            source = self.replay_source or (
                checked["esp32"]
                if camera == "esp32"
                else str(checked["webcam"])
            )
            self.options = dict(checked, mode="combined")
            try:
                self.engine.start(
                    source,
                    CORRECTION_CODES.get(
                        checked["exercise"], checked["exercise"]
                    ),
                    True,
                )
                self.score.start(checked)
                self.score.bind_session(self.session_id)
                self.lifecycle.start()
            except Exception:
                LOGGER.error("통합 시작 실패 session_id=%s", self.session_id)
                self.stop()
                raise

    def pause(self) -> None:
        """관측 입력만 중단합니다. 중복 호출은 제어 이벤트를 추가하지 않습니다.

        Raises:
            ValueError: 실행하지 않은 세션을 일시정지한 경우.
        """
        with self.control:
            if not self.score.snapshot().get("running"):
                raise ValueError("실행 중인 운동이 없습니다.")
            if not self.lifecycle.pause():
                return
            self.events.append(
                {"event": "pause", "monotonic": time.monotonic()}
            )
            if self.engine.reader:
                self.engine.reader.paused = True
            self.score.reset_motion()

    def resume(self) -> None:
        """완료 결과를 유지하고 단절된 진행 중 관측을 초기화한 뒤 재개합니다."""
        with self.control:
            if not self.paused:
                return
            self.score.reset_motion()
            self.engine.reset()
            if self.engine.reader:
                self.engine.reader.paused = False
            self.lifecycle.resume()
            self.events.append(
                {"event": "resume", "monotonic": time.monotonic()}
            )

    def stop(self) -> None:
        """두 실행을 마감하고 저장 성공 뒤에만 요약을 게시합니다.

        Raises:
            OSError: 주입한 저장소가 기록을 저장하지 못한 경우. 결과를
                저장 완료로 표시하지 않으며 같은 세션에서 재시도할 수 있습니다.
            ValueError: 저장 요약이 저장소 계약을 만족하지 않는 경우.
        """
        with self.control:
            self.score.stop()
            self.engine.stop()
            self.lifecycle.stop()
            if not self.session_id or self.last_summary:
                return
            state = self.snapshot()
            correction = state["correction"]
            score = state.get("score", {})
            summary = dict(
                schema="integrated-session-v1",
                control_events=self.events
                + [{"event": "stop", "monotonic": time.monotonic()}],
                session_id=self.session_id,
                started_unix=self.started,
                ended_unix=time.time(),
                options=self.options,
                score=score,
                correction=correction,
                correction_directory=correction.get("session_path"),
                score_directory=score.get("recording", {}).get("directory"),
                temporal_contract="교정은 연속 시간창 결과이며 특정 반복의 확정 오류 라벨로 환산하지 않습니다.",
                recording_errors=[
                    e
                    for e in (
                        correction.get("recording_error"),
                        score.get("recording", {}).get("error"),
                    )
                    if e
                ],
            )
            score_directory = summary.get("score_directory")
            summary["score_epoch_directories"] = self.repository.score_epochs(
                score_directory
            )
            self.repository.save(self.session_id, summary)
            self.last_summary = summary

    def list_sessions(self) -> list[dict[str, object]]:
        """저장 구현을 모르는 HTTP 계층에 세션 목록을 전달합니다."""
        return self.repository.list_sessions()
