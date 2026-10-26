"""통합 기록의 저장·조회 포트를 정의합니다. 파일·DB 구현은 포함하지 않습니다."""

from collections.abc import Mapping
from typing import Protocol


class CameraControlPort(Protocol):
    """원본 카메라의 일시정지 제어만 application에 노출합니다."""

    paused: bool


class CorrectionEnginePort(Protocol):
    """원본 교정 어댑터의 실행·조회 계약입니다. 모델 출력을 재해석하지 않습니다."""

    integration_session_id: str | None
    reader: CameraControlPort | None

    def start(self, source: str, exercise: str, record: bool) -> None:
        """선택 입력에서 원본 교정과 기록을 시작합니다."""
        ...

    def stop(self) -> None:
        """카메라·모델 자원을 종료하고 기록을 마감합니다."""
        ...

    def reset(self) -> None:
        """단절 전의 진행 중 관측을 제거합니다."""
        ...

    def snapshot(self) -> dict[str, object]:
        """기존 교정 상태 계약의 복사본을 반환합니다."""
        ...

    def image(self) -> bytes | None:
        """표시용 JPEG를 반환합니다. 입력이 없으면 None입니다."""
        ...


class ScoreCoordinatorPort(Protocol):
    """점수 실행 수명과 결과 조회 계약입니다. 반복 소유권은 변경하지 않습니다."""

    def start(self, options: Mapping[str, object]) -> None:
        """검증한 입력 설정으로 점수 처리를 시작합니다."""
        ...

    def stop(self) -> None:
        """실제 입력 처리를 종료하고 완료 결과를 보존합니다."""
        ...

    def snapshot(self) -> dict[str, object]:
        """현재 결과를 복사하여 반환합니다."""
        ...

    def retained_score(self) -> dict[str, object]:
        """일시정지 표시용 마지막 완료 결과를 반환합니다."""
        ...

    def bind_session(self, session_id: str) -> None:
        """입력을 전달하기 전에 공통 세션 ID를 연결합니다."""
        ...

    def reset_motion(self) -> None:
        """완료 결과를 유지하고 진행 중 동작만 초기화합니다."""
        ...


class SessionRepository(Protocol):
    """공통 세션 ID의 기록 연결을 저장·조회하는 application 포트입니다.

    manifest는 모델 출력 계약을 변경하지 않는 JSON 호환 스냅샷입니다.
    부동소수점 단위와 관절 좌표는 각 분석 기록의 metadata에 유지합니다.
    """

    def save(self, session_id: str, manifest: Mapping[str, object]) -> None:
        """요약을 저장하며 실패는 OSError·ValueError로 호출자에게 전달합니다."""
        ...

    def list_sessions(self) -> list[dict[str, object]]:
        """저장된 세션 요약을 반환합니다. 기록이 없으면 빈 목록입니다."""
        ...

    def score_epochs(self, directory: str | None) -> list[str]:
        """점수 기록의 연결 구간을 반환합니다. 경로 없음은 빈 목록입니다."""
        ...
