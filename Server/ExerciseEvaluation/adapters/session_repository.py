"""통합 세션 Repository 포트를 기존 JSON 파일 형식에 연결합니다."""

import json
import re
from collections.abc import Mapping
from pathlib import Path

from Server.ExerciseEvaluation.score_recording import write_json


class JsonSessionRepository:
    """공통 세션 요약 파일을 소유하며 기존 관절·점수 기록은 이동하지 않습니다."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, session_id: str, manifest: Mapping[str, object]) -> None:
        """원자적 교체로 기존 summary.json 계약을 유지합니다.

        Args:
            session_id: 시작 시 생성한 UUID hex 32자입니다.
            manifest: 두 분석 기록과 제어 이벤트를 연결한 요약입니다.

        Raises:
            ValueError: 세션 ID 또는 요약의 ID가 잘못된 경우.
            OSError: 디렉터리 생성·저장·교체에 실패한 경우.
        """
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("잘못된 통합 세션 ID입니다.")
        if manifest.get("session_id") != session_id:
            raise ValueError("통합 요약과 저장 경로의 세션 ID가 다릅니다.")
        directory = self.root / session_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_json(directory / "summary.json", dict(manifest))

    def list_sessions(self) -> list[dict[str, object]]:
        """요약 목록을 읽고 손상된 기록을 조용히 누락시키지 않습니다.

        Raises:
            OSError: 파일을 읽을 수 없는 경우.
            ValueError: 저장 내용이 올바른 JSON 객체가 아닌 경우.
        """
        records = []
        for path in sorted(self.root.glob("*/summary.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("통합 요약은 JSON 객체여야 합니다.")
            records.append(value)
        return records

    def score_epochs(self, directory: str | None) -> list[str]:
        """기존 점수 metadata가 있는 구간만 정렬하여 반환합니다."""
        if directory is None:
            return []
        return [
            str(path.parent)
            for path in sorted(
                Path(directory).parent.glob("epoch-*/metadata.json")
            )
        ]
