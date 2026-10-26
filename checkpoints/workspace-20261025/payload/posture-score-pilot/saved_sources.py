"""유튜브 입력 기록을 로컬 JSON에 보관한다. 영상 파일은 삭제하지 않는다."""

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from youtube_input import validate_request


class SourceLibrary:
    """단일 서버 프로세스의 영상 기록과 파일 쓰기 잠금을 소유한다.

    초기 파일이 없을 때만 기존 두 예제를 등록한다. 삭제한 예제는
    재시작 시 복원하지 않는다. 스레드 사이의 쓰기는 잠금으로 직렬화한다.
    """

    def __init__(self, root: Path) -> None:
        """저장 위치와 기본 예제를 초기화한다.

        Args:
            root: 실습 프로그램의 절대 폴더 경로.

        Raises:
            OSError: 파일을 읽거나 쓸 수 없는 경우.
            ValueError: 기존 JSON 기록이 손상된 경우. 덮어쓰지 않는다.
        """
        self.path = root / "saved-sources.json"
        self.lock = threading.Lock()
        if not self.path.exists():
            entries = []
            demos = root / "youtube-demo-jobs.json"
            if demos.exists():
                for exercise, ident in json.loads(demos.read_text()).items():
                    result = root / "jobs" / ident / "result.json"
                    if not result.is_file():
                        continue
                    source = json.loads(result.read_text()).get("source")
                    if not source:
                        continue
                    request = validate_request(
                        {
                            "url": source["url"],
                            "exercise": exercise,
                            "start_seconds": source["original_start_seconds"],
                            "duration_seconds": (
                                source["original_end_seconds"]
                                - source["original_start_seconds"]
                            ),
                        }
                    )
                    entries.append(
                        {
                            **request,
                            "id": uuid.uuid4().hex,
                            "title": source["title"],
                            "last_job_id": ident,
                            "updated_at": self._now(),
                        }
                    )
            self._write(entries)
        self.list()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> list[dict]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("저장된 유튜브 목록 형식이 올바르지 않습니다.")
        return data

    def _write(self, entries: list[dict]) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list(self) -> list[dict]:
        """최근 사용 순으로 기록을 반환한다. 실패 시 파일 예외를 전달한다."""
        with self.lock:
            return sorted(
                self._read(), key=lambda item: item["updated_at"], reverse=True
            )

    def remember(self, request: dict) -> str:
        """같은 운동·주소·구간은 갱신하고 새 입력은 자동 저장한다.

        Args:
            request: 검증된 주소·운동·시작 시간·분석 길이(초) 사전.

        Returns:
            생성하거나 갱신한 기록의 UUID 문자열.

        Raises:
            OSError: 기록을 저장할 수 없는 경우.
            ValueError: JSON 기록이 손상된 경우.
        """
        with self.lock:
            entries = self._read()
            found = next(
                (
                    entry
                    for entry in entries
                    if all(
                        entry.get(key) == value
                        for key, value in request.items()
                    )
                ),
                None,
            )
            if found is None:
                found = {
                    **request,
                    "id": uuid.uuid4().hex,
                    "title": request["url"],
                    "last_job_id": None,
                }
                entries.append(found)
            found["updated_at"] = self._now()
            self._write(entries)
            return found["id"]

    def complete(self, ident: str, title: str, job_id: str) -> None:
        """완료된 분석을 기록에 연결한다. 이미 삭제된 기록은 복원하지 않는다.

        Args:
            ident: 유튜브 기록 UUID.
            title: 원본 영상 제목.
            job_id: 정상적으로 생성된 분석 작업 UUID.

        Returns:
            None. 저장 실패 시 OSError 또는 ValueError를 전달한다.
        """
        with self.lock:
            entries = self._read()
            for entry in entries:
                if entry["id"] == ident:
                    entry.update(title=title, last_job_id=job_id)
                    self._write(entries)
                    return

    def delete(self, ident: str) -> bool:
        """기록만 제거하고 원본·분석 영상은 유지한다.

        Args:
            ident: 제거할 유튜브 기록 UUID.

        Returns:
            삭제했으면 True, 이미 없는 기록이면 False.
            저장 실패 시 OSError 또는 ValueError를 전달한다.
        """
        with self.lock:
            entries = self._read()
            remaining = [entry for entry in entries if entry["id"] != ident]
            if len(remaining) == len(entries):
                return False
            self._write(remaining)
            return True
