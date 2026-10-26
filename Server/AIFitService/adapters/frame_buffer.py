"""표시용 최신 프레임 한 개를 보관하는 비동기 공유 객체입니다."""

import asyncio
from datetime import datetime, timezone


class LatestFrameBuffer:
    """화면용 프레임을 하나만 보관합니다. 운동 완료 이벤트나 DB 기록에는 사용하지 않습니다.

    느린 구독자는 중간 영상을 건너뛰고 최신 영상을 받습니다. 시퀀스는 새 영상
    도착 여부를 판단하는 번호이며 운동 횟수가 아닙니다. 앱 수명마다 새로 생성합니다.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._frame: bytes | None = None
        self._sequence = 0
        self._last_frame_at: str | None = None

    async def publish(self, frame: bytes) -> int:
        """최신 JPEG를 교체하고 대기 중인 모든 구독자를 깨웁니다."""
        async with self._condition:
            self._frame = frame
            self._sequence += 1
            self._last_frame_at = datetime.now(timezone.utc).isoformat()
            self._condition.notify_all()
            return self._sequence

    async def next_frame(self, after_sequence: int) -> tuple[int, bytes]:
        """지정한 번호보다 새 영상이 도착할 때까지 취소 가능한 대기를 수행합니다."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._sequence > after_sequence
            )
            assert self._frame is not None
            return self._sequence, self._frame

    async def status(self) -> dict:
        """인증 정보와 영상 본문 없이 수신 상태를 반환합니다."""
        async with self._condition:
            return {
                "has_frame": self._frame is not None,
                "sequence": self._sequence,
                "last_frame_at": self._last_frame_at,
            }
