"""MJPEG에서 JPEG 시작·종료 표식을 추출하는 순수 바이트 어댑터입니다.

HTTP multipart 헤더 해석이나 이미지 디코딩은 수행하지 않습니다. 네트워크가
임의 위치에서 청크를 나누는 상황을 처리하고 미완성 프레임의 메모리를 제한합니다.
"""


class JPEGFrameParser:
    """청크 경계를 넘는 프레임을 조립합니다. 인스턴스 하나가 연결 하나를 소유합니다."""

    def __init__(self, max_frame_bytes: int = 8 * 1024 * 1024) -> None:
        if max_frame_bytes < 4:
            raise ValueError(
                "max_frame_bytes는 JPEG 표식 4바이트 이상이어야 합니다."
            )
        self._limit = max_frame_bytes
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        """다음 청크를 기다리는 바이트 수입니다. 제한 크기 이하로 유지됩니다."""
        return len(self._buffer)

    def feed(self, chunk: bytes) -> list[bytes]:
        """완성된 프레임을 도착 순서대로 반환하고 잘못된 입력을 버립니다.

        Args:
            chunk: 한 번 수신한 바이트입니다. 빈 청크도 허용합니다.

        Returns:
            크기 제한 이내의 JPEG 표식으로 둘러싸인 바이트 목록입니다.
            이것만으로 유효한 영상임이 보장되지는 않으며 디코더 검사가 필요합니다.
        """
        self._buffer.extend(chunk)
        frames = []
        while True:
            start = self._buffer.find(b"\xff\xd8")
            if start < 0:
                # Keep one byte so a split FF/D8 marker survives.
                del self._buffer[:-1]
                break
            if start:
                del self._buffer[:start]
            end = self._buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self._buffer) > self._limit:
                    # Resynchronize to a newer start marker if one is available.
                    next_start = self._buffer.rfind(b"\xff\xd8", 2)
                    if (
                        next_start >= 0
                        and len(self._buffer) - next_start <= self._limit
                    ):
                        del self._buffer[:next_start]
                    else:
                        del self._buffer[:-1]
                break
            frame_end = end + 2
            if frame_end <= self._limit:
                frames.append(bytes(self._buffer[:frame_end]))
            del self._buffer[:frame_end]
        return frames
