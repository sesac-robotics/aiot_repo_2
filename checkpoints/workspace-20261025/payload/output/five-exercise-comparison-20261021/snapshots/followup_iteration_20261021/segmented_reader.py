"""포장 TAR의 분할 페이로드를 복사 없이 하나의 읽기 스트림으로 제공한다.

각 구간은 논리 시작·원본 물리 시작·길이를 바이트로 지정한다. 별도 28GB
합친 파일을 만들지 않아 디스크 중복을 줄이며 원본 스트림은 수정하지 않는다.
"""

import bisect
import io


class SegmentedReader(io.RawIOBase):
    """연속된 논리 구간을 원본 파일의 물리 구간에 연결하는 읽기 전용 스트림.

    원본 스트림의 수명은 호출자가 관리한다. 동시 읽기는 지원하지 않는다.
    """

    def __init__(self, source, segments: list[tuple[int, int, int]]):
        """논리 구간의 누락·중복을 검사하고 읽기 위치를 초기화한다.

        Args:
            source: seek·read를 제공하는 원본 바이너리 스트림.
            segments: 논리 시작·물리 시작·길이의 바이트 튜플 목록.

        Raises:
            ValueError: 구간이 없거나 논리 주소가 연속하지 않은 경우.
        """
        super().__init__()
        self._source = source
        self._segments = sorted(segments)
        self._starts = []
        expected = 0
        for logical, physical, length in self._segments:
            if logical != expected or physical < 0 or length <= 0:
                raise ValueError("Non-contiguous payload segments")
            self._starts.append(logical)
            expected += length
        if not expected:
            raise ValueError("Empty segmented payload")
        self._length = expected
        self._position = 0

    def readable(self) -> bool:
        """읽기 지원 여부 True를 반환한다. 단위와 실패 조건은 없다."""
        return True

    def seekable(self) -> bool:
        """위치 이동 지원 여부 True를 반환한다. 단위와 실패 조건은 없다."""
        return True

    def tell(self) -> int:
        """현재 논리 위치를 바이트로 반환한다. 실패 조건은 없다."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """논리 위치를 이동하고 새 바이트 위치를 반환한다.

        Args:
            offset: 기준점에서 이동할 바이트 수.
            whence: 시작·현재·끝 기준을 나타내는 SEEK 상수.

        Returns:
            int: 이동 후 논리 바이트 위치. EOF 뒤 이동도 허용한다.

        Raises:
            ValueError: 음수 위치 또는 지원하지 않는 기준점.
        """
        bases = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self._position,
            io.SEEK_END: self._length,
        }
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("Invalid seek")
        self._position = bases[whence] + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        """분할 경계를 넘겨 읽고 읽은 바이트를 반환한다.

        Args:
            size: 최대 바이트 수. 음수이면 EOF까지 읽는다.

        Returns:
            bytes: 요청한 범위의 내용. EOF에서는 빈 바이트다.

        Raises:
            OSError: 원본 구간이 예상보다 짧거나 원본 읽기가 실패한 경우.
            ValueError: 한 번에 64MiB를 넘는 메모리 읽기를 요청한 경우.
        """
        remaining = max(0, self._length - self._position)
        wanted = remaining if size < 0 else min(size, remaining)
        if wanted > 64 * 1024**2:
            raise ValueError("Read in bounded chunks")
        chunks = []
        while wanted:
            index = bisect.bisect_right(self._starts, self._position) - 1
            logical, physical, length = self._segments[index]
            within = self._position - logical
            count = min(wanted, length - within)
            self._source.seek(physical + within)
            chunk = self._source.read(count)
            if len(chunk) != count:
                raise OSError("Truncated physical segment")
            chunks.append(chunk)
            self._position += count
            wanted -= count
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        """버퍼에 읽은 바이트를 채우고 채운 길이를 반환한다.

        Args:
            buffer: 쓰기 가능한 바이트 버퍼.

        Returns:
            int: 읽은 바이트 수. EOF이면 0이다.

        Raises:
            OSError: 원본 구간 읽기 실패.
            ValueError: 단일 버퍼가 64MiB를 초과하는 경우.
        """
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)
