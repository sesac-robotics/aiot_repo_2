"""비상업 연구용 REHAB24 ZIP에서 스쿼트 영상만 범위 요청으로 확보한다.

이 자료의 테이블 푸시업은 대상 일반 푸시업에 섞지 않는다. 영상은 나중에
직접 관찰하여 독립 등급을 만들며 제공자 correctness를 점수로 바꾸지 않는다.
"""

import argparse
import io
import pathlib
import urllib.request
import zipfile

import audit
import three_protocol
from resources import check_resources


class RangeReader(io.RawIOBase):
    """원격 ZIP을 HTTP 206 응답으로만 읽는 위치 보유 읽기 전용 객체다."""

    def __init__(self, url: str, length: int):
        """URL·전체 바이트 길이를 받고 위치를 0으로 설정한다.

        Args:
            url: 공식 ZIP 다운로드 URL.
            length: 공식 API가 제공한 바이트 길이.
        """
        self.url, self.length, self.position = url, length, 0
        self.transferred = 0

    def seekable(self) -> bool:
        """ZIP 디렉터리 탐색이 가능하므로 True를 반환한다."""
        return True

    def tell(self) -> int:
        """현재 파일 위치를 바이트 단위로 반환한다."""
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        """시작·현재·끝 기준 바이트 위치를 변경한다.

        Args:
            offset: 부호 있는 바이트 오프셋.
            whence: 0 시작, 1 현재, 2 파일 끝 기준.

        Returns:
            int: 변경 후 0 이상의 바이트 위치.

        Raises:
            ValueError: 알 수 없는 기준 또는 파일 밖 위치.
        """
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        new = (0, self.position, self.length)[whence] + offset
        if not 0 <= new <= self.length:
            raise ValueError("Seek outside remote ZIP")
        self.position = new
        return new

    def read(self, size: int = -1) -> bytes:
        """현재 범위를 다운로드하고 응답 길이·Content-Range를 검증한다.

        Args:
            size: 바이트 수, 음수면 나머지 전체. 한 번에 최대 160MiB.

        Returns:
            bytes: 요청한 범위. EOF는 빈 바이트다.

        Raises:
            ValueError: 과도한 요청·206 미지원·잘못된 범위 응답.
            OSError: 네트워크 읽기 실패.
        """
        size = min(
            self.length - self.position, size if size >= 0 else self.length
        )
        if size > 160 * 1024 * 1024:
            raise ValueError("Bounded remote read exceeded")
        if not size:
            return b""
        a, b = self.position, self.position + size - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={a}-{b}"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range")
                != f"bytes {a}-{b}/{self.length}"
            ):
                raise ValueError("Server must honor the exact HTTP range")
            content = response.read(size + 1)
        if len(content) != size:
            raise ValueError("Truncated or oversized range")
        self.position += size
        self.transferred += size
        return content


def run(acquire: bool = False) -> dict:
    """공식 ZIP 목록을 검사하고 선택 시 스쿼트에 해당하는 파일만 저장한다.

    Args:
        acquire: False는 인덱스 조회만, True는 Ex6 영상만 CRC 검사 후 저장.

    Returns:
        dict: ZIP 목록·실제 전송 바이트·다운로드 해시와 직접 검토 미실행 상태.

    Raises:
        OSError: 네트워크·저장 실패.
        ValueError: 범위 응답·크기·경로 오류.
        zipfile.BadZipFile: ZIP 디렉터리 또는 파일 CRC 불일치.
        RuntimeError: 전원·온도·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    record = audit.read_json(root / "rehab24_record.json")
    item = next(f for f in record["files"] if f["key"] == "videos.zip")
    reader = RangeReader(item["links"]["self"], item["size"])
    report = dict(
        dataset_url="https://zenodo.org/records/13305826",
        terms="Academic or nonprofit noncommercial research only; cite Cernek et al. SISAP 2024",
        index=[],
        downloaded=[],
        archive_bytes=item["size"],
        direct_reviews=0,
        score_labels_created=0,
    )
    with zipfile.ZipFile(reader) as archive:
        for info in archive.infolist():
            report["index"].append(
                dict(
                    name=info.filename,
                    bytes=info.file_size,
                    compressed_bytes=info.compress_size,
                )
            )
            parts = pathlib.PurePosixPath(info.filename).parts
            if (
                not acquire
                or not any(p.lower() in ("ex6", "6") for p in parts)
                or not info.filename.lower().endswith(".mp4")
            ):
                continue
            if ".." in parts or info.file_size > 160 * 1024 * 1024:
                raise ValueError("Unsafe selected archive member")
            check_resources()
            target = audit.BASE / "data/rehab24-squat" / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                content = archive.read(info)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(content)
                temporary.replace(target)
            report["downloaded"].append(
                dict(
                    path=str(target),
                    sha256=audit.digest(target),
                    bytes=target.stat().st_size,
                )
            )
            audit.save_json(root / "rehab24_acquisition.json", report)
    report["http_transferred_bytes_this_run"] = reader.transferred
    audit.save_json(
        root
        / ("rehab24_acquisition.json" if acquire else "rehab24_zip_index.json"),
        report,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    print(run(parser.parse_args().acquire))
