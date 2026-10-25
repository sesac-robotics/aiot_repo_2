"""공유 루트 내부의 일반 파일만 읽는 SFTP 어댑터.

Linux dir_fd와 O_NOFOLLOW로 각 경로 요소를 연다. 상위 경로 이동,
심볼릭 링크, 특수 파일과 모든 쓰기는 허용하지 않는다. 루트 디스크의
소유자는 공유 중 파일 이동·수정을 피해야 한다. OS 수준 샌드박스는 아니다.
"""

import errno
import os
import pathlib
import stat

import paramiko


class RootReader:
    """디렉터리 FD 하나를 소유하며 컨텍스트 종료 시 닫는다.

    Args:
        root: 공유할 실제 디렉터리. 최종 요소가 링크이면 실패한다.

    Raises:
        OSError: 디렉터리가 없거나 읽을 수 없는 경우.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def __enter__(self) -> "RootReader":
        return self

    def __exit__(self, *_: object) -> None:
        os.close(self.fd)

    def open_fd(self, path: str) -> int:
        """가상 절대·상대 경로를 읽기 FD로 연다. 닫기는 호출자 책임이다.

        Args:
            path: 공유 루트 기준 UTF-8 경로. '..' 요소를 허용하지 않는다.

        Returns:
            일반 파일 또는 디렉터리의 읽기 전용 FD.

        Raises:
            OSError: 경로 탈출·링크·특수 파일·접근 실패.
        """
        parts = pathlib.PurePosixPath(path).parts
        if ".." in parts or "\x00" in path:
            raise PermissionError(errno.EACCES, "허용되지 않는 경로")
        parts = [part for part in parts if part not in ("/", "//", ".")]
        fd = os.dup(self.fd)
        try:
            for index, part in enumerate(parts):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(parts) - 1:
                    flags |= os.O_DIRECTORY
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            mode = os.fstat(fd).st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise PermissionError(errno.EACCES, "일반 파일만 지원합니다")
            return fd
        except OSError:
            os.close(fd)
            raise


class ReadHandle(paramiko.SFTPHandle):
    """열린 일반 파일 FD를 소유하며 SFTP CLOSE에서 해제한다."""

    def __init__(self, fd: int) -> None:
        super().__init__(os.O_RDONLY)
        self.fd = fd

    def read(self, offset: int, length: int) -> bytes | int:
        """바이트 단위 offset부터 최대 length를 읽고 실패 시 SFTP 코드를 반환한다."""
        try:
            return os.pread(self.fd, min(length, 1024 * 1024), offset)
        except OSError as error:
            return paramiko.SFTPServer.convert_errno(error.errno)

    def stat(self) -> paramiko.SFTPAttributes | int:
        """현재 파일 속성 또는 실패 코드를 반환한다. 크기는 바이트다."""
        try:
            return paramiko.SFTPAttributes.from_stat(os.fstat(self.fd))
        except OSError as error:
            return paramiko.SFTPServer.convert_errno(error.errno)

    def close(self) -> int:
        """소유 FD를 닫고 성공 코드를 반환한다. 중복 호출은 안전하다."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        return paramiko.SFTP_OK


class ReadOnlySftp(paramiko.SFTPServerInterface):
    """조회만 구현하고 나머지 변경 요청은 기본 구현에서 거부한다.

    root는 서버가 소유한 RootReader로 모든 연결이 공유한다.
    각 요청이 별도 FD를 사용하므로 파일 위치를 서로 변경하지 않는다.
    """

    def __init__(self, server: paramiko.ServerInterface, *, root: RootReader):
        super().__init__(server)
        self.root = root

    def list_folder(self, path: str) -> list | int:
        """디렉터리의 이름·속성을 반환한다. 링크는 목록에서도 제외한다."""
        try:
            fd = self.root.open_fd(path)
            try:
                entries = []
                for name in os.listdir(fd):
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode):
                        entries.append(
                            paramiko.SFTPAttributes.from_stat(info, name)
                        )
                return entries
            finally:
                os.close(fd)
        except OSError as error:
            return paramiko.SFTPServer.convert_errno(error.errno)

    def stat(self, path: str) -> paramiko.SFTPAttributes | int:
        """루트 내부 경로의 속성 또는 실패 코드를 반환한다."""
        try:
            fd = self.root.open_fd(path)
            try:
                return paramiko.SFTPAttributes.from_stat(os.fstat(fd))
            finally:
                os.close(fd)
        except OSError as error:
            return paramiko.SFTPServer.convert_errno(error.errno)

    lstat = stat

    def open(self, path: str, flags: int, attr: paramiko.SFTPAttributes):
        """읽기 요청에만 핸들을 반환한다. 생성·추가·수정은 실패 코드다."""
        denied = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & denied:
            return paramiko.SFTP_PERMISSION_DENIED
        try:
            fd = self.root.open_fd(path)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                return paramiko.SFTP_PERMISSION_DENIED
            return ReadHandle(fd)
        except OSError as error:
            return paramiko.SFTPServer.convert_errno(error.errno)
