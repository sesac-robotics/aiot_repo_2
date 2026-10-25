"""공개키와 상대 IP로 제한한 읽기 전용 SSH/SFTP 서버.

셸·명령 실행·포트 전달·비밀번호 인증은 지원하지 않는다.
기존 SSH 서비스와 별개 포트를 사용하며 시스템 설정을 변경하지 않는다.
"""

import base64
import hashlib
import logging
import pathlib
import socket
import threading
import time

import paramiko

from tools.t9_share import filesystem

LOGGER = logging.getLogger(__name__)


def load_public_key(path: pathlib.Path) -> paramiko.PKey:
    """OpenSSH 공개키 한 개를 읽는다. 개인키·여러 키·옵션 행은 거부한다.

    Args:
        path: ssh-keygen이 생성한 .pub 파일.

    Returns:
        인증 비교용 공개키 객체.

    Raises:
        ValueError: 공개키 형식이 잘못된 경우.
        OSError: 파일을 읽을 수 없는 경우.
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 1:
        raise ValueError("공개키 파일 한 개만 지정하세요")
    fields = lines[0].split()
    if len(fields) < 2 or fields[0] not in ("ssh-ed25519", "ssh-rsa"):
        raise ValueError("Ed25519 또는 RSA .pub 파일이 필요합니다")
    return paramiko.PKey.from_type_string(
        fields[0], base64.b64decode(fields[1], validate=True)
    )


def fingerprint(key: paramiko.PKey) -> str:
    """공개키의 SHA256 지문을 표시용 문자열로 반환한다."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


class KeyAuth(paramiko.ServerInterface):
    """t9 가상 계정과 지정 공개키만 인증한다. OS 계정과 무관하다."""

    def __init__(self, key: paramiko.PKey) -> None:
        self.key = key
        self.channels = 0

    def get_allowed_auths(self, username: str) -> str:
        """클라이언트에 공개키 인증만 제공한다."""
        return "publickey"

    def check_auth_publickey(self, username: str, key: paramiko.PKey) -> int:
        """계정·키가 일치하면 인증 성공, 그 외에는 실패 코드를 반환한다."""
        if username == "t9" and key == self.key:
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind: str, chanid: int) -> int:
        """SFTP에 필요한 session 채널만 열고 나머지는 거부한다."""
        if kind == "session" and self.channels < 4:
            self.channels += 1
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED


def serve_connection(
    connection: socket.socket,
    host_key: paramiko.PKey,
    client_key: paramiko.PKey,
    root: filesystem.RootReader,
    stopping: threading.Event,
) -> None:
    """SSH 연결 하나를 처리하고 종료 시 소켓·파일 핸들을 정리한다.

    Args:
        connection: 서버가 수락한 TCP 소켓. 이 함수가 소유한다.
        host_key: 공유 서버 신원 키.
        client_key: 허용할 클라이언트 공개키 한 개.
        root: 호출자가 수명을 보장하는 공유 루트.
        stopping: 서버 종료 신호.

    Returns:
        None. 인증·통신 실패는 로그 후 연결을 닫는다.
    """
    transport = paramiko.Transport(connection)
    try:
        transport.banner_timeout = 15
        transport.auth_timeout = 30
        transport.add_server_key(host_key)
        transport.set_subsystem_handler(
            "sftp", paramiko.SFTPServer, filesystem.ReadOnlySftp, root=root
        )
        transport.start_server(server=KeyAuth(client_key))
        auth_deadline = time.monotonic() + 30
        while transport.is_active() and not stopping.wait(0.5):
            if (
                not transport.is_authenticated()
                and time.monotonic() > auth_deadline
            ):
                break
    except (OSError, EOFError, paramiko.SSHException) as error:
        LOGGER.warning("SSH 연결 종료: %s", type(error).__name__)
    finally:
        transport.close()
        connection.close()


def serve(
    root_path: pathlib.Path,
    bind_ip: str,
    peer_ip: str,
    port: int,
    host_key: paramiko.PKey,
    client_key: paramiko.PKey,
) -> None:
    """지정 IP·TCP 포트에서 최대 네 연결을 처리한다. Ctrl+C로 종료한다.

    서버 설정·디스크 권한·방화벽은 변경하지 않는다. peer_ip 외 접속은
    SSH 인증 전에 닫는다. root_path와 키의 유효성은 호출자가 검사한다.

    Raises:
        OSError: 루트·포트·바인딩 주소에 접근할 수 없는 경우.
    """
    stopping = threading.Event()
    workers = []
    with filesystem.RootReader(root_path) as root:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((bind_ip, port))
            listener.listen(4)
            listener.settimeout(1)
            LOGGER.info(
                "공유 시작 %s:%d → %s (읽기 전용)", bind_ip, port, root_path
            )
            try:
                while True:
                    try:
                        connection, address = listener.accept()
                    except socket.timeout:
                        continue
                    workers = [
                        worker for worker in workers if worker.is_alive()
                    ]
                    if address[0] != peer_ip or len(workers) >= 4:
                        connection.close()
                        continue
                    worker = threading.Thread(
                        target=serve_connection,
                        args=(connection, host_key, client_key, root, stopping),
                        daemon=True,
                    )
                    worker.start()
                    workers.append(worker)
            except KeyboardInterrupt:
                LOGGER.info("공유 종료 중")
            finally:
                stopping.set()
                for worker in workers:
                    worker.join(timeout=35)
