"""합성 데이터와 루프백 SSH로 읽기·경로 격리·인증·쓰기 거부를 검증한다.

실제 T9·유선 IP·방화벽·사용자 키는 변경하지 않는다.
실행: python -m unittest tools.t9_share.test_share -v
"""

import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

import paramiko

from tools.t9_share import __main__ as cli
from tools.t9_share import filesystem, server


class RootReaderTest(unittest.TestCase):
    """파일 핸들 격리와 경로 탈출 실패를 임시 폴더로 확인한다."""

    def test_symlinks_traversal_and_special_files_are_denied(self) -> None:
        """상위 이동·파일 링크·디렉터리 링크·FIFO를 거부한다."""
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "data"
            root.mkdir()
            (base / "secret").write_text("secret")
            (root / "link").symlink_to(base / "secret")
            (root / "outside").symlink_to(base, target_is_directory=True)
            os.mkfifo(root / "fifo")
            with filesystem.RootReader(root) as reader:
                for path in (
                    "../secret",
                    "/../secret",
                    "link",
                    "outside/secret",
                    "fifo",
                ):
                    with self.subTest(path=path), self.assertRaises(OSError):
                        reader.open_fd(path)


class SftpIntegrationTest(unittest.TestCase):
    """실제 암호화된 SSH 연결을 사용하며 모든 자원을 종료 시 해제한다."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root_path = pathlib.Path(self.temporary.name)
        self.data = bytes(range(256)) * 8192
        (self.root_path / "운동.bin").write_bytes(self.data)
        (self.root_path / "folder").mkdir()
        (self.root_path / "hidden-link").symlink_to("/etc/passwd")
        self.root = filesystem.RootReader(self.root_path)
        self.key = paramiko.RSAKey.generate(2048)
        self.host_key = paramiko.RSAKey.generate(2048)
        self.stopping = threading.Event()
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(5)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        self.client = paramiko.SSHClient()
        self.client.get_host_keys().add(
            f"[127.0.0.1]:{self.port}", self.host_key.get_name(), self.host_key
        )

    def _accept(self) -> None:
        connection, _ = self.listener.accept()
        server.serve_connection(
            connection, self.host_key, self.key, self.root, self.stopping
        )

    def _connect(self, key=None) -> None:
        self.client.connect(
            "127.0.0.1",
            port=self.port,
            username="t9",
            pkey=key or self.key,
            allow_agent=False,
            look_for_keys=False,
            timeout=5,
            auth_timeout=5,
        )

    def tearDown(self) -> None:
        self.client.close()
        self.stopping.set()
        self.thread.join(6)
        self.listener.close()
        self.root.__exit__()
        self.temporary.cleanup()
        self.assertFalse(self.thread.is_alive())

    def test_read_seek_and_concurrent_file_handles(self) -> None:
        """UTF-8 이름과 MP4 탐색에 필요한 임의 위치 읽기를 검증한다."""
        self._connect()
        with self.client.open_sftp() as sftp:
            self.assertEqual(set(sftp.listdir("/")), {"folder", "운동.bin"})
            self.assertEqual(sftp.stat("/운동.bin").st_size, len(self.data))
            with (
                sftp.open("/운동.bin", "rb") as first,
                sftp.open("운동.bin", "rb") as second,
            ):
                first.seek(1048570)
                self.assertEqual(first.read(90000), self.data[1048570:1138570])
                self.assertEqual(second.read(12345), self.data[:12345])
                self.assertEqual(first.stat().st_size, len(self.data))

    def test_writes_and_escape_are_denied_and_original_unchanged(self) -> None:
        """수정·삭제·이름 변경·생성·권한 변경·링크·탈출을 거부한다."""
        self._connect()
        with self.client.open_sftp() as sftp:
            operations = [
                lambda: sftp.open("/운동.bin", "w"),
                lambda: sftp.open("/new", "w"),
                lambda: sftp.open("/운동.bin", "a"),
                lambda: sftp.remove("/운동.bin"),
                lambda: sftp.rename("/운동.bin", "/new"),
                lambda: sftp.mkdir("/new"),
                lambda: sftp.rmdir("/folder"),
                lambda: sftp.chmod("/운동.bin", 0o777),
                lambda: sftp.symlink("/운동.bin", "/new"),
                lambda: sftp.open("/hidden-link"),
                lambda: sftp.open("/../etc/passwd"),
            ]
            for operation in operations:
                with self.assertRaises(OSError):
                    operation()
        self.assertEqual((self.root_path / "운동.bin").read_bytes(), self.data)
        self.assertFalse((self.root_path / "new").exists())

    def test_wrong_key_is_rejected(self) -> None:
        """미등록 개인키로는 인증할 수 없다."""
        with self.assertRaises(paramiko.AuthenticationException):
            self._connect(paramiko.RSAKey.generate(2048))

    def test_shell_exec_and_forwarding_are_rejected(self) -> None:
        """SFTP 외 명령 실행·셸·TCP 포트 전달은 거부한다."""
        self._connect()
        for operation in (
            lambda: self.client.exec_command("cat /etc/passwd", timeout=5),
            lambda: self.client.get_transport().open_channel(
                "direct-tcpip", ("localhost", 22), ("localhost", 12345)
            ),
        ):
            with self.assertRaises(paramiko.SSHException):
                operation()

    def test_host_key_mismatch_is_rejected(self) -> None:
        """신뢰 파일과 다른 서버는 연결하지 않는다."""
        self.client.get_host_keys().add(
            f"[127.0.0.1]:{self.port}", "ssh-rsa", self.key
        )
        with self.assertRaises(paramiko.BadHostKeyException):
            self._connect()

    @unittest.skipUnless(shutil.which("sshfs"), "SSHFS 설치 시 실행")
    def test_real_sshfs_mount_read_and_unmount(self) -> None:
        """FUSE 폴더의 읽기·탐색·쓰기 거부·연결 해제를 실제 실행한다."""
        with tempfile.TemporaryDirectory() as local:
            local_path = pathlib.Path(local)
            identity = local_path / "identity"
            self.key.write_private_key_file(str(identity))
            identity.chmod(0o600)
            known = local_path / "known_hosts"
            known.write_text(
                f"[127.0.0.1]:{self.port} {self.host_key.get_name()} "
                f"{self.host_key.get_base64()}\n"
            )
            mountpoint = local_path / "mounted"
            args = cli.argparse.Namespace(
                host="127.0.0.1",
                port=self.port,
                identity=str(identity),
                known_hosts=str(known),
                mountpoint=str(mountpoint),
            )
            try:
                # 실제 LAN 장치 변경 없이 루프백으로 SSHFS 호환성만 검증한다.
                with mock.patch.object(
                    cli, "check_wired_route", return_value="lo"
                ):
                    cli.mount_share(args)
                self.assertTrue(os.path.ismount(mountpoint))
                with (mountpoint / "운동.bin").open("rb") as source:
                    source.seek(900000)
                    self.assertEqual(
                        source.read(300000), self.data[900000:1200000]
                    )
                with self.assertRaises(OSError):
                    (mountpoint / "new").write_text("not allowed")
            finally:
                if os.path.ismount(mountpoint):
                    subprocess.run(
                        ["fusermount3", "-u", str(mountpoint)], check=True
                    )
            self.assertFalse(os.path.ismount(mountpoint))


class KeyTest(unittest.TestCase):
    """신원 키 재생성 방지와 공개키 입력을 확인한다."""

    def test_server_identity_survives_restart(self) -> None:
        """서버 재시작 시 동일한 키를 사용하고 비공개 권한을 유지한다."""
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "key"
            first = cli.create_host_key(path)
            self.assertEqual(first, cli.create_host_key(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                cli.create_host_key(path)

    def test_public_key_roundtrip(self) -> None:
        """허용 공개키만 읽고 개인키·옵션 행을 거부한다."""
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "client.pub"
            key = paramiko.RSAKey.generate(2048)
            path.write_text(f"{key.get_name()} {key.get_base64()} test\n")
            self.assertEqual(server.load_public_key(path), key)
            path.write_text("command=bad ssh-ed25519 bad")
            with self.assertRaises(ValueError):
                server.load_public_key(path)


if __name__ == "__main__":
    unittest.main()
