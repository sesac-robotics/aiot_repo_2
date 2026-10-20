"""유선 T9 공유·연결 명령 진입점. Python 3.10 이상과 Linux를 사용한다.

실행: python -m tools.t9_share --help (저장소 루트에서 실행).
serve는 Paramiko, mount는 SSHFS가 필요하다. 키·신뢰 파일은 홈에 두며
Git에 넣지 않는다. import만으로 네트워크·키·마운트를 만들지 않는다.
"""

import argparse
import ipaddress
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys

import paramiko

from tools.t9_share import server

LOGGER = logging.getLogger(__name__)
STATE_DIR = pathlib.Path.home() / ".local/share/aiot-t9-share"


def run(arguments: list[str]) -> str:
    """셸 없이 외부 도구를 실행한다. 인증 입력은 사용자의 터미널을 쓴다.

    Args:
        arguments: 실행 파일과 분리된 인자. 비밀 문자열을 전달하지 않는다.

    Returns:
        조회가 필요한 명령의 표준 출력은 이 함수 대신 직접 조회한다.
        여기서는 빈 문자열을 반환한다.

    Raises:
        OSError: 실행 도구가 없는 경우.
        subprocess.CalledProcessError: 명령 실패. 실패를 성공으로 숨기지 않는다.
    """
    subprocess.run(arguments, check=True)
    return ""


def check_wired_route(local_ip: str | None, peer_ip: str) -> str:
    """상대 IPv4로 가는 경로가 유선이고 지정 로컬 IP를 쓰는지 검사한다.

    Returns:
        실제 유선 장치 이름. 네트워크 설정은 변경하지 않는다.

    Raises:
        ValueError: Wi-Fi·루프백 경로, 주소 불일치 또는 미설정 주소.
        OSError: ip 도구·장치 정보를 읽을 수 없는 경우.
    """
    peer = ipaddress.IPv4Address(peer_ip)
    if peer.is_loopback or peer.is_multicast or peer.is_unspecified:
        raise ValueError("상대 노트북의 유선 IPv4를 입력하세요")
    result = subprocess.run(
        ["ip", "-j", "route", "get", str(peer)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    route = json.loads(result.stdout)[0]
    device = route.get("dev", "")
    device_path = pathlib.Path("/sys/class/net") / device
    if (
        not device
        or device == "lo"
        or (device_path / "wireless").exists()
        or not (device_path / "device").exists()
        or (device_path / "type").read_text().strip() != "1"
    ):
        raise ValueError(
            "유선 경로가 없습니다. 카메라 연결 도구로 IP를 적용하세요"
        )
    if local_ip is not None and route.get("prefsrc") != local_ip:
        raise ValueError("서버 IP가 실제 유선 IP와 다릅니다")
    return device


def create_host_key(path: pathlib.Path) -> paramiko.RSAKey:
    """서버 신원 키를 최초 한 번 생성한다. 기존 키는 유지한다.

    Args:
        path: 홈의 비공개 상태 폴더 안 파일 경로.

    Returns:
        RSA 3072비트 서버 키. 소유자 외 파일 접근 권한은 거부한다.

    Raises:
        ValueError: 기존 키 파일이 링크이거나 권한이 너무 넓은 경우.
        OSError: 생성·읽기 실패.
    """
    if not path.exists():
        key = paramiko.RSAKey.generate(3072)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            key.write_private_key(output)
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("서버 개인키는 링크가 아니며 권한 600이어야 합니다")
    return paramiko.RSAKey.from_private_key_file(str(path))


def start_share(args: argparse.Namespace) -> None:
    """입력·유선 경로를 검사하고 읽기 전용 서버를 실행한다.

    Args:
        args: serve 명령의 IP·포트·데이터 루트·클라이언트 공개키.

    Returns:
        None. Ctrl+C로 종료하면 공유 연결도 닫힌다.

    Raises:
        ValueError, OSError: 입력·키·유선 경로·디스크 접근 실패.
    """
    ipaddress.IPv4Address(args.bind)
    if args.bind == args.peer or not 1024 <= args.port <= 65535:
        raise ValueError("서로 다른 IP와 포트 1024~65535를 지정하세요")
    check_wired_route(args.bind, args.peer)
    root = pathlib.Path(args.root).expanduser().resolve(strict=True)
    if not root.is_dir() or root == pathlib.Path("/"):
        raise ValueError("공유할 데이터 폴더를 지정하세요")
    client_key = server.load_public_key(pathlib.Path(args.client_key))
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    if STATE_DIR.is_symlink() or STATE_DIR.stat().st_mode & 0o077:
        raise ValueError("상태 폴더 권한이 700이어야 합니다")
    host_key = create_host_key(STATE_DIR / "host_key")
    public_path = STATE_DIR / "t9_known_hosts"
    public_path.write_text(
        f"[{args.bind}]:{args.port} {host_key.get_name()} "
        f"{host_key.get_base64()}\n",
        encoding="utf-8",
    )
    print(f"서버 지문: {server.fingerprint(host_key)}", flush=True)
    print(f"유선혁님에게 전달할 공개 신뢰 파일: {public_path}", flush=True)
    print(
        f"허용할 사용자 키 지문: {server.fingerprint(client_key)}", flush=True
    )
    server.serve(root, args.bind, args.peer, args.port, host_key, client_key)


def mount_share(args: argparse.Namespace) -> None:
    """공개 서버 키를 검증하며 SSHFS 읽기 전용 폴더를 연결한다.

    Args:
        args: 서버 IPv4·포트·개인키·known_hosts·로컬 마운트 경로.

    Returns:
        None. 성공 후 학습 경로로 사용할 로컬 폴더를 출력한다.

    Raises:
        ValueError: 비어 있지 않은 폴더·키·잘못된 유선 경로.
        OSError, subprocess.CalledProcessError: 의존 도구·SSH 연결 실패.
    """
    check_wired_route(None, args.host)
    if not 1024 <= args.port <= 65535:
        raise ValueError("포트 범위는 1024~65535입니다")
    if shutil.which("sshfs") is None:
        raise ValueError("SSHFS가 필요합니다: sudo apt install sshfs")
    mountpoint = pathlib.Path(args.mountpoint).expanduser().resolve()
    mountpoint.mkdir(parents=True, exist_ok=True)
    if os.path.ismount(mountpoint) or any(mountpoint.iterdir()):
        raise ValueError(
            "마운트 위치는 비어 있고 연결되지 않은 폴더여야 합니다"
        )
    identity = pathlib.Path(args.identity).expanduser().resolve(strict=True)
    known = pathlib.Path(args.known_hosts).expanduser().resolve(strict=True)
    if identity.stat().st_mode & 0o077:
        raise ValueError("개인키 권한은 600이어야 합니다")
    for value in (str(identity), str(known), str(mountpoint)):
        if "," in value or "\n" in value:
            raise ValueError("SSHFS 경로에는 쉼표·줄바꿈을 사용할 수 없습니다")
    options = [
        "ro",
        "reconnect",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=3",
        "IdentitiesOnly=yes",
        "StrictHostKeyChecking=yes",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "ConnectTimeout=10",
        f"IdentityFile={identity}",
        f"UserKnownHostsFile={known}",
    ]
    run(
        [
            "sshfs",
            f"t9@{args.host}:/",
            str(mountpoint),
            "-p",
            str(args.port),
            "-o",
            ",".join(options),
        ]
    )
    print(f"학습 입력 경로: {mountpoint}")
    print("모델·캐시·체크포인트는 이 폴더 밖의 로컬 디스크에 저장하세요.")


def main() -> int:
    """명령을 실행하고 성공 0·실패 1을 반환한다. 비밀번호를 저장하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("serve", help="이영준 노트북에서 T9 읽기 공유")
    host.add_argument("--root", default="/run/media/jayden/T9/Data")
    host.add_argument("--bind", default="192.168.50.2")
    host.add_argument("--peer", default="192.168.50.1")
    host.add_argument("--port", type=int, default=8022)
    host.add_argument(
        "--client-key", required=True, help="유선혁님의 .pub 파일"
    )
    client = commands.add_parser("mount", help="유선혁 노트북에서 T9 연결")
    client.add_argument("--host", default="192.168.50.2")
    client.add_argument("--port", type=int, default=8022)
    client.add_argument(
        "--identity", default=str(pathlib.Path.home() / ".ssh/t9_training")
    )
    client.add_argument("--known-hosts", required=True)
    client.add_argument(
        "--mountpoint", default=str(pathlib.Path.home() / "t9-data")
    )
    release = commands.add_parser(
        "unmount", help="유선혁 노트북의 T9 연결 해제"
    )
    release.add_argument(
        "--mountpoint", default=str(pathlib.Path.home() / "t9-data")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "serve":
            start_share(args)
        elif args.command == "mount":
            mount_share(args)
        else:
            run(
                [
                    "fusermount3",
                    "-u",
                    str(pathlib.Path(args.mountpoint).expanduser()),
                ]
            )
        return 0
    except (
        ValueError,
        OSError,
        subprocess.SubprocessError,
        paramiko.SSHException,
    ) as error:
        LOGGER.error("실패: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
