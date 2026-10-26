"""우분투에서 카메라 수신 노트북 간 유선 시험 연결을 설정한다.

NetworkManager와 GTK 3가 필요하다. Wi-Fi·기존 프로파일·방화벽은
수정하지 않는다. 사용자 적용 요청마다 자동 연결하지 않는 전용 프로파일을
만들고, 원복 시 해당 프로파일만 삭제한다. AI 서버와 ESP32는 제어하지 않는다.
실행: /usr/bin/python3 tools/wired_camera_link.py
"""

import concurrent.futures
import dataclasses
import ipaddress
import json
import logging
import shutil
import subprocess
import uuid
import webbrowser

LOGGER = logging.getLogger(__name__)


def run_command(arguments: list[str], timeout_s: int = 20) -> str:
    """셸을 거치지 않고 명령을 실행하고 표준 출력을 반환한다.

    Args:
        arguments: 실행 파일과 개별 인자. 비밀번호는 전달하지 않는다.
        timeout_s: 최대 대기 시간(초).

    Returns:
        앞뒤 공백을 제거한 표준 출력.

    Raises:
        RuntimeError: 명령 실패, 실행 파일 부재 또는 시간 초과.
    """
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"명령 실행 실패: {error}") from error
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "명령 실패"
        )
    return result.stdout.strip()


@dataclasses.dataclass(frozen=True)
class LinkSettings:
    """한 번의 시험 연결 입력을 보관한다. IP는 IPv4, prefix는 비트 수다.

    local_cidr는 자신의 주소와 prefix, peer_ip는 상대 주소다.
    server_ip는 영상 서버 A의 주소이며 port는 TCP 포트(1~65535)다.
    이 객체는 네트워크 자원을 소유하거나 설정을 적용하지 않는다.
    """

    interface: str
    local_cidr: str
    peer_ip: str
    server_ip: str
    port: int

    @property
    def stream_url(self) -> str:
        """현재 서버 설정의 HTTP MJPEG 주소를 반환한다."""
        return f"http://{self.server_ip}:{self.port}/stream"


def validate_settings(
    interface: str,
    local_ip: str,
    prefix: str,
    peer_ip: str,
    port: str,
    role: str,
) -> LinkSettings:
    """화면 입력을 검증하고 정규화된 설정을 반환한다.

    Args:
        interface: 선택한 유선 장치 이름. 실제 장치 여부는 적용 시 검사한다.
        local_ip: 자신의 IPv4 주소.
        prefix: 서브넷 prefix 비트 수(1~30).
        peer_ip: 같은 서브넷에 있는 상대 IPv4 주소.
        port: 기존 영상 서버의 TCP 포트(1~65535).
        role: A는 영상 서버, B는 영상 확인 노트북.

    Returns:
        검증된 연결 설정. 네트워크를 변경하지 않는다.

    Raises:
        ValueError: 주소·범위·역할이 잘못되거나 두 주소가 같은 경우.
    """
    if not interface or role not in ("A", "B"):
        raise ValueError("유선 장치와 A/B 역할을 선택하세요.")
    bits, port_number = int(prefix), int(port)
    if not 1 <= bits <= 30 or not 1 <= port_number <= 65535:
        raise ValueError("prefix는 1~30, 포트는 1~65535여야 합니다.")
    local = ipaddress.IPv4Interface(f"{local_ip.strip()}/{bits}")
    peer = ipaddress.IPv4Address(peer_ip.strip())
    network = local.network
    for address in (local.ip, peer):
        if (
            address not in network
            or address in (network.network_address, network.broadcast_address)
            or address.is_multicast
            or address.is_unspecified
            or address.is_loopback
            or address.is_link_local
        ):
            raise ValueError("같은 서브넷의 사용 가능한 IPv4를 입력하세요.")
    if local.ip == peer:
        raise ValueError("자신과 상대의 IP는 달라야 합니다.")
    server = local.ip if role == "A" else peer
    return LinkSettings(
        interface, str(local), str(peer), str(server), port_number
    )


class NetworkLink:
    """이 실행에서 생성한 NetworkManager 프로파일 하나를 소유한다.

    GUI의 단일 작업 스레드에서만 호출한다. 기존 활성 연결 UUID를 보관해
    원복 시 복구한다. 앱을 강제 종료하면 시험 프로파일이 남을 수 있으나
    autoconnect=no이므로 부팅 후 자동 연결하지 않는다.
    """

    def __init__(self) -> None:
        """생성한 프로파일과 이전 연결의 소유 상태를 초기화한다."""
        self.profile_uuid: str | None = None
        self.previous_uuid: str | None = None

    def ethernet_devices(self) -> list[str]:
        """케이블 연결 여부와 무관하게 Ethernet 장치 이름을 반환한다.

        Raises:
            RuntimeError: NetworkManager 조회 실패.
        """
        output = run_command(["nmcli", "-t", "-f", "DEVICE,TYPE", "device"])
        return [
            line.rsplit(":", 1)[0]
            for line in output.splitlines()
            if line.endswith(":ethernet")
        ]

    def apply(self, settings: LinkSettings) -> str:
        """Wi-Fi를 유지하며 전용 수동 IPv4 연결을 활성화한다.

        Args:
            settings: 검증된 자신의 주소·상대 주소·장치 설정.

        Returns:
            활성화한 프로파일 UUID와 시험 안내.

        Raises:
            RuntimeError: 대역 충돌, 권한·케이블·장치 문제 또는 적용 실패.
                실패 시 생성한 프로파일 삭제와 이전 연결 복구를 시도한다.
        """
        if self.profile_uuid or self.previous_uuid:
            raise RuntimeError("기존 시험 연결을 먼저 원복하세요.")
        if settings.interface not in self.ethernet_devices():
            raise RuntimeError("선택한 장치는 현재 유선 장치가 아닙니다.")
        addresses = json.loads(run_command(["ip", "-j", "-4", "address"]))
        network = ipaddress.IPv4Interface(settings.local_cidr).network
        for device in addresses:
            if device["ifname"] == settings.interface:
                continue
            for address in device.get("addr_info", []):
                other = ipaddress.IPv4Interface(
                    f"{address['local']}/{address['prefixlen']}"
                ).network
                if network.overlaps(other):
                    raise RuntimeError(
                        f"{device['ifname']}의 {other}와 대역이 겹칩니다. "
                        "다른 IP 대역을 선택하세요."
                    )
        previous = run_command(
            [
                "nmcli",
                "-g",
                "GENERAL.CON-UUID",
                "device",
                "show",
                settings.interface,
            ]
        )
        self.previous_uuid = previous if previous not in ("", "--") else None
        profile_uuid = str(uuid.uuid4())
        run_command(
            [
                "nmcli",
                "connection",
                "add",
                "type",
                "ethernet",
                "ifname",
                settings.interface,
                "con-name",
                f"camera-link-{profile_uuid[:8]}",
                "connection.uuid",
                profile_uuid,
                "connection.autoconnect",
                "no",
                "ipv4.method",
                "manual",
                "ipv4.addresses",
                settings.local_cidr,
                "ipv4.never-default",
                "yes",
                "ipv6.method",
                "disabled",
            ]
        )
        self.profile_uuid = profile_uuid
        try:
            run_command(
                [
                    "nmcli",
                    "--wait",
                    "15",
                    "connection",
                    "up",
                    "uuid",
                    profile_uuid,
                ],
                timeout_s=25,
            )
        except RuntimeError as error:
            try:
                self.restore()
            except RuntimeError as cleanup_error:
                raise RuntimeError(
                    f"{error}\n원복 미완료: {cleanup_error}\n원복을 재시도하세요."
                ) from error
            raise
        return (
            f"적용 완료: {settings.local_cidr}\n프로파일 UUID: {profile_uuid}\n"
            "상대 노트북도 설정한 뒤 연결 확인을 누르세요.\n"
            "Wi-Fi·방화벽·AI 서버 설정은 변경하지 않았습니다."
        )

    def restore(self) -> str:
        """이 도구의 시험 프로파일을 삭제하고 이전 연결을 복구한다.

        Returns:
            원복 결과. 이전 활성 연결이 없었다면 별도 연결을 시작하지 않는다.

        Raises:
            RuntimeError: 삭제·복구 실패. 미완료 소유 상태는 재시도용으로 유지.
        """
        if self.profile_uuid:
            run_command(
                [
                    "nmcli",
                    "connection",
                    "delete",
                    "uuid",
                    self.profile_uuid,
                ]
            )
            self.profile_uuid = None
        if self.previous_uuid:
            run_command(
                [
                    "nmcli",
                    "--wait",
                    "15",
                    "connection",
                    "up",
                    "uuid",
                    self.previous_uuid,
                ],
                timeout_s=25,
            )
            self.previous_uuid = None
        return "원복 완료. 이 도구가 만든 연결만 삭제했습니다."

    def check(self, settings: LinkSettings) -> str:
        """상대 IP의 경로·ICMP 응답을 검사한다. 영상 지연 측정은 아니다.

        Args:
            settings: 검사할 유선 장치 및 상대 주소.

        Returns:
            실제 선택 경로와 ping 결과. 방화벽으로 ICMP가 차단될 수도 있다.

        Raises:
            RuntimeError: 유선 경로 불일치 또는 상대 응답 실패.
        """
        routes = json.loads(
            run_command(
                [
                    "ip",
                    "-j",
                    "route",
                    "get",
                    settings.peer_ip,
                ]
            )
        )
        if not routes or routes[0].get("dev") != settings.interface:
            raise RuntimeError(
                "상대 IP로 가는 경로가 선택한 유선 장치가 아닙니다."
            )
        ping = run_command(
            [
                "ping",
                "-n",
                "-I",
                settings.interface,
                "-c",
                "5",
                "-W",
                "1",
                settings.peer_ip,
            ],
            timeout_s=12,
        )
        return (
            f"유선 경로 확인: {settings.interface} → {settings.peer_ip}\n{ping}\n"
            "ping은 네트워크 왕복 시간이며 영상 전체 지연과 다릅니다."
        )


class LinkWindow:
    """GTK 화면을 소유하며 느린 명령을 단일 Worker에 위임한다.

    위젯 접근은 GUI 스레드에서만 수행한다. 종료 전 명시적인 원복을
    요구하며, Worker 실행 중에는 중복 적용과 종료를 차단한다.
    """

    def __init__(self) -> None:
        """GTK 구성 요소와 초기 화면을 만든다. 네트워크는 변경하지 않는다."""
        import gi  # 운영체제 GTK 의존성을 실행 시에만 로드한다.

        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk

        self._gtk = Gtk
        self._glib = GLib
        self._network = NetworkLink()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._window = Gtk.Window(title="ESP32-CAM 노트북 유선 연결 시험")
        self._window.set_default_size(720, 620)
        self._window.set_border_width(18)
        self._window.connect("delete-event", self._on_close)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._window.add(box)
        heading = Gtk.Label(
            label="ESP32-CAM → Wi-Fi → A(AI 서버) → 유선 → B(영상 확인)"
        )
        box.pack_start(heading, False, False, 0)
        note = Gtk.Label(
            label=(
                "두 노트북에 같은 프로그램을 실행하고 LAN 케이블을 연결하세요.\n"
                "A의 Wi-Fi는 유지하세요. 적용 버튼을 누르기 전에는 조회만 합니다."
            )
        )
        box.pack_start(note, False, False, 0)
        grid = Gtk.Grid(column_spacing=16, row_spacing=10)
        box.pack_start(grid, False, False, 0)
        self._role = Gtk.ComboBoxText()
        self._role.append("A", "A · 카메라 영상을 처리하는 서버")
        self._role.append("B", "B · 영상을 확인하는 노트북")
        self._role.set_active_id("B")
        self._interface = Gtk.ComboBoxText()
        self._local = Gtk.Entry(text="192.168.50.2")
        self._peer = Gtk.Entry(text="192.168.50.1")
        self._prefix = Gtk.Entry(text="24")
        self._port = Gtk.Entry(text="8000")
        self._fields = [
            self._role,
            self._interface,
            self._local,
            self._prefix,
            self._peer,
            self._port,
        ]
        for row, (title, widget) in enumerate(
            zip(
                [
                    "이 노트북 역할",
                    "유선 장치",
                    "자신의 IPv4",
                    "서브넷 prefix",
                    "상대 노트북 IPv4",
                    "A 서버 포트 (기존 값)",
                ],
                self._fields,
            )
        ):
            label = Gtk.Label(label=title, xalign=0)
            grid.attach(label, 0, row, 1, 1)
            widget.set_hexpand(True)
            grid.attach(widget, 1, row, 1, 1)
        self._role.connect("changed", self._on_role)
        actions = Gtk.Box(spacing=8)
        box.pack_start(actions, False, False, 0)
        self._buttons = []
        for title, callback in [
            ("장치 새로고침", self._on_refresh),
            ("유선 IP 적용", self._on_apply),
            ("연결 확인", self._on_check),
            ("영상 열기", self._on_open),
            ("원복", self._on_restore),
        ]:
            button = Gtk.Button(label=title)
            button.connect("clicked", callback)
            actions.pack_start(button, True, True, 0)
            self._buttons.append(button)
        self._output = Gtk.TextView(editable=False, cursor_visible=False)
        self._output.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        scroll = Gtk.ScrolledWindow()
        scroll.add(self._output)
        box.pack_start(scroll, True, True, 0)
        self._set_output(
            "기존 영상 창을 닫고 한 창씩 Wi-Fi/유선을 비교하세요.\n"
            "영상이 열리지 않으면 서버 바인딩 주소·포트·방화벽을 확인하세요.\n"
            "이 도구는 AI 서버를 시작하거나 서버 소스를 수정하지 않습니다."
        )
        self._window.show_all()
        self._on_refresh(None)

    def _set_output(self, message: str) -> None:
        self._output.get_buffer().set_text(message)

    def _settings(self) -> LinkSettings:
        return validate_settings(
            self._interface.get_active_text() or "",
            self._local.get_text(),
            self._prefix.get_text(),
            self._peer.get_text(),
            self._port.get_text(),
            self._role.get_active_id() or "",
        )

    def _submit(self, task, callback=None) -> None:
        """작업을 직렬 실행하고 결과 처리를 GUI 스레드로 돌려보낸다."""
        if self._busy:
            return
        self._busy = True
        for widget in self._buttons + self._fields:
            widget.set_sensitive(False)
        self._set_output("처리 중… 권한 요청 창이 나오면 확인하세요.")
        future = self._executor.submit(task)

        def completed(result) -> None:
            self._glib.idle_add(self._finish, result, callback)

        future.add_done_callback(completed)

    def _finish(self, future, callback) -> bool:
        self._busy = False
        for widget in self._buttons + self._fields:
            widget.set_sensitive(True)
        try:
            result = future.result()
            if callback:
                callback(result)
            else:
                self._set_output(str(result))
        except Exception as error:  # 최상위 Worker 경계에서 실패를 표시한다.
            LOGGER.exception("Network operation failed")
            self._set_output(
                f"실패: {error}\n케이블·양쪽 IP·NetworkManager 권한을 확인하세요."
            )
        return False

    def _on_role(self, _widget) -> None:
        is_a = self._role.get_active_id() == "A"
        self._local.set_text("192.168.50.1" if is_a else "192.168.50.2")
        self._peer.set_text("192.168.50.2" if is_a else "192.168.50.1")

    def _on_refresh(self, _widget) -> None:
        def display(devices: list[str]) -> None:
            self._interface.remove_all()
            for device in devices:
                self._interface.append_text(device)
            self._interface.set_active(0)
            self._set_output(
                "유선 장치: "
                + (", ".join(devices) or "없음 · LAN 어댑터 확인")
                + "\n기본 IP가 기존 Wi-Fi/VPN 대역과 겹치면 변경하세요."
            )

        self._submit(self._network.ethernet_devices, display)

    def _on_apply(self, _widget) -> None:
        try:
            settings = self._settings()
        except ValueError as error:
            self._set_output(f"입력 확인: {error}")
            return
        self._submit(lambda: self._network.apply(settings))

    def _on_check(self, _widget) -> None:
        try:
            settings = self._settings()
        except ValueError as error:
            self._set_output(f"입력 확인: {error}")
            return
        self._submit(lambda: self._network.check(settings))

    def _on_open(self, _widget) -> None:
        try:
            settings = self._settings()
        except ValueError as error:
            self._set_output(f"입력 확인: {error}")
            return
        webbrowser.open(settings.stream_url)
        self._set_output(
            f"영상 주소: {settings.stream_url}\n"
            "기존 서버가 이 IP에서 접속을 받아야 합니다.\n"
            "다른 영상 창은 닫고 비교하세요."
        )

    def _on_restore(self, _widget) -> None:
        self._submit(self._network.restore)

    def _on_close(self, _window, _event) -> bool:
        if self._busy:
            return True
        if self._network.profile_uuid or self._network.previous_uuid:
            self._set_output(
                "종료 전에 원복 버튼을 눌러 시험 연결을 정리하세요."
            )
            return True
        self._executor.shutdown(wait=False)
        self._gtk.main_quit()
        return False

    def run(self) -> None:
        """GTK 이벤트 루프를 실행한다. 창 종료 후 반환하며 설정은 자동 적용하지 않는다."""
        self._gtk.main()


def main() -> None:
    """필수 명령을 확인하고 GUI를 실행한다.

    Raises:
        SystemExit: 필수 명령 또는 GTK가 없거나 화면 초기화가 실패한 경우.
    """
    logging.basicConfig(level=logging.INFO)
    missing = [
        name for name in ("nmcli", "ip", "ping") if not shutil.which(name)
    ]
    if missing:
        raise SystemExit("필수 명령이 없습니다: " + ", ".join(missing))
    try:
        LinkWindow().run()
    except (ImportError, ValueError, RuntimeError) as error:
        raise SystemExit(
            f"화면 실행 실패: {error}\n"
            "우분투 시스템 Python으로 실행하세요. GTK가 없으면 "
            "python3-gi, gir1.2-gtk-3.0 패키지가 필요합니다."
        ) from error


if __name__ == "__main__":
    main()
