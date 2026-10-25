"""실제 네트워크를 변경하지 않고 유선 시험 도구의 경계 조건을 검증한다."""

import unittest
from unittest import mock

import wired_camera_link as link


class LinkTests(unittest.TestCase):
    """입력·경로·실패 원복 계약을 가짜 명령 실행기로 확인한다."""

    def test_roles_select_server_address(self) -> None:
        """A는 자신의 주소, B는 상대 주소로 영상 URL을 만든다."""
        for role, expected in (("A", "192.168.50.2"), ("B", "192.168.50.1")):
            settings = link.validate_settings(
                "eno1", "192.168.50.2", "24", "192.168.50.1", "8000", role
            )
            self.assertEqual(settings.server_ip, expected)

    def test_invalid_addresses_are_rejected(self) -> None:
        """동일 IP·다른 대역·예약 주소·범위 오류를 거부한다."""
        for local, prefix, peer, port in [
            ("192.168.50.2", "24", "192.168.50.2", "8000"),
            ("192.168.50.2", "24", "192.168.51.1", "8000"),
            ("192.168.50.0", "24", "192.168.50.1", "8000"),
            ("192.168.50.2", "24", "192.168.50.255", "8000"),
            ("127.0.0.2", "24", "127.0.0.1", "8000"),
            ("192.168.50.2", "31", "192.168.50.1", "8000"),
            ("192.168.50.2", "24", "192.168.50.1", "65536"),
        ]:
            with self.subTest(local=local, peer=peer, port=port):
                with self.assertRaises(ValueError):
                    link.validate_settings(
                        "eno1", local, prefix, peer, port, "B"
                    )

    @mock.patch.object(link, "run_command")
    def test_wifi_overlap_prevents_mutation(self, command) -> None:
        """Wi-Fi 대역 충돌을 설정 변경 전에 거부한다."""
        command.side_effect = [
            "eno1:ethernet",
            '[{"ifname":"wlan0","addr_info":['
            '{"local":"192.168.50.8","prefixlen":24}]}]',
        ]
        settings = link.validate_settings(
            "eno1", "192.168.50.2", "24", "192.168.50.1", "8000", "B"
        )
        with self.assertRaisesRegex(RuntimeError, "겹칩니다"):
            link.NetworkLink().apply(settings)
        self.assertEqual(command.call_count, 2)

    @mock.patch.object(link, "run_command")
    def test_failed_activation_restores_previous_connection(self, command):
        """활성화 실패 시 시험 프로파일만 삭제하고 이전 UUID를 복구한다."""
        command.side_effect = [
            "eno1:ethernet",
            "[]",
            "previous-uuid",
            "created",
            RuntimeError("케이블 없음"),
            "deleted",
            "restored",
        ]
        network = link.NetworkLink()
        settings = link.validate_settings(
            "eno1", "192.168.50.2", "24", "192.168.50.1", "8000", "B"
        )
        with self.assertRaisesRegex(RuntimeError, "케이블 없음"):
            network.apply(settings)
        self.assertIsNone(network.profile_uuid)
        self.assertIsNone(network.previous_uuid)
        self.assertIn("delete", command.call_args_list[-2].args[0])
        self.assertEqual(
            command.call_args_list[-1].args[0][-1], "previous-uuid"
        )

    @mock.patch.object(link, "run_command")
    def test_wrong_route_is_rejected(self, command) -> None:
        """Wi-Fi 경로를 유선 성공으로 보고하지 않는다."""
        command.return_value = '[{"dev":"wlan0"}]'
        settings = link.validate_settings(
            "eno1", "192.168.50.2", "24", "192.168.50.1", "8000", "B"
        )
        with self.assertRaisesRegex(RuntimeError, "유선 장치"):
            link.NetworkLink().check(settings)
        self.assertEqual(command.call_count, 1)

    @mock.patch.object(link, "run_command")
    def test_failed_restore_retains_retry_state(self, command) -> None:
        """이전 연결 복구 실패 후에도 재시도할 UUID를 보관한다."""
        network = link.NetworkLink()
        network.profile_uuid = "owned-uuid"
        network.previous_uuid = "previous-uuid"
        command.side_effect = ["deleted", RuntimeError("복구 실패")]
        with self.assertRaises(RuntimeError):
            network.restore()
        self.assertIsNone(network.profile_uuid)
        self.assertEqual(network.previous_uuid, "previous-uuid")


if __name__ == "__main__":
    unittest.main()
