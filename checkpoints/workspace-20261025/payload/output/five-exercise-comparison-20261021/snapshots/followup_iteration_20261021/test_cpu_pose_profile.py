"""CPU 대체 경로가 기존 GPU 보호·미관측 계약을 우회하지 않는지 검사한다."""

import unittest

import numpy as np

from extract import extract_view
from plank_world_trial import measurements


class CpuProfileTests(unittest.TestCase):
    """실제 추출 없이 장치·자원 프로필 입력과 관측 실패 조건을 확인한다."""

    def test_gpu_does_not_accept_custom_resource_guard(self) -> None:
        """GPU 경로는 CPU 전용 배터리 보호 함수로 실행할 수 없다."""
        called = []
        with self.assertRaises(ValueError):
            extract_view(
                {},
                "primary",
                "unused",
                delegate="GPU",
                resource_checker=lambda: called.append(True),
            )
        self.assertEqual(called, [])

    def test_invalid_delegate_fails_before_any_io(self) -> None:
        """철자가 잘못된 실행 장치를 기본 GPU로 조용히 대체하지 않는다."""
        with self.assertRaises(ValueError):
            extract_view({}, "primary", "unused", delegate="CPu")

    def test_estimated_world_does_not_fill_invisible_image_joints(self) -> None:
        """3D 추정값이 있어도 화면 미관측은 학습 특징으로 되살리지 않는다."""
        image = np.full((5, 33, 2), np.nan)
        world = np.ones((5, 33, 3))
        x, note = measurements(image, world)
        self.assertIsNone(x)
        self.assertEqual(note["joint_observation_fraction"], 0)
