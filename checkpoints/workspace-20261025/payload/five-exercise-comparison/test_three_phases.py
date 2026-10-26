"""완료 구간과 정지·미복귀·관절 결측을 구별하는 단계 검출을 검사한다."""

import unittest

import numpy as np

import three_phases


class ThreePhaseTests(unittest.TestCase):
    """실제 각도를 가진 합성 팔로 단계 실패 조건을 검사한다."""

    def sequence(self, degrees: list[float]) -> np.ndarray:
        """팔꿈치 각도(도)로부터 왼팔 픽셀 좌표를 만든다.

        Args:
            degrees: 시간순 팔꿈치 각도 목록.

        Returns:
            np.ndarray: 나머지 관절이 NaN인 N×33×2 좌표.
        """
        seq = np.full((len(degrees), 33, 2), np.nan)
        angle = np.radians(degrees)
        seq[:, 13] = [200, 200]
        seq[:, 11] = [300, 200]
        seq[:, 15] = np.column_stack(
            [200 + 100 * np.cos(angle), 200 + 100 * np.sin(angle)]
        )
        return seq

    def test_complete_shallow_error_is_not_filtered_as_bad(self) -> None:
        """깊이가 얕은 오류여도 시작·하강·복귀가 있으면 후보를 보존한다."""
        seq = self.sequence([170, 170, 165, 155, 150, 155, 165, 170, 170])
        result = three_phases.detect("pushups", seq, np.arange(9) * 3)
        self.assertEqual(len(result["repetitions"]), 1)

    def test_static_pose_is_not_a_completed_rep(self) -> None:
        """잘 보이는 정지 팔을 한 반복으로 세지 않는다."""
        seq = self.sequence([170] * 15)
        self.assertEqual(
            three_phases.detect("pushups", seq, np.arange(15))["repetitions"],
            [],
        )

    def test_descent_without_return_is_not_complete(self) -> None:
        """내려가기만 한 구간에 복귀 단계를 만들어내지 않는다."""
        seq = self.sequence([170, 170, 160, 150, 130, 110, 90, 90, 90])
        self.assertEqual(
            three_phases.detect("pushups", seq, np.arange(9))["repetitions"], []
        )

    def test_long_gap_is_not_interpolated_into_a_rep(self) -> None:
        """관절이 오래 사라진 구간을 보간해 완료했다고 하지 않는다."""
        seq = self.sequence(
            [
                170,
                170,
                170,
                160,
                140,
                100,
                80,
                100,
                140,
                160,
                170,
                170,
                170,
                170,
                170,
                170,
                170,
                170,
                170,
                170,
            ]
        )
        seq[8:11, 15] = np.nan
        result = three_phases.detect("pushups", seq, np.arange(20))
        self.assertEqual(result["hold"], "long_phase_gap")
