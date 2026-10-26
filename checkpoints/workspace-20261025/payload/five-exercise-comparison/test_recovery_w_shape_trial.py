"""몸통 아래로 붙는 상완과 옆으로 벌린 W를 구분하는 규칙 계약이다."""

import unittest
import numpy as np
from recovery_w_calibration import DEFAULT, classify as original
from recovery_w_shape_trial import classify


class WShapeTests(unittest.TestCase):
    """새 경계 비활성 시 기존 재현과 팔꿈치 아래 방향 판정을 확인한다."""

    def test_disabled_lower_bound_reproduces_original(self):
        x = np.array(
            [
                [60, 0.2, -0.95, 60, 0.2, -0.95, 0],
                [60, 0.2, -0.5, 60, 0.2, -0.5, 0],
                [0, -0.2, 0, 0, -0.2, 0, 0],
            ]
        )
        np.testing.assert_array_equal(
            classify(x, np.array([list(DEFAULT) + [-1, -1]])),
            original(x, np.array([DEFAULT])),
        )

    def test_arms_down_curl_is_partial_instead_of_normal_W(self):
        x = np.array(
            [
                [60, 0.2, -0.95, 60, 0.2, -0.95, 0],
                [60, 0.2, -0.5, 60, 0.2, -0.5, 0],
            ]
        )
        self.assertEqual(
            classify(x, np.array([list(DEFAULT) + [-0.9, -0.98]])).tolist(),
            [[1, 2]],
        )

    def test_missing_or_reversed_lower_bounds_fail(self):
        with self.assertRaises(ValueError):
            classify(
                np.full((1, 7), np.nan),
                np.array([list(DEFAULT) + [-0.9, -0.98]]),
            )
        with self.assertRaises(ValueError):
            classify(np.ones((1, 7)), np.array([list(DEFAULT) + [-0.98, -0.9]]))


if __name__ == "__main__":
    unittest.main()
