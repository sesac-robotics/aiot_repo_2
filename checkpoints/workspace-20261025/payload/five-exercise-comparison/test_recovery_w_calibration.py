"""보정 후보가 W 양팔·대칭 의미와 결측 금지 계약을 지키는지 검사한다."""

import unittest
import numpy as np
from recovery_w_calibration import DEFAULT, classify


class WCalibrationTests(unittest.TestCase):
    """미관측 값을 만점으로 채우거나 한 팔 수행을 정상으로 만드는 오류를 막는다."""

    def test_bilateral_shape_and_asymmetry(self):
        x = np.array(
            [
                [60, 0.2, 0, 60, 0.2, 0, 0],
                [0, -0.2, 0.9, 60, 0.2, 0, 60],
                [0, -0.2, 0.9, 0, -0.2, 0.9, 0],
                [60, 0.2, 0, 20, 0.2, 0, 40],
            ]
        )
        self.assertEqual(
            classify(x, np.array([DEFAULT])).tolist(), [[2, 1, 0, 1]]
        )

    def test_missing_geometry_is_not_imputed(self):
        with self.assertRaises(ValueError):
            classify(np.full((1, 7), np.nan), np.array([DEFAULT]))

    def test_reversed_boundaries_are_rejected(self):
        invalid = list(DEFAULT)
        invalid[0] = 1
        with self.assertRaises(ValueError):
            classify(np.ones((1, 7)), np.array([invalid]))


if __name__ == "__main__":
    unittest.main()
