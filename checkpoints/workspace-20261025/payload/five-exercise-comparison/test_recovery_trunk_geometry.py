"""방향 특징의 촬영 좌표 불변성과 결측 보존을 확인한다."""

import unittest
import numpy as np
from recovery_trunk_geometry import signed_offsets


class TrunkGeometryTests(unittest.TestCase):
    """골반 방향을 잃거나 결측을 정상으로 채우는 회귀를 막는다."""

    def test_sag_and_pike_have_opposite_signs(self):
        seq = np.full((2, 33, 2), np.nan)
        seq[:, 11] = [10, 10]
        seq[:, 27] = [110, 10]
        seq[:, 23] = [[50, 20], [50, 0]]
        values = signed_offsets(seq, "left")
        self.assertGreater(values[0], 0)
        self.assertLess(values[1], 0)
        mirrored = seq.copy()
        mirrored[:, :, 0] = 200 - mirrored[:, :, 0]
        np.testing.assert_allclose(values, signed_offsets(mirrored, "left"))
        np.testing.assert_allclose(values, signed_offsets(seq * 2 + 30, "left"))

    def test_missing_or_degenerate_joints_remain_missing(self):
        seq = np.full((2, 33, 2), np.nan)
        seq[0, [11, 23, 27]] = [10, 10]
        self.assertTrue(np.isnan(signed_offsets(seq, "left")).all())


if __name__ == "__main__":
    unittest.main()
