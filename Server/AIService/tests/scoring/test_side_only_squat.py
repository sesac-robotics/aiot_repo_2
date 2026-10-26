"""스쿼트 측면 관측에서 가려진 반대 관절을 요구하지 않는지 검증합니다.

A side-only recording must hold frontal criteria, not crash or invent them."""

import unittest

import numpy as np

from app.scoring.three_items import describe


class SideOnlySquatTest(unittest.TestCase):
    def test_missing_front_keeps_side_observations(self):
        seq = np.full((20, 33, 2), np.nan)
        for offset in (0, 1):
            for joint, xy in [
                (11, (100, 100)),
                (23, (100, 200)),
                (25, (120, 280)),
                (27, (120, 360)),
                (29, (110, 365)),
                (31, (145, 365)),
            ]:
                seq[:, joint + offset] = np.array(xy) + [offset * 8, 0]
        result = describe("squats", seq, None)
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0]["eligible"])
        self.assertTrue(result[1]["eligible"])
        self.assertFalse(result[2]["eligible"])
        self.assertFalse(result[3]["eligible"])
