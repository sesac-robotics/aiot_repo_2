"""상지 지지 변경의 양성 근거와 미관측 보류를 검증한다."""

import unittest
import numpy as np
from plank_support_motion import detect


class SupportMotionTest(unittest.TestCase):
    """얼굴·다리 없이 보이는 변경만 검출하며 정상 점수를 만들지 않는다."""

    def setUp(self):
        self.seq = np.full((30, 33, 2), np.nan)
        self.seq[:, 11] = [100, 100]
        self.seq[:, 23] = [300, 100]
        self.seq[:, 13] = [100, 200]

    def test_stable_does_not_grant_normal_grade(self):
        self.assertIsNone(detect(self.seq)["continuity_grade_upper_bound"])

    def test_arm_change_without_legs(self):
        self.seq[20:, 13, 0] += 100
        self.assertEqual(detect(self.seq)["continuity_grade_upper_bound"], 1)

    def test_translation_scale_invariant(self):
        self.seq[20:, 13, 0] += 100
        a = detect(self.seq)
        b = detect(self.seq * 3 + [40, 80])
        self.assertAlmostEqual(
            a["relative_displacement"], b["relative_displacement"]
        )

    def test_missing_elbow_is_not_a_change(self):
        self.seq[:, 13] = np.nan
        self.assertIsNone(detect(self.seq)["continuity_grade_upper_bound"])


if __name__ == "__main__":
    unittest.main()
