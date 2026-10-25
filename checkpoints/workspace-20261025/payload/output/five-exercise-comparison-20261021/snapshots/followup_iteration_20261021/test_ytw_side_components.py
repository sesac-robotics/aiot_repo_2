"""측면 성분의 구도·미관측·총점 보류 계약을 검사한다."""

import unittest
import numpy as np
from ytw_side_components import evaluate


class SideComponentTest(unittest.TestCase):
    """정면 입력·누락 관절에서 전체 Y·총점을 만들지 않는다."""

    def test_front_view_is_not_assumed_side(self):
        result = evaluate(np.zeros((30, 33, 2)), False)
        self.assertIsNone(result["hinge_grade"])

    def test_missing_trunk_is_held(self):
        self.assertIsNone(
            evaluate(np.full((30, 33, 2), np.nan), True)["hinge_grade"]
        )

    def test_partial_observation_never_becomes_full_y(self):
        seq = np.full((30, 33, 2), np.nan)
        seq[:, 23] = [100, 200]
        seq[:, 11] = [200, 100]
        seq[:, 13] = [300, 0]
        seq[:, 15] = [400, -100]
        result = evaluate(seq, True)
        self.assertEqual(result["hinge_grade"], 2)
        self.assertIsNone(result["full_y_grade"])
        self.assertIsNone(result["total_score"])

    def test_visible_arm_is_selected_separately_from_trunk(self):
        seq = np.full((30, 33, 2), np.nan)
        seq[:, 23] = [100, 200]
        seq[:, 11] = [200, 100]
        seq[:, 24] = [100, 200]
        seq[:, 12] = [200, 100]
        seq[:, 14] = [300, 0]
        seq[:, 16] = [400, -100]
        result = evaluate(seq, True)
        self.assertEqual(result["selected_arm_side"], "right")
        self.assertEqual(result["y_sagittal_component_grade"], 2)


if __name__ == "__main__":
    unittest.main()
