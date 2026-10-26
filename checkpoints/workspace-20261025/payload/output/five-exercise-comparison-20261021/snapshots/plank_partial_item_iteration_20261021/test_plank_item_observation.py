"""부분 입력에서 항목 보류 범위와 특징의 관측 계약을 검사한다."""

import unittest
import numpy as np
from plank_item_observation import assess, features


class ItemObservationTest(unittest.TestCase):
    """필요하지 않은 신체 잘림과 필요한 관절 누락을 구별한다."""

    def setUp(self):
        self.seq = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (200, 100)),
            (13, (200, 150)),
            (23, (400, 120)),
            (25, (550, 135)),
            (27, (700, 150)),
        ]:
            self.seq[:, joint] = xy

    def test_face_hand_toe_not_required(self):
        self.assertTrue(
            all(r["eligible"] for r in assess(self.seq, 3, [True] * 4)["items"])
        )

    def test_no_legs_still_elbow(self):
        self.seq[:, [25, 27]] = np.nan
        result = assess(self.seq, 3, [True] * 4)
        self.assertEqual(
            [r["eligible"] for r in result["items"]],
            [False, True, False, False],
        )
        self.assertIsNone(result["total_score"])

    def test_no_elbow_keeps_trunk(self):
        self.seq[:, 13] = np.nan
        self.assertEqual(
            [r["eligible"] for r in assess(self.seq, 3, [True] * 4)["items"]],
            [True, False, True, True],
        )

    def test_short_segment_not_continuity(self):
        self.assertFalse(
            assess(self.seq, 0.8, [True] * 4)["items"][3]["eligible"]
        )
        self.assertTrue(
            assess(self.seq, 0.8, [True] * 4)["items"][0]["eligible"]
        )

    def test_unrelated_joints_do_not_enter_elbow_features(self):
        a, names = features(self.seq, 1, "left")
        self.seq[:, 27] = [900, 900]
        b, other = features(self.seq, 1, "left")
        np.testing.assert_equal(a, b)
        self.assertEqual(names, other)

    def test_fps_roundoff_not_a_short_hold(self):
        self.assertTrue(
            assess(self.seq, 2.999767, [True] * 4)["items"][3]["eligible"]
        )
        self.assertFalse(
            assess(self.seq, 2.95, [True] * 4)["items"][3]["eligible"]
        )

    def test_each_item_can_choose_visible_side(self):
        self.seq[:, 13] = np.nan
        for left in [11, 23]:
            self.seq[:, left + 1] = self.seq[:, left]
        self.seq[:, 14] = [200, 150]
        result = assess(self.seq, 3, [True] * 4)
        self.assertEqual(result["items"][0]["side"], "left")
        self.assertEqual(result["items"][1]["side"], "right")
        self.assertTrue(result["items"][1]["eligible"])

    def test_required_joints_must_overlap_in_time(self):
        for joint, start in [(11, 0), (23, 6), (27, 12)]:
            self.seq[start : start + 6, joint] = np.nan
        result = assess(self.seq, 3, [True] * 4)
        self.assertFalse(result["items"][0]["eligible"])
        self.assertAlmostEqual(
            result["items"][0]["simultaneous_observation"], 0.4
        )

    def test_view_permission_still_required(self):
        self.assertFalse(
            assess(self.seq, 3, [False, True, True, True])["items"][0][
                "eligible"
            ]
        )


if __name__ == "__main__":
    unittest.main()
