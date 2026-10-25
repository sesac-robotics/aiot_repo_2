"""하체·얼굴 없는 사건 입력과 기하 불변성을 검사한다."""

import unittest
import numpy as np
from upper_support_event_model import event_features


class UpperEventTest(unittest.TestCase):
    """필요하지 않은 신체 부위는 특징에 영향을 주지 않는다."""

    def setUp(self):
        self.seq = np.full((30, 33, 2), np.nan)
        self.seq[:, 11] = [100, 100]
        self.seq[:, 13] = [100, 200]
        self.seq[20:, 13, 0] += 60

    def test_cropped_lower_body_is_allowed(self):
        x, note = event_features(self.seq)
        self.assertIsNotNone(x)
        self.assertEqual(note["coverage"], 1.0)

    def test_head_motion_does_not_change_features(self):
        a, _ = event_features(self.seq)
        self.seq[:, 0] = np.arange(60).reshape(30, 2)
        b, _ = event_features(self.seq)
        np.testing.assert_equal(a, b)

    def test_translation_and_scale(self):
        a, _ = event_features(self.seq)
        b, _ = event_features(self.seq * 2 + [70, 30])
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_missing_elbow_is_held(self):
        self.seq[:, 13] = np.nan
        x, _ = event_features(self.seq)
        self.assertIsNone(x)


if __name__ == "__main__":
    unittest.main()
