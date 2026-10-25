"""W 인식 실패와 실제로 관측된 미수행을 혼동하지 않도록 검사한다."""

import unittest
import numpy as np
from w_observed_absence_trial import detect


def sequence():
    """T 유지 후 양팔을 내리는 결측 없는 합성 픽셀 시계열을 반환한다."""
    seq = np.zeros((30, 33, 2), dtype=float)
    for shoulder, elbow, wrist, hip, sign in (
        (11, 13, 15, 23, -1),
        (12, 14, 16, 24, 1),
    ):
        seq[:, shoulder] = [sign * 10, 0]
        seq[:, hip] = [sign * 10, 100]
        seq[:, elbow] = [sign * 10, 40]
        seq[:, wrist] = [sign * 10, 80]
        seq[4:16, elbow] = [sign * 45, 0]
        seq[4:16, wrist] = [sign * 80, 0]
    return seq


class ObservedAbsenceTests(unittest.TestCase):
    """관측된 실패·진짜 W·가림·T 없음의 경계를 검사한다."""

    def test_observed_return_without_w_is_a_candidate(self):
        """T 뒤 충분히 관측된 팔 내림은 미수행 후보가 될 수 있다."""
        result = detect(sequence())
        self.assertIsNotNone(result)
        self.assertEqual(result["measured_shape_grade"], 0)

    def test_true_w_evidence_prevents_absence_claim(self):
        """W 손 높이·굽힘이 연속으로 보이면 나중에 팔을 내려도 미수행이 아니다."""
        seq = sequence()
        for e, w, sign in ((13, 15, -1), (14, 16, 1)):
            seq[16:21, e] = [sign * 40, 0]
            seq[16:21, w] = [sign * 40, -30]
        self.assertIsNone(detect(seq))

    def test_missing_return_is_not_an_error_grade(self):
        """후반이 가려진 것을 W 0등급으로 바꾸지 않는다."""
        seq = sequence()
        seq[18:] = np.nan
        self.assertIsNone(detect(seq))

    def test_no_t_is_not_an_absent_w(self):
        """T를 확인하지 못한 정지 자세를 순서상 W 실패로 부르지 않는다."""
        seq = sequence()
        seq[4:16] = seq[0]
        self.assertIsNone(detect(seq))


if __name__ == "__main__":
    unittest.main()
