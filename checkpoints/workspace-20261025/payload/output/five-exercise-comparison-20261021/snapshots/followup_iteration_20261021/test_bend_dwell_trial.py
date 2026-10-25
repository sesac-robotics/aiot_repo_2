"""굽힘 유지 실험이 휴식·누락에서 W를 만들지 않는지 확인한다."""

import unittest
import numpy as np
import test_phases
import ytw_phases
import ytw_bend_dwell_trial


class BendDwellTests(unittest.TestCase):
    """실험에서 기존 완전 구간 보존과 관측 실패 동작을 검사한다."""

    def test_complete_windows_preserved(self):
        """기존 완전 단계의 위치는 바꾸지 않는다."""
        seq = test_phases.PhaseTest().sequence()
        self.assertEqual(
            ytw_phases.detect(seq)["phases"],
            ytw_bend_dwell_trial.detect(seq)["phases"],
        )

    def test_straight_arm_rest_not_w(self):
        """T 뒤 양팔을 편 휴식은 굽힘 증가 조건을 충족하지 않는다."""
        seq = test_phases.PhaseTest().sequence()
        seq[6:14, 15] = [-20, 130]
        seq[6:14, 16] = [20, 130]
        seq[27:, 15] = [-20, 130]
        seq[27:, 16] = [20, 130]
        seq[27:, 13] = [-20, 65]
        seq[27:, 14] = [20, 65]
        self.assertNotIn(
            "W_attempt", ytw_bend_dwell_trial.detect(seq)["phases"]
        )

    def test_missing_arms_not_scored(self):
        """팔 누락 자료에서 보간한 W·완전 구간을 생성하지 않는다."""
        seq = test_phases.PhaseTest().sequence()
        seq[:, [13, 14, 15, 16]] = np.nan
        result = ytw_bend_dwell_trial.detect(seq)
        self.assertNotIn("W_attempt", result["phases"])
        self.assertFalse(result["complete_attempt_sequence"])


if __name__ == "__main__":
    unittest.main()
