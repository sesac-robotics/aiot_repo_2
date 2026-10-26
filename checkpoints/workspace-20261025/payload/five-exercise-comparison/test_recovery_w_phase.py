"""T 뒤 복귀가 W보다 먼저 등장하는 단계 선택 회귀를 검사한다."""

import unittest
from unittest.mock import patch
import numpy as np
from recovery_w_phase import detect


class WPhaseTests(unittest.TestCase):
    """뒤의 팔 굽힘 선택과 완전 단계 보존·없는 수행을 만들지 않음을 검사한다."""

    def fixture(self):
        seq = np.zeros((18, 33, 2))
        seq[:, 11] = [-10, 0]
        seq[:, 12] = [10, 0]
        seq[:, 23] = [-10, 100]
        seq[:, 24] = [10, 100]
        seq[:, 13] = [-20, 25]
        seq[:, 14] = [20, 25]
        seq[:, 15] = [-30, 50]
        seq[:, 16] = [30, 50]
        seq[:5, 13] = [-50, 0]
        seq[:5, 14] = [50, 0]
        seq[:5, 15] = [-100, 0]
        seq[:5, 16] = [100, 0]
        base = dict(
            partial_fallback_used=True,
            complete_attempt_sequence=False,
            phases=dict(
                T_attempt=dict(start_index=0, end_index=4),
                W_attempt=dict(start_index=5, end_index=7),
            ),
        )
        return seq, base

    def test_selects_later_bend_instead_of_first_return(self):
        seq, base = self.fixture()
        seq[10:13, 13] = [-30, 10]
        seq[10:13, 14] = [30, 10]
        seq[10:13, 15] = [-20, -20]
        seq[10:13, 16] = [20, -20]
        with patch(
            "recovery_w_phase.ytw_partial_trial.detect", return_value=base
        ):
            result = detect(seq)
        self.assertTrue(result["w_refined"])
        self.assertGreaterEqual(result["phases"]["W_attempt"]["start_index"], 9)
        self.assertFalse(result["complete_attempt_sequence"])
        self.assertEqual(result["original_W_attempt"]["start_index"], 5)

    def test_no_bend_keeps_original_attempt(self):
        seq, base = self.fixture()
        with patch(
            "recovery_w_phase.ytw_partial_trial.detect", return_value=base
        ):
            result = detect(seq)
        self.assertFalse(result["w_refined"])
        self.assertEqual(result["phases"]["W_attempt"]["start_index"], 5)

    def test_complete_sequence_is_not_rewritten(self):
        seq, base = self.fixture()
        base["complete_attempt_sequence"] = True
        with patch(
            "recovery_w_phase.ytw_partial_trial.detect", return_value=base
        ):
            result = detect(seq)
        self.assertFalse(result["w_refined"])
        self.assertEqual(result["phases"]["W_attempt"]["start_index"], 5)


if __name__ == "__main__":
    unittest.main()
