"""W 부분 성분 규칙의 정상·실패·가림 처리를 확인한다."""

import unittest

import numpy as np

import w_component_trial


class ComponentTests(unittest.TestCase):
    """화면 모양의 알려진 합성 사례로 부분 수행 의미를 검증한다."""

    def sequence(self) -> np.ndarray:
        """몸통 100픽셀, 팔꿈치 90도 굽힘의 5표본을 반환한다."""
        seq = np.zeros((5, 33, 2), dtype=float)
        seq[:, 11], seq[:, 12] = [20, 0], [-20, 0]
        seq[:, 23], seq[:, 24] = [20, 100], [-20, 100]
        seq[:, 13], seq[:, 14] = [70, 0], [-70, 0]
        seq[:, 15], seq[:, 16] = [70, -50], [-70, -50]
        return seq

    def test_bent_low_forearms_preserve_partial_components(self) -> None:
        """양팔 굽힘을 수행했지만 손 방향을 놓친 경우 부분 수행이다."""
        seq = self.sequence()
        seq[:, 15], seq[:, 16] = [70, 50], [-70, 50]
        result = w_component_trial.evaluate(seq, 0, 4)
        self.assertEqual(result["baseline_grade"], 0)
        self.assertEqual(result["grade"], 1)

    def test_straight_arms_stay_zero(self) -> None:
        """단순 팔 펼침은 W의 부분 굽힘으로 오인하지 않는다."""
        seq = self.sequence()
        seq[:, 15], seq[:, 16] = [120, 0], [-120, 0]
        self.assertEqual(w_component_trial.evaluate(seq, 0, 4)["grade"], 0)

    def test_overhead_folded_y_stays_zero(self) -> None:
        """팔꿈치가 크게 올라간 굽힌 Y를 W 부분 수행으로 바꾸지 않는다."""
        seq = self.sequence()
        seq[:, 13], seq[:, 14] = [40, -50], [-40, -50]
        seq[:, 15], seq[:, 16] = [0, -80], [0, -80]
        self.assertEqual(w_component_trial.evaluate(seq, 0, 4)["grade"], 0)

    def test_missing_wrist_stays_unknown(self) -> None:
        """미관측을 부분 수행 1로 채우지 않는다."""
        seq = self.sequence()
        seq[:2, 15] = np.nan
        self.assertIsNone(w_component_trial.evaluate(seq, 0, 4)["grade"])

    def test_good_shape_is_unchanged(self) -> None:
        """기존 정상 2등급의 의미와 경계를 유지한다."""
        self.assertEqual(
            w_component_trial.evaluate(self.sequence(), 0, 4)["grade"], 2
        )
