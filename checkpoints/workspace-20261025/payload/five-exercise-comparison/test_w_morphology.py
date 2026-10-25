"""W 기준의 정상·펴진 팔·낮은 손·가림·좌우 차이를 검사한다."""

import unittest

import numpy as np

import w_morphology


class WMorphologyTests(unittest.TestCase):
    """점수 라벨과 무관한 합성 화면 관절로 규칙의 경계를 검사한다."""

    def sequence(self) -> np.ndarray:
        """몸통 100픽셀, 양팔 90도 굽힘과 위쪽 손인 5장 배열을 반환한다."""
        seq = np.zeros((5, 33, 2), dtype=float)
        seq[:, 11], seq[:, 12] = [20, 0], [-20, 0]
        seq[:, 23], seq[:, 24] = [20, 100], [-20, 100]
        seq[:, 13], seq[:, 14] = [70, 0], [-70, 0]
        seq[:, 15], seq[:, 16] = [70, -50], [-70, -50]
        return seq

    def test_visible_bilateral_w(self) -> None:
        """양팔 굽힘과 손 위쪽 모양은 2등급을 반환해야 한다."""
        self.assertEqual(
            w_morphology.evaluate(self.sequence(), 0, 4)["grade"], 2
        )

    def test_straight_arms_are_not_w(self) -> None:
        """펴진 팔은 손 높이와 무관하게 정상 W로 평가하지 않아야 한다."""
        seq = self.sequence()
        seq[:, 15], seq[:, 16] = [120, 0], [-120, 0]
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 0)

    def test_low_hands_are_not_w(self) -> None:
        """팔을 굽혀도 양손이 아래에 있으면 W를 충족하지 않아야 한다."""
        seq = self.sequence()
        seq[:, 15], seq[:, 16] = [70, 50], [-70, 50]
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 0)

    def test_one_arm_shape_is_partial(self) -> None:
        """한쪽만 W 모양이면 부분 수행 기준인 1등급이어야 한다."""
        seq = self.sequence()
        seq[:, 15] = [70, 50]
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 1)

    def test_overhead_folded_arms_are_not_w(self) -> None:
        """손이 위에 있어도 팔꿈치가 머리 쪽으로 올라간 굽힌 Y는 W가 아니다."""
        seq = self.sequence()
        seq[:, 13], seq[:, 14] = [40, -50], [-40, -50]
        seq[:, 15], seq[:, 16] = [0, -80], [0, -80]
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 0)

    def test_occluded_wrist_is_unknown(self) -> None:
        """관측되지 않는 손을 0점이나 정상으로 채우지 않아야 한다."""
        seq = self.sequence()
        seq[:2, 15] = np.nan
        self.assertIsNone(w_morphology.evaluate(seq, 0, 4)["grade"])

    def test_screen_rotation_preserves_grade(self) -> None:
        """화면 회전은 몸통 기준 손 높이와 등급을 바꾸지 않아야 한다."""
        seq = self.sequence() @ np.array([[0, -1], [1, 0]])
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 2)

    def test_partial_bend_is_partial_grade(self) -> None:
        """양팔 12도 굽힘은 완전 펴짐과 구별한 부분 등급이어야 한다."""
        seq = self.sequence()
        seq[:, 15] = [
            70 + 50 * np.cos(np.deg2rad(12)),
            -50 * np.sin(np.deg2rad(12)),
        ]
        seq[:, 16] = [
            -70 - 50 * np.cos(np.deg2rad(12)),
            -50 * np.sin(np.deg2rad(12)),
        ]
        self.assertEqual(w_morphology.evaluate(seq, 0, 4)["grade"], 1)


if __name__ == "__main__":
    unittest.main()
