"""골반 대리값·결측·학습 폴드 비중의 의미 있는 경계를 검사한다."""

import unittest

import numpy as np

import plank_rule_trial
from plank_balance_trial import fitting_weights


class PlankTrialTests(unittest.TestCase):
    """접촉 확정 없이 기하·결측·참여자 비중의 계약을 검증한다."""

    def sequence(self) -> np.ndarray:
        """몸통 길이 120픽셀인 수평 전신 지지의 10표본 좌표를 반환한다."""
        seq = np.full((10, 33, 2), np.nan)
        for indices in [
            (11, 13, 15, 23, 25, 27, 31),
            (12, 14, 16, 24, 26, 28, 32),
        ]:
            for i, point in zip(
                indices,
                [
                    (300, 100),
                    (300, 180),
                    (330, 180),
                    (180, 100),
                    (115, 105),
                    (50, 110),
                    (50, 180),
                ],
                strict=True,
            ):
                seq[:, i] = point
        return seq

    def test_scale_translation_preserves_ratios(self) -> None:
        """배율·이동만 바뀐 동일 자세는 같은 등급이어야 한다."""
        seq = self.sequence()
        self.assertEqual(
            plank_rule_trial.evaluate(seq)["levels"],
            plank_rule_trial.evaluate(seq * 2 + [40, 80])["levels"],
        )

    def test_low_hip_is_not_full_support(self) -> None:
        """골반이 팔꿈치–발끝 선 가까이 내려가면 지지 만점을 주지 않는다."""
        seq = self.sequence()
        seq[:, [23, 24], 1] = 175
        self.assertEqual(plank_rule_trial.evaluate(seq)["levels"][2], 0)

    def test_missing_foot_is_not_filled(self) -> None:
        """양쪽 발끝이 없으면 새 지지 항목과 총점 모두 보류한다."""
        seq = self.sequence()
        seq[:, [31, 32]] = np.nan
        result = plank_rule_trial.evaluate(seq)
        self.assertIsNone(result["score"])
        self.assertIn("leg_and_body_support", result["hold_items"])

    def test_class_and_person_mass(self) -> None:
        """반복 정상 시범이 클래스·참여자 전체 가중치를 늘리지 않는다."""
        weights = fitting_weights(
            [dict(score=100)] * 3 + [dict(score=40)], ["A", "A", "B", "C"]
        )
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
