"""YTW의 투영 규칙이 비관측을 채우거나 등급을 정답으로 읽지 않게 한다."""

import unittest

import numpy as np

from ytw_projected_rule_trial import grades, fit, predict


class ProjectedRuleTests(unittest.TestCase):
    """화면 특징·명시 경계·부분 총점 계약을 검사한다."""

    def test_y_requires_reach_and_extension(self):
        """Y 높이가 충분해도 팔 굽힘이 크면 정상 등급을 주지 않는다."""
        x = np.array(
            [[145, 145, 145, 160, 160, 160], [145, 145, 145, 90, 90, 90]]
        )
        self.assertEqual(grades(x, (135, 95, 150, 100), 0).tolist(), [2, 0])

    def test_hinge_sway_lowers_projection_grade(self):
        """전경 대리 비율이 같아도 좌우 흔들림이 크면 낮게 판정한다."""
        x = np.array([[2, 2, 2, 0, 0, 2], [2, 2, 2, -25, 0, 25]])
        self.assertEqual(grades(x, (1.8, 1, 10, 25), 3).tolist(), [2, 0])

    def test_unseen_projection_is_not_imputed(self):
        """가려진 관절에서 생긴 NaN을 정상값으로 바꾸지 않는다."""
        with self.assertRaises(ValueError):
            grades(np.full((1, 6), np.nan), (135, 95, 150, 100), 0)

    def test_held_role_cannot_choose_thresholds(self):
        """검증 역할이 규칙 경계 선택에 들어오면 거부한다."""
        with self.assertRaises(ValueError):
            fit([dict(split="validation")])

    def test_missing_stage_cannot_get_total(self):
        """단계가 하나라도 관측되지 않으면 총점을 만들지 않는다."""
        row = dict(
            id="x",
            person="p",
            levels=[None] * 4,
            items=[dict(eligible=False, feature=None)] * 4,
        )
        result = predict({}, row)
        self.assertIsNone(result["ai_score"])
        self.assertIsNone(result["rule_score"])


if __name__ == "__main__":
    unittest.main()
