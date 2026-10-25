"""W 실험의 미검출·경계 접촉·대응 표본 비교의 의미를 검사한다."""

import unittest
from w_trial_audit import interval_metrics, matched_summary


class TrialAuditTests(unittest.TestCase):
    """경계만 접한 검출과 관측 탈락을 성공·오차 개선으로 숨기지 않는다."""

    def test_touch_is_not_strict_match(self):
        """한 프레임 접촉은 겹침이지만 핵심 위치 검출 성공이 아니다."""
        result = interval_metrics([10, 20], [20, 40])
        self.assertTrue(result["overlap"])
        self.assertFalse(result["strict_core_match"])
        self.assertAlmostEqual(result["core_coverage"], 1 / 11)

    def test_missing_and_invalid_intervals(self):
        """미검출은 실패로 기록하고 역순 구간은 거부한다."""
        self.assertFalse(interval_metrics([10, 20], None)["strict_core_match"])
        self.assertTrue(
            interval_metrics([10, 20], [12, 18])["strict_core_match"]
        )
        with self.assertRaises(ValueError):
            interval_metrics([20, 10], None)

    def test_comparison_uses_matched_cases(self):
        """크롭에서 사라진 어려운 사례를 MAE 개선으로 세지 않는다."""
        rows = [
            dict(
                id="a",
                draft=0,
                baseline_manual=dict(grade=2),
                roi_manual=dict(grade=None),
            ),
            dict(
                id="b",
                draft=1,
                baseline_manual=dict(grade=1),
                roi_manual=dict(grade=1),
            ),
            dict(
                id="unknown",
                draft=None,
                baseline_manual=dict(grade=2),
                roi_manual=dict(grade=0),
            ),
        ]
        result = matched_summary(rows)
        self.assertEqual(result["paired_ids"], ["b"])
        self.assertEqual(result["baseline"]["paired_MAE_grade"], 0)
        self.assertEqual(result["lost_assessment_ids"], ["a"])
        self.assertEqual(result["roi"]["grades_on_unknown"], 1)


if __name__ == "__main__":
    unittest.main()
