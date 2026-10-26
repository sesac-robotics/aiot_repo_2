"""부분 라벨의 총점 판정을 정확도 성공이나 정답 결측 0점으로 세지 않도록 검사한다."""

import unittest

import three_quality_summary


class QualitySummaryTest(unittest.TestCase):
    """전체와 관측 가능한 초안의 분모가 다른 경우를 확인한다."""

    def test_prediction_on_unknown_reference_is_not_a_labeled_success(
        self,
    ) -> None:
        """초안 총점이 없는 만점 예측은 별도 점검 대상으로 분리한다."""
        common = dict(
            target_levels=[None, 2, 2, 2],
            ai_levels=[2, 2, 2, 2],
            rule_levels=[None, 2, 2, 2],
            observations=[dict(item=str(i)) for i in range(4)],
        )
        rows = [
            dict(
                common,
                id="unknown",
                target_score=None,
                ai_score=100,
                rule_score=None,
            ),
            dict(
                common,
                id="observable",
                target_score=100,
                ai_score=None,
                rule_score=None,
                target_levels=[2, 2, 2, 2],
            ),
        ]
        result = three_quality_summary.summarize(rows)
        self.assertEqual(result["reference_total_available"], 1)
        self.assertEqual(
            result["methods"]["ai"]["all_submitted_assessed_fraction"], 0.5
        )
        self.assertEqual(
            result["methods"]["ai"]["reference_observable_assessed_fraction"], 0
        )
        self.assertEqual(
            result["methods"]["ai"][
                "reference_unknown_but_total_predicted_ids"
            ],
            ["unknown"],
        )
