"""판정 가능 표본이 다른 방법의 기준선 비교가 왜곡되지 않도록 검사한다."""

import unittest

import three_pipeline


class ThreeMetricsTest(unittest.TestCase):
    """공통 표본 오차와 전체 입력 판정률을 분리하는지 확인한다."""

    def test_unassessed_hard_example_does_not_inflate_only_baseline_error(
        self,
    ) -> None:
        """어려운 미판정 표본을 기준선에만 포함해 AI 우위를 꾸미지 않는다."""
        rows = [
            dict(
                id="seen",
                target_score=100,
                ai_score=90,
                rule_score=80,
                baseline_score=100,
            ),
            dict(
                id="held",
                target_score=0,
                ai_score=None,
                rule_score=None,
                baseline_score=100,
            ),
        ]
        result = three_pipeline.paired_metrics(rows)
        self.assertEqual(result["paired_records"], 1)
        self.assertEqual(result["metrics"]["baseline_score"]["MAE_points"], 0)
        self.assertEqual(result["metrics"]["ai_score"]["MAE_points"], 10)
        self.assertEqual(
            three_pipeline.metrics(rows, "ai_score")["assessed_fraction"], 0.5
        )
