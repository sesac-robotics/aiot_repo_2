"""항목 간 상쇄와 미관측이 총점 정확도로 오인되지 않는지 검사한다."""

import unittest
from item_error_audit import compare_items


class ItemErrorAuditTests(unittest.TestCase):
    """실제 라벨 없이 배점 오차의 합산·상쇄·누락 계약을 검사한다."""

    def test_opposite_item_errors_do_not_disappear(self) -> None:
        """반대 부호 항목 오차는 총점과 별도로 절대 오차에 남아야 한다."""
        result = compare_items([0, 2, 0, 2], [0, 1, 1, 2], [40, 25, 20, 15])
        self.assertEqual(result["absolute_total_error_points"], 2.5)
        self.assertEqual(result["paired_item_absolute_error_sum_points"], 22.5)
        self.assertEqual(result["cancellation_points"], 20)

    def test_unknown_item_does_not_create_total(self) -> None:
        """미관측 항목이 있으면 총점 오차·상쇄량을 계산하지 않아야 한다."""
        result = compare_items([2, 2], [2, None], [60, 40])
        self.assertIsNone(result["absolute_total_error_points"])
        self.assertIsNone(result["cancellation_points"])
        self.assertEqual(result["paired_weight_points"], 60)

    def test_invalid_grade_fails(self) -> None:
        """범위 밖 등급을 유효한 점수 차이로 합산하지 않아야 한다."""
        with self.assertRaises(ValueError):
            compare_items([2], [3], [100])


if __name__ == "__main__":
    unittest.main()
