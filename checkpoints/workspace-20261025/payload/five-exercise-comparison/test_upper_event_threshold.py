"""사건 경계 선택의 클래스 균형·범위·동률 동작을 검사한다."""

import unittest
from upper_event_nested_trial import choose_threshold


class ThresholdTest(unittest.TestCase):
    """빈 데이터·범위 오류를 거부하고 내부 예측만으로 경계를 정한다."""

    def test_internal_balanced_error(self):
        self.assertEqual(
            choose_threshold(
                [
                    dict(target=0, probability=0.1),
                    dict(target=1, probability=0.4),
                ]
            ),
            0.25,
        )

    def test_tie_prefers_higher_threshold(self):
        self.assertEqual(
            choose_threshold(
                [dict(target=0, probability=0), dict(target=1, probability=1)]
            ),
            0.75,
        )

    def test_invalid_inputs(self):
        for rows in [[], [dict(target=1, probability=2)]]:
            with self.assertRaises(ValueError):
                choose_threshold(rows)


if __name__ == "__main__":
    unittest.main()
