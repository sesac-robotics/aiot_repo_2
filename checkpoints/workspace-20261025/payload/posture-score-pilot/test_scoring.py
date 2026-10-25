"""반복 경계ㆍ관절 누락ㆍ입력 기하 계산을 검사한다."""

import json
from pathlib import Path
import unittest
import numpy as np
from scoring import RepetitionTracker, angle, measure

RULES = json.loads((Path(__file__).parent / "rules.json").read_text())


class ScoringTests(unittest.TestCase):
    """관절 누락과 미완료 동작이 점수로 집계되지 않는지 확인한다."""

    def feed(self, tracker, angles, start=0):
        """0.1초 간격의 도 단위 입력을 추적기에 전달하고 종료 초를 반환한다."""
        for i, p in enumerate(angles):
            tracker.update(
                start + i * 0.1,
                (
                    {"primary": p, "torso": 20, "alignment": 178}
                    if p is not None
                    else None
                ),
            )
        return start + len(angles) * 0.1

    def test_complete_and_incomplete(self):
        """완결 반복 한 개만 집계하고 영상 끝의 하강은 집계하지 않는다."""
        t = RepetitionTracker("squat", RULES)
        seq = (
            [175] * 5
            + list(np.linspace(170, 90, 12))
            + list(np.linspace(90, 175, 12))
            + [175] * 5
        )
        end = self.feed(t, seq)
        self.assertEqual(len(t.results), 1)
        self.assertTrue(0 <= t.results[0]["score"] <= 100)
        self.feed(t, [170, 150, 130, 110], end)
        self.assertEqual(len(t.results), 1)
        self.assertTrue(t.active)

    def test_two_repetitions(self):
        """연속된 두 반복을 중복 없이 시간 순서대로 집계한다."""
        tracker = RepetitionTracker("pushup", RULES)
        cycle = (
            [175] * 5
            + list(np.linspace(170, 85, 15))
            + list(np.linspace(85, 175, 15))
            + [175] * 5
        )
        self.feed(tracker, cycle + cycle)
        self.assertEqual([r["number"] for r in tracker.results], [1, 2])
        self.assertLess(tracker.results[0]["end"], tracker.results[1]["start"])

    def test_missing_cancels_rep(self):
        """관절이 연속 누락되면 그 전후를 이어 반복으로 만들지 않는다."""
        t = RepetitionTracker("pushup", RULES)
        self.feed(
            t, [175] * 5 + [140, 120, 90] + [None] * 5 + [120, 140] + [175] * 5
        )
        self.assertEqual(t.results, [])
        self.assertGreaterEqual(t.discarded, 1)

    def test_no_start_top(self):
        """영상 중간의 굽힌 자세에서 시작한 동작은 집계하지 않는다."""
        t = RepetitionTracker("squat", RULES)
        self.feed(t, [100, 120, 140] + [175] * 5)
        self.assertEqual(t.results, [])

    def test_geometry_missing(self):
        """누락 관절은 None으로, 직선은 180도로 측정한다."""
        self.assertAlmostEqual(
            angle(np.array([0, 0]), np.array([1, 0]), np.array([2, 0])), 180
        )
        self.assertIsNone(measure({}, "left", "squat"))


if __name__ == "__main__":
    unittest.main()
