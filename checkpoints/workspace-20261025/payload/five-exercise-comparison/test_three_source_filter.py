"""검색 추천의 다른 주제·운동이 원본 다운로드와 점수 자료로 이어지지 않게 검사한다."""

import unittest

import three_external_discovery


class SourceFilterTest(unittest.TestCase):
    """이번 검색에서 실제 섞였던 비운동·다른 프레스·다른 종목을 검사한다."""

    def test_irrelevant_recommendations_do_not_pass_download_gate(self) -> None:
        """영상 제목은 종목 후보 필터로만 쓰고 오류 등급으로 쓰지 않는다."""
        for exercise, title, expected in [
            ("pushups", "A reason the pilot won't stay lit", False),
            ("pushups", "The Push Up and Common Faults", True),
            ("shoulder", "How to Use the Chest Press Machine", False),
            ("shoulder", "Standing Dumbbell Shoulder Press", True),
            ("shoulder", "How To PROPERLY Lateral Raise", False),
            ("squats", "Hip Shift in Deadlift", False),
            ("squats", "Fixing a Twist During Squats", True),
        ]:
            with self.subTest(title=title):
                self.assertEqual(
                    three_external_discovery.metadata_matches(exercise, title),
                    expected,
                )
