"""같은 영상의 정상·오류·변형 구간 분리에서 분할 누출을 검사한다."""

import unittest

from export_review_segments import validate_segments


class SegmentGroupsTest(unittest.TestCase):
    """원본이 같거나 참여자가 같으면 시험 분리를 허용하지 않는다."""

    def test_same_video_different_windows_cannot_cross_split(self) -> None:
        """정상과 오류 시범을 잘라도 원본이 같으면 분할 충돌이어야 한다."""
        a = dict(
            id="good",
            source_sha256="source",
            person_group="actor",
            split="train",
            start_frame=10,
            end_frame_exclusive=100,
        )
        b = dict(
            a, id="bad", split="test", start_frame=120, end_frame_exclusive=210
        )
        with self.assertRaises(ValueError):
            validate_segments([a, b])

    def test_same_person_different_sources_cannot_cross_split(self) -> None:
        """같은 사람의 다른 원본·구도도 학습과 시험에 섞이지 않는다."""
        a = dict(
            id="front",
            source_sha256="s1",
            person_group="actor",
            split="train",
            start_frame=10,
            end_frame_exclusive=100,
        )
        b = dict(a, id="side", source_sha256="s2", split="test")
        with self.assertRaises(ValueError):
            validate_segments([a, b])


if __name__ == "__main__":
    unittest.main()
