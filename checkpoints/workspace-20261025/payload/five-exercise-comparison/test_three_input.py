"""추정 관절이 화면 밖 신체의 실제 관측으로 채택되는 회귀를 검사한다."""

import unittest
from unittest.mock import patch

import numpy as np

import three_input
import three_items


class FramingInputTest(unittest.TestCase):
    """수동 촬영 정책의 원본 식별·점수 독립성·항목 보존을 검사한다."""

    def setUp(self) -> None:
        """화면 내부에 발 좌표까지 추정된 잘림 입력을 준비한다."""
        self.manifest = dict(
            id="cropped",
            video_sha256="a" * 64,
            segment=dict(start_frame=3, end_frame=32, fps=15),
        )
        self.policy = dict(
            records=[
                dict(
                    **self.manifest,
                    view="side",
                    unobservable_joints=[27, 28, 29, 30, 31, 32],
                )
            ]
        )
        self.sequence = np.ones((30, 33, 2)) * 100
        self.sequence[:, 11:13] = (100, 100)
        self.sequence[:, 13:15] = (120, 130)
        self.sequence[:, 15:17] = (100, 160)
        self.sequence[:, 23:25] = (180, 110)

    def test_cropped_feet_hold_only_dependent_pushup_items(self) -> None:
        """높은 신뢰도의 가짜 발도 몸통·지지 총점에 사용하지 않는다."""
        with patch("three_input.audit.read_json", return_value=self.policy):
            result = three_input.apply_framing(
                self.sequence, self.manifest, "side"
            )
        notes = three_items.describe("pushups", result, None)
        self.assertEqual(
            [n["eligible"] for n in notes], [False, True, True, False]
        )
        self.assertTrue(np.isfinite(self.sequence).all())

    def test_wrong_source_fails_instead_of_reusing_manual_policy(self) -> None:
        """ID가 같아도 다른 해시·구간이면 촬영 정책을 전용하지 않는다."""
        with patch("three_input.audit.read_json", return_value=self.policy):
            with self.assertRaises(ValueError):
                three_input.apply_framing(
                    self.sequence,
                    dict(self.manifest, video_sha256="b" * 64),
                    "side",
                )

    def test_unlisted_shoulder_input_keeps_feet_independent(self) -> None:
        """목록 밖 숄더 프레스는 이 푸시업의 수동 정책으로 제한하지 않는다."""
        with patch("three_input.audit.read_json", return_value=self.policy):
            result = three_input.apply_framing(
                self.sequence, dict(self.manifest, id="shoulder"), "primary"
            )
        np.testing.assert_array_equal(result, self.sequence)

    def test_peak_crop_holds_reach_without_discarding_lowering(self) -> None:
        """상단 손 잘림을 확인해도 보이는 하단·몸통 항목은 제거하지 않는다."""
        self.policy["records"][0]["criterion_holds"] = ["overhead_reach"]
        notes = [
            dict(item=name, eligible=True, holds=[])
            for name in [
                "overhead_reach",
                "lower_return",
                "bilateral_timing",
                "frontal_trunk_hold",
            ]
        ]
        with patch("three_input.audit.read_json", return_value=self.policy):
            result = three_input.apply_observation(notes, self.manifest, "side")
        self.assertEqual(
            [n["eligible"] for n in result], [False, True, True, True]
        )
        self.assertTrue(notes[0]["eligible"])
