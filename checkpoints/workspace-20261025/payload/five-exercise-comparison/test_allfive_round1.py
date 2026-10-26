"""5종 후보의 관측 특징·분할·보존 계약을 검사한다."""

import unittest

import numpy as np

from allfive_round1_features import pitch_features, phase_matches
from allfive_round1_train import fit, predict
from recovery_train import component_groups


class AllFiveRound1Tests(unittest.TestCase):
    """발 기울기·미관측·학습 역할 제한 회귀를 막는다."""

    def test_pitch_translation_and_scale_invariance(self):
        """기울기 특징은 같은 영상의 좌표 이동·배율에 영향을 받지 않는다."""
        seq = np.full((10, 33, 2), np.nan)
        seq[:, 29] = [10, 10]
        seq[:, 31] = [30, 20]
        expected = pitch_features(seq)
        self.assertTrue(np.allclose(expected, pitch_features(seq * 3 + 42)))
        self.assertAlmostEqual(expected[0], 26.565051177)

    def test_missing_feet_are_not_filled(self):
        """발 동시 관측이 80%에 미달하면 새 특징을 만들지 않는다."""
        seq = np.full((10, 33, 2), np.nan)
        seq[:7, 29] = [10, 10]
        seq[:7, 31] = [30, 20]
        self.assertIsNone(pitch_features(seq))

    def test_short_foot_projection_is_not_used(self):
        """3픽셀보다 짧은 발 투영을 신뢰할 만한 기울기로 사용하지 않는다."""
        seq = np.full((10, 33, 2), np.nan)
        seq[:, 29] = [10, 10]
        seq[:, 31] = [11, 10]
        self.assertIsNone(pitch_features(seq))

    def test_label_phase_cannot_be_attached_to_another_stage(self):
        """W 등급을 T 구간 특징에 붙이지 않는다."""
        frames = np.arange(10, 110, 10)
        wrong = dict(start_index=1, end_index=3, anchor_index=2)
        correct = dict(start_index=6, end_index=8, anchor_index=7)
        self.assertFalse(phase_matches(wrong, frames, [70, 95]))
        self.assertTrue(phase_matches(correct, frames, [70, 95]))
        self.assertIsNone(pitch_features(np.full((5, 33, 2), np.nan)))

    def test_validation_labels_cannot_calibrate(self):
        """검증 행을 후보 선택 함수에 넣으면 즉시 거부한다."""
        with self.assertRaises(ValueError):
            fit([dict(split="validation")])

    def test_unavailable_does_not_create_a_score(self):
        """관측하지 못한 네 항목에 임의 점수나 만점을 채우지 않는다."""
        row = dict(
            id="unseen",
            person="a",
            exercise="ytw",
            criterion_ids=["Y", "T", "W", "hinge"],
            levels=[None] * 4,
            weights=[30, 25, 25, 20],
            notes=[
                dict(item=i, bank="front", eligible=False, feature=None)
                for i in ["Y", "T", "W", "hinge"]
            ],
        )
        result = predict({}, row)
        self.assertIsNone(result["ai_score"])
        self.assertIsNone(result["rule_score"])
        self.assertIsNone(result["target_score"])

    def test_same_recording_group_cannot_cross_roles(self):
        """실제 원본 SHA가 달라도 연결된 녹화 그룹의 분할 누출을 막는다."""
        rows = [
            dict(
                person="a",
                sources={"x"},
                recording_groups=["same"],
                split="train",
            ),
            dict(
                person="b",
                sources={"y"},
                recording_groups=["same"],
                split="validation",
            ),
        ]
        with self.assertRaises(ValueError):
            component_groups(rows)


if __name__ == "__main__":
    unittest.main()
