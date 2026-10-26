"""영상 시차 부호·정지 영상·미승인 시점 결합 방지를 검사한다."""

import unittest

import numpy as np

from multiview_squat_sync import lag_scores
from multiview_squat_trial import validate_alignment


class AlignmentTests(unittest.TestCase):
    """자동 상관과 직접 승인·추출 계약이 서로 대체되지 않는지 확인한다."""

    def test_positive_lag_means_later_front_frame(self):
        """정면 지연 7프레임을 같은 부호로 복원하는지 확인한다."""
        rng = np.random.default_rng(42)
        side = rng.normal(size=120)
        front = np.r_[np.zeros(7), side[:-7]]
        self.assertEqual(lag_scores(side, front)[0]["lag_frames"], 7)

    def test_constant_trace_produces_no_alignment_candidate(self):
        """움직임 없는 영상에 시차를 발명하지 않는지 확인한다."""
        self.assertEqual(lag_scores([0] * 40, [0] * 40), [])

    def test_high_correlation_cannot_replace_direct_review(self):
        """직접 검토 승인이 없는 시점 결합을 거부하는지 확인한다."""
        frames = np.arange(20)
        with self.assertRaises(ValueError):
            validate_alignment(dict(lag_frames=3), frames, frames + 3)

    def test_native_rounding_error_one_frame_is_explicitly_allowed(self):
        """실제 10Hz 추출의 반올림 오차만 허용하는지 확인한다."""
        decision = dict(
            paired_interval_approved_by_direct_samples=True, lag_frames=13
        )
        side = np.array([12, 14, 15, 16, 18, 20])
        front = np.array([25, 26, 28, 30, 31, 32])
        validate_alignment(decision, side, front)

    def test_drift_or_repeated_frame_rejected(self):
        """과도한 시차와 중복 프레임을 입력 실패로 처리하는지 확인한다."""
        decision = dict(
            paired_interval_approved_by_direct_samples=True, lag_frames=13
        )
        side = np.arange(10)
        with self.assertRaises(ValueError):
            validate_alignment(decision, side, side + 16)
        with self.assertRaises(ValueError):
            validate_alignment(decision, side, np.full(10, 13))


if __name__ == "__main__":
    unittest.main()
